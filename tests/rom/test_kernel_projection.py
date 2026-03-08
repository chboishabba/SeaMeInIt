import json

import numpy as np

from smii.rom.aggregation import aggregate_fields, RomSample
from smii.rom.basis import KernelBasis, KernelProjector, load_basis
from smii.rom.constraints import load_constraints
from smii.rom.gates import (
    CouplingRule,
    GateReason,
    RomGate,
    build_gate_from_manifest,
    load_coupling_manifest,
)


def test_kernel_projector_single_and_batch():
    basis = KernelBasis.from_arrays(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        source_mesh="mesh_ref",
        normalization="area-weighted",
    )
    projector = KernelProjector(basis)

    single = projector.project([1.0, 2.0])
    assert single.shape == (3,)
    np.testing.assert_allclose(single, np.array([1.0, 4.0, 3.0]))

    batch = projector.project_batch(np.array([[1.0, 0.5], [0.0, 1.0]]))
    assert batch.shape == (3, 2)
    np.testing.assert_allclose(batch[:, 0], np.array([1.0, 0.0, 1.0]))
    np.testing.assert_allclose(batch[:, 1], np.array([0.5, 2.0, 1.5]))

    encoded = projector.encode(single)
    np.testing.assert_allclose(encoded, np.array([4.0, 11.0]))

    encoded_batch = projector.encode_batch(batch)
    np.testing.assert_allclose(encoded_batch, np.array([[2.0, 2.0], [1.0, 5.5]]))


def test_kernel_projector_encode_round_trip_for_orthonormal_basis():
    basis = KernelBasis.from_arrays(np.eye(3))
    projector = KernelProjector(basis)

    coeffs = np.array([0.2, -0.5, 1.1], dtype=float)
    field = projector.project(coeffs)
    recovered = projector.encode(field)
    np.testing.assert_allclose(recovered, coeffs)


def test_load_basis_round_trip(tmp_path):
    basis = np.eye(2)
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    path = tmp_path / "basis.npz"
    np.savez(path, basis=basis, vertices=vertices, meta={"source_mesh": "canonical"})

    loaded = load_basis(path)
    assert loaded.metadata.vertex_count == 2
    assert loaded.metadata.component_count == 2
    assert loaded.metadata.source_mesh == "canonical"
    np.testing.assert_allclose(loaded.matrix, basis)
    assert loaded.vertices is not None
    np.testing.assert_allclose(loaded.vertices, vertices)


def test_aggregate_fields_computes_stats():
    basis = KernelBasis.from_arrays(np.eye(2))
    projector = KernelProjector(basis)

    samples = [
        RomSample(pose_id="a", coeffs={"T": np.array([1.0, 2.0])}),
        RomSample(pose_id="b", coeffs={"T": np.array([3.0, 4.0])}),
    ]

    agg = aggregate_fields(samples, projector)
    assert agg.sample_count == 2
    assert agg.total_samples == 2
    assert agg.rejection_report.rejection_rate == 0.0
    stats = agg.per_field["T"]
    assert stats.sample_count == 2

    expected_fields = np.array([[1.0, 2.0], [3.0, 4.0]]).T  # shape (2, 2)
    expected_mean = expected_fields.mean(axis=1)
    expected_variance = expected_fields.var(axis=1)

    np.testing.assert_allclose(stats.mean, expected_mean)
    np.testing.assert_allclose(stats.maximum, np.array([3.0, 4.0]))
    np.testing.assert_allclose(stats.variance, expected_variance)


def test_constraint_registry_loading(tmp_path):
    root = tmp_path / "constraints"
    root.mkdir()
    (root / "forbidden_vertices.json").write_text(
        '{"regions": {"axilla": [1, 2], "groin": [3]}}', encoding="utf-8"
    )
    (root / "anchors.json").write_text('{"landmarks": {"sternum": 42}}', encoding="utf-8")
    (root / "symmetry_pairs.json").write_text(
        '{"vertex_pairs": [[5, 6]], "edge_pairs": [{"left": [1, 2], "right": [3, 4]}]}',
        encoding="utf-8",
    )
    (root / "panel_limits.json").write_text(
        '{"default": {"max_primary_panels": 6}, "garment_overrides": {"undersuit": {"max_primary_panels": 4}}}',
        encoding="utf-8",
    )

    registry = load_constraints(root)
    assert registry.is_vertex_forbidden(2)
    assert not registry.is_vertex_forbidden(99)
    assert registry.anchor_vertex("sternum") == 42
    assert registry.symmetry_partner(5) == 6
    panel_limits = registry.panel_limits()
    assert panel_limits["default"].get("max_primary_panels") == 6
    assert registry.panel_limits_for("undersuit").get("max_primary_panels") == 4


def test_rom_gate_blocks_on_threshold():
    gate = RomGate(
        [
            CouplingRule(id="high_shear", description="Shear too high", threshold=0.5, block=True),
            CouplingRule(id="crease", description="Crease risk", threshold=0.1, block=False, severity="warn"),
        ]
    )
    decision = gate.evaluate({"high_shear": 0.6, "crease": 0.2})
    assert not decision.accepted
    assert isinstance(decision.blocking_reasons[0], GateReason)
    assert decision.blocking_reasons[0].id == "high_shear"
    assert len(decision.reasons) == 2


def test_rom_gate_derives_seam_tear_risk_from_observations():
    gate = RomGate(
        [
            CouplingRule(
                id="seam_tear_risk",
                description="Predicted seam/fabric tear risk is too high.",
                threshold=0.7,
                block=True,
            )
        ]
    )

    accepted = gate.evaluate({"shear_hotspot": 0.3, "pressure_hotspot": 0.2})
    assert accepted.accepted
    assert not accepted.reasons

    rejected = gate.evaluate({"shear_hotspot": 0.8, "pressure_hotspot": 0.2})
    assert not rejected.accepted
    assert rejected.blocking_reasons[0].id == "seam_tear_risk"

    rejected_intersection = gate.evaluate({"edge_self_intersection": 1})
    assert not rejected_intersection.accepted
    assert rejected_intersection.blocking_reasons[0].id == "seam_tear_risk"


def test_aggregate_fields_handles_edges_and_gating():
    basis = KernelBasis.from_arrays(np.eye(3))
    projector = KernelProjector(basis)
    edges = [(0, 1), (1, 2)]
    gate = RomGate(
        [CouplingRule(id="blocker", description="blocks on flag", threshold=0.5, block=True)]
    )

    samples = [
        RomSample(
            pose_id="ok",
            coeffs={"T": np.array([1.0, 0.0, 0.0])},
            observations={"blocker": 0.0},
        ),
        RomSample(
            pose_id="bad",
            coeffs={"T": np.array([2.0, 0.0, 0.0])},
            observations={"blocker": 0.6},
        ),
    ]

    agg = aggregate_fields(samples, projector, edges=edges, gate=gate, diagnostics_top_k=1)
    assert agg.sample_count == 1
    assert agg.total_samples == 2
    assert agg.rejection_report.rejected_samples == 1
    assert agg.rejection_report.reasons[0].id == "blocker"
    assert agg.per_edge_field["T"].mean.shape == (2,)
    assert agg.per_edge_field["T"].sample_count == 1
    np.testing.assert_allclose(agg.per_edge_field["T"].maximum, np.array([1.0, 0.0]))
    diagnostics = agg.diagnostics["T"]
    assert diagnostics.vertex_hotspots[0].index in (0, 1, 2)
    assert diagnostics.edge_hotspots[0].edge in edges


def test_aggregate_fields_gates_on_dynamic_seam_tear_risk():
    basis = KernelBasis.from_arrays(np.eye(1))
    projector = KernelProjector(basis)
    gate = RomGate(
        [
            CouplingRule(
                id="seam_tear_risk_dynamic",
                description="fatigue risk too high",
                threshold=0.5,
                block=True,
            )
        ]
    )
    samples = [
        RomSample(
            pose_id=f"pose_{idx}",
            coeffs={"T": np.array([1.0])},
            observations={"shear_hotspot": 0.9},
        )
        for idx in range(6)
    ]

    agg = aggregate_fields(samples, projector, gate=gate)
    assert agg.total_samples == 6
    assert agg.sample_count == 4
    assert agg.rejection_report.rejected_samples == 2
    assert agg.rejection_report.reasons[0].id == "seam_tear_risk_dynamic"
    assert agg.rejection_report.reasons[0].count == 2


def test_aggregate_fields_optional_and_missing_keys():
    basis = KernelBasis.from_arrays(np.eye(2))
    projector = KernelProjector(basis)
    samples = [
        RomSample(pose_id="first", coeffs={"core": np.array([1.0, 0.0])}),
        RomSample(pose_id="second", coeffs={"core": np.array([0.0, 1.0]), "diag": np.array([1.0, 1.0])}),
    ]

    agg = aggregate_fields(
        samples,
        projector,
        field_keys=("core",),
        optional_fields=("diag", "missing"),
        include_observed_fields=False,
    )

    assert agg.per_field["core"].sample_count == 2
    assert agg.per_field["diag"].sample_count == 1
    np.testing.assert_allclose(agg.per_field["diag"].maximum, np.array([1.0, 1.0]))
    assert agg.per_field["missing"].sample_count == 0
    assert np.isnan(agg.per_field["missing"].mean).all()


def test_load_coupling_manifest(tmp_path):
    manifest_path = tmp_path / "couplings.json"
    manifest_payload = {
        "object_map": {"objects": {"body": "canonical"}, "observations": {"forbidden_vertices": "rom.forbidden"}},
        "rules": [
            {"id": "forbidden_vertices", "description": "blocked", "threshold": 1, "block": True},
            {"id": "pressure_hotspot", "description": "warn", "threshold": 0.5, "block": False},
            {"id": "seam_tear_risk", "description": "tear", "threshold": 0.7, "block": True},
            {"id": "seam_tear_risk_dynamic", "description": "tear fatigue", "threshold": 0.8, "block": True},
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    manifest = load_coupling_manifest(manifest_path)
    assert manifest.object_map["objects"]["body"] == "canonical"
    gate = build_gate_from_manifest(manifest)
    decision = gate.evaluate({"forbidden_vertices": 1})
    assert not decision.accepted
    tear_decision = gate.evaluate({"shear_hotspot": 0.8})
    assert not tear_decision.accepted
    assert any(reason.id == "seam_tear_risk" for reason in tear_decision.blocking_reasons)
