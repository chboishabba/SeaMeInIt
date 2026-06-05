from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from smii.seams import (
    CutTopologyReceipt,
    MetricCorrectionEntry,
    MetricCorrectionReceipt,
    SolverPromotionReceipt,
)
from scripts.unwrap_panels import (
    PanelPatch,
    _branch_spoke_split_parent_panels,
    _face_edges,
    _failure_relief_split_parent_panels,
    _failure_relief_variant_id,
    _serialization_failure_field_receipt,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_mesh(path: Path) -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    np.savez(path, vertices=vertices, faces=faces)


def test_branch_spoke_split_preserves_branch_and_parent_faces() -> None:
    faces = (
        (0, 1, 2),
        (0, 2, 3),
        (4, 5, 6),
        (4, 6, 7),
    )
    panel = PanelPatch(
        vertices=tuple(range(8)),
        edges=_face_edges(faces),
        faces=faces,
    )

    split = _branch_spoke_split_parent_panels(panel, [0], ring_depth=1)

    assert len(split) == 2
    assert sorted(face for patch in split for face in patch.faces) == sorted(faces)
    assert any(set(patch.faces) == {(0, 1, 2), (0, 2, 3)} for patch in split)


def test_failure_relief_split_preserves_parent_faces() -> None:
    faces = (
        (0, 1, 2),
        (2, 1, 3),
        (2, 3, 4),
        (4, 3, 5),
    )
    panel = PanelPatch(
        vertices=tuple(range(6)),
        edges=_face_edges(faces),
        faces=faces,
    )
    failure_field = {
        "candidate_relief_paths": [
            {
                "failure_face_indices": [2, 3],
                "separates_bad_region": True,
            }
        ]
    }

    split = _failure_relief_split_parent_panels(panel, failure_field)

    assert len(split) == 2
    assert sorted(face for patch in split for face in patch.faces) == sorted(faces)
    assert any(set(patch.faces) == {(2, 3, 4), (4, 3, 5)} for patch in split)


def test_serialization_failure_field_derives_relief_path_from_distortion() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    faces = (
        (0, 1, 2),
        (2, 1, 3),
        (2, 3, 4),
        (4, 3, 5),
    )
    panel = PanelPatch(
        vertices=tuple(range(6)),
        edges=_face_edges(faces),
        faces=faces,
    )
    uv = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 10.0],
            [1.0, 10.0],
        ],
        dtype=float,
    )

    field = _serialization_failure_field_receipt(
        panel_id="P0",
        backend="lscm",
        vertices=vertices,
        panel=panel,
        uv=uv,
        distortion_threshold=0.05,
    )

    assert field["schema_version"] == "smii.serialization_failure_field.v1"
    assert field["high_distortion_faces"] == [2, 3]
    assert field["candidate_relief_paths"]
    path = field["candidate_relief_paths"][0]
    assert path["source"] == "distortion_gradient_path"
    assert path["failure_face_indices"] == [2, 3]
    assert path["separates_bad_region"] is True
    assert path["face_partition_preserves_faces"] is True


def test_serialization_failure_field_derives_multi_path_tree_for_disconnected_islands() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [5.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = (
        (0, 1, 2),
        (2, 1, 3),
        (4, 5, 6),
        (6, 5, 7),
        (8, 9, 10),
        (10, 9, 11),
    )
    panel = PanelPatch(
        vertices=tuple(range(12)),
        edges=_face_edges(faces),
        faces=faces,
    )
    uv = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [40.0, 0.0],
            [50.0, 0.0],
            [40.0, 10.0],
            [50.0, 10.0],
        ],
        dtype=float,
    )

    field = _serialization_failure_field_receipt(
        panel_id="P0",
        backend="xatlas",
        vertices=vertices,
        panel=panel,
        uv=uv,
        distortion_threshold=0.05,
    )

    assert field["failure_face_components"] == [[0, 1], [4, 5]]
    assert len(field["candidate_relief_paths"]) == 2
    assert _failure_relief_variant_id(field) == "failure_relief_tree"

    split = _failure_relief_split_parent_panels(panel, field)

    assert len(split) == 3
    assert sorted(face for patch in split for face in patch.faces) == sorted(faces)
    assert any(set(patch.faces) == {(0, 1, 2), (2, 1, 3)} for patch in split)
    assert any(set(patch.faces) == {(8, 9, 10), (10, 9, 11)} for patch in split)


def _write_nonplanar_mesh(path: Path) -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=int,
    )
    np.savez(path, vertices=vertices, faces=faces)


def _write_seam_edges(path: Path) -> None:
    np.savez_compressed(path, seam_edges=np.empty((0, 2), dtype=int))


def _write_fabric_profile(
    path: Path,
    *,
    fabric_id: str = "test_woven",
    s_parallel: float = 0.10,
    s_perp: float = 0.05,
    s_shear: float = 0.05,
    allow_bias: bool = False,
) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "fabric_id": fabric_id,
                "description": "Test low-compliance woven profile.",
                "compliance": {
                    "s_parallel": s_parallel,
                    "s_perp": s_perp,
                    "s_shear": s_shear,
                },
                "mdl_modifiers": {
                    "seam_length": 1.0,
                    "seam_count": 1.0,
                    "panel_count": 1.0,
                },
                "constraints": {
                    "max_grain_rotation_per_panel_deg": 10,
                    "allow_bias": allow_bias,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_labeled_seam_edges(path: Path) -> None:
    np.savez_compressed(
        path,
        seam_edges=np.empty((0, 2), dtype=int),
        face_labels=np.array([0, 1], dtype=int),
    )


def _write_labeled_typed_branch_seam_edges(path: Path) -> None:
    np.savez_compressed(
        path,
        seam_edges=np.array([[0, 2], [0, 1], [0, 3]], dtype=int),
        face_labels=np.array([0, 1], dtype=int),
    )


def _write_labeled_nonplanar_branch_mesh(path: Path) -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=int)
    np.savez(path, vertices=vertices, faces=faces)


def _write_connected_square_mesh(path: Path) -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    np.savez(path, vertices=vertices, faces=faces)


def _write_solver_receipt(
    path: Path,
    seam_edges_path: Path,
    *,
    promotion: int = 1,
    panels_are_disks: bool = True,
) -> None:
    SolverPromotionReceipt(
        seam_cost_receipt_hash="seam-cost-receipt-sha256",
        solver_mode="shortest_path",
        anchor_count=2,
        anchor_source="field_minima",
        connected_component_count=1,
        anchor_fallback_used=False,
        seam_edge_count=0,
        seam_vertex_count=0,
        total_seam_cost=0.0,
        panel_count=2,
        panels_are_disks=panels_are_disks,
        seam_hash=_sha256_file(seam_edges_path),
        promotion=promotion,
        blocked_consumers=[],
    ).to_json(path)


def _write_cut_topology_receipt(
    path: Path,
    *,
    solver_receipt_path: Path,
    seam_edges_path: Path,
    mesh_path: Path,
    typed_operator_count: int = 1,
) -> None:
    CutTopologyReceipt(
        solver_receipt_hash=_sha256_file(solver_receipt_path),
        mesh_hash=_sha256_file(mesh_path),
        seam_edges_hash=_sha256_file(seam_edges_path),
        seam_edge_segment_count=3,
        seam_vertex_count=4,
        seam_connected_component_count=1,
        seam_endpoint_count=3,
        seam_branch_vertex_count=1,
        panel_count=2,
        panel_face_counts=[1, 1],
        panel_boundary_edge_counts=[3, 3],
        panels_are_disks=True,
        typed_dart_count=typed_operator_count,
        typed_gusset_count=0,
        promotion=1,
        blocked_consumers=[],
        cut_topology_blockers=[],
        ordinary_boundary_component_count=0,
        typed_operator_count=typed_operator_count,
        invalid_fragmentation_count=0,
        seam_graph_classifications=["typed_correction_operator"],
    ).to_json(path)


def _write_metric_correction_receipt(
    path: Path,
    *,
    solver_receipt_path: Path,
    cut_topology_receipt_path: Path,
    seam_edges_path: Path,
) -> None:
    MetricCorrectionReceipt(
        solver_receipt_hash=_sha256_file(solver_receipt_path),
        cut_topology_receipt_hash=_sha256_file(cut_topology_receipt_path),
        seam_edges_hash=_sha256_file(seam_edges_path),
        panels_requiring_correction=[0],
        corrections=[
            MetricCorrectionEntry(
                panel_label=0,
                correction_type="dart",
                delta_metric_meaning="local first-fundamental-form relaxation",
                raw_residual=0.04,
                corrected_residual=0.01,
                energy_terms={"shape": 0.01},
                result_state="correctionOk",
                blockers=[],
            )
        ],
        raw_residual_total=0.04,
        corrected_residual_total=0.01,
        residual_gate=0.05,
        promotion=1,
        blocked_consumers=[],
        metric_correction_blockers=[],
    ).to_json(path)


def _write_metric_solver_receipt(
    path: Path,
    seam_edges_path: Path,
    corrections_path: Path,
) -> None:
    SolverPromotionReceipt(
        seam_cost_receipt_hash="seam-cost-receipt-sha256",
        solver_mode="metric_panelization",
        anchor_count=2,
        anchor_source="field_minima",
        connected_component_count=1,
        anchor_fallback_used=False,
        seam_edge_count=0,
        seam_vertex_count=0,
        total_seam_cost=0.0,
        panel_count=2,
        panels_are_disks=True,
        seam_hash=_sha256_file(seam_edges_path),
        promotion=1,
        blocked_consumers=[],
        correction_payload_hash=_sha256_file(corrections_path),
        raw_residual_total=0.4,
        corrected_residual_total=0.15,
        selected_correction_count=1,
    ).to_json(path)


def _run_unwrap(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, "scripts/unwrap_panels.py", *args],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_unwrap_panels_emits_promoted_receipt_from_promoted_solver(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    out_dir = tmp_path / "out"
    panel_receipt_path = out_dir / "panel_unwrap_receipt.json"

    _write_mesh(mesh_path)
    _write_seam_edges(seam_edges_path)
    _write_solver_receipt(solver_receipt_path, seam_edges_path)

    result = _run_unwrap(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-panel-receipt",
        str(panel_receipt_path),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(panel_receipt_path.read_text(encoding="utf-8"))
    assert receipt["solver_receipt_hash"] == _sha256_file(solver_receipt_path)
    assert receipt["panel_count"] == 2
    assert receipt["panels_all_disks"]
    assert receipt["promotion"] == 1
    assert receipt["seam_topology_hash"] == _sha256_file(seam_edges_path)
    assert receipt["unwrap_backend"] == "lscm"
    assert receipt["backend_is_bootstrap"] is False
    assert receipt["serialization_promoted"] is True
    assert receipt["selected_backend_per_panel"] == ["lscm", "lscm"]
    assert receipt["serialization_competition_receipt"]["promotion"] == 1
    assert {
        candidate["backend"]
        for candidate in receipt["serialization_competition_receipt"]["panels"][0]["candidates"]
    } >= {"bootstrap_projection", "lscm", "xatlas"}
    assert receipt["distortion_margin"] >= 0
    assert receipt["panel_unwrap_blockers"] == []
    assert receipt.get("fabric_metric_receipt") is None
    assert receipt["correction_tree_receipt"]["schema_version"] == "smii.correction_tree_receipt.v1"
    assert receipt["correction_tree_receipt"]["promotion"] == 1

    uv_path = out_dir / "panel_uvs.npz"
    assert receipt["uv_hash"] == _sha256_file(uv_path)
    payload = np.load(uv_path)
    assert set(payload.files) == {"panel_0", "panel_1"}


def test_unwrap_panels_prefers_face_labels_when_present(tmp_path: Path) -> None:
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    out_dir = tmp_path / "out"
    panel_receipt_path = out_dir / "panel_unwrap_receipt.json"

    _write_connected_square_mesh(mesh_path)
    _write_labeled_seam_edges(seam_edges_path)
    _write_solver_receipt(solver_receipt_path, seam_edges_path)

    result = _run_unwrap(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-panel-receipt",
        str(panel_receipt_path),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(panel_receipt_path.read_text(encoding="utf-8"))
    assert receipt["panel_count"] == 2
    payload = np.load(out_dir / "panel_uvs.npz")
    assert set(payload.files) == {"panel_0", "panel_1"}


def test_unwrap_panels_blocks_unpromoted_solver_receipt(tmp_path: Path) -> None:
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"

    _write_mesh(mesh_path)
    _write_seam_edges(seam_edges_path)
    _write_solver_receipt(solver_receipt_path, seam_edges_path, promotion=0)

    result = _run_unwrap(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(tmp_path / "out"),
    )

    assert result.returncode != 0
    assert "SolverPromotionReceipt not promoted" in result.stderr


def test_unwrap_panels_reports_topology_not_unwrapper_when_not_disks(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"

    _write_mesh(mesh_path)
    _write_seam_edges(seam_edges_path)
    _write_solver_receipt(
        solver_receipt_path,
        seam_edges_path,
        panels_are_disks=False,
    )

    result = _run_unwrap(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(tmp_path / "out"),
    )

    assert result.returncode != 0
    assert "seam topology is incomplete" in result.stderr
    assert "unwrapper is not the problem" in result.stderr


def test_unwrap_panels_lscm_emits_non_bootstrap_receipt(tmp_path: Path) -> None:
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"

    _write_mesh(mesh_path)
    _write_seam_edges(seam_edges_path)
    _write_solver_receipt(solver_receipt_path, seam_edges_path)

    result = _run_unwrap(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(tmp_path / "out"),
        "--solver",
        "lscm",
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(
        (tmp_path / "out" / "panel_unwrap_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["unwrap_backend"] == "lscm"
    assert receipt["backend_is_bootstrap"] is False
    assert receipt["promotion"] == 1


def test_unwrap_panels_records_distortion_margin_and_blocker(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    out_dir = tmp_path / "out"
    panel_receipt_path = out_dir / "panel_unwrap_receipt.json"

    _write_nonplanar_mesh(mesh_path)
    _write_seam_edges(seam_edges_path)
    _write_solver_receipt(solver_receipt_path, seam_edges_path)

    result = _run_unwrap(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-panel-receipt",
        str(panel_receipt_path),
        "--max-subdivisions",
        "0",
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(panel_receipt_path.read_text(encoding="utf-8"))
    assert receipt["promotion"] == 0
    assert receipt["distortion_margin"] < 0
    assert "distortion_exceeds_threshold" in receipt["panel_unwrap_blockers"]
    assert "manufacturing" in receipt["blocked_consumers"]
    assert receipt["correction_tree_receipt"]["promotion"] == 1


def test_unwrap_panels_reports_fabric_relative_metric_gate(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    fabric_path = tmp_path / "fabric.json"
    out_dir = tmp_path / "out"
    panel_receipt_path = out_dir / "panel_unwrap_receipt.json"

    _write_nonplanar_mesh(mesh_path)
    _write_seam_edges(seam_edges_path)
    _write_solver_receipt(solver_receipt_path, seam_edges_path)
    _write_fabric_profile(fabric_path)

    result = _run_unwrap(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-panel-receipt",
        str(panel_receipt_path),
        "--max-subdivisions",
        "0",
        "--fabric-profile",
        str(fabric_path),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(panel_receipt_path.read_text(encoding="utf-8"))
    fabric_metric = receipt["fabric_metric_receipt"]
    assert fabric_metric["schema_version"] == "smii.fabric_aware_panel_metric.v1"
    assert fabric_metric["fabric_profile"] == "test_woven"
    assert fabric_metric["metric_boundary"] == "generic_uv_distortion_is_only_a_proxy"
    assert fabric_metric["worst_fabric_violation"] > 0.0
    assert "fabric_metric_violation_exceeds_profile" in receipt["panel_unwrap_blockers"]


def test_unwrap_panels_prices_untyped_branch_operator_for_stretch_fabric(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    cut_topology_receipt_path = tmp_path / "cut_topology_receipt.json"
    corrections_path = tmp_path / "corrections.json"
    fabric_path = tmp_path / "fabric.json"
    out_dir = tmp_path / "out"
    panel_receipt_path = out_dir / "panel_unwrap_receipt.json"

    _write_labeled_nonplanar_branch_mesh(mesh_path)
    _write_labeled_typed_branch_seam_edges(seam_edges_path)
    _write_solver_receipt(solver_receipt_path, seam_edges_path)
    _write_cut_topology_receipt(
        cut_topology_receipt_path,
        solver_receipt_path=solver_receipt_path,
        seam_edges_path=seam_edges_path,
        mesh_path=mesh_path,
        typed_operator_count=0,
    )
    corrections_path.write_text(
        json.dumps(
            {
                "panels": [
                    {"panel_label": 0, "corrected_metric_residual": 0.20},
                    {"panel_label": 1, "corrected_metric_residual": 0.18},
                ],
                "selected_corrections": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_fabric_profile(
        fabric_path,
        fabric_id="test_knit",
        s_parallel=1.0,
        s_perp=0.9,
        s_shear=1.3,
        allow_bias=True,
    )

    result = _run_unwrap(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-panel-receipt",
        str(panel_receipt_path),
        "--cut-topology-receipt",
        str(cut_topology_receipt_path),
        "--corrections",
        str(corrections_path),
        "--fabric-profile",
        str(fabric_path),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(panel_receipt_path.read_text(encoding="utf-8"))
    scoring = receipt["correction_operator_scoring_receipt"]
    assert scoring["schema_version"] == "smii.correction_operator_scoring.v1"
    assert scoring["branch_count"] == 1
    assert scoring["typed_branch_count"] == 1
    assert scoring["diagnostic_branch_count"] == 0
    assert scoring["estimated_worst_fabric_violation_after"] < scoring["fabric_violation_before"]
    assert scoring["nodes"][0]["selected_operator"] != "diagnostic_carry"
    realized = receipt["realized_correction_operator_receipt"]
    assert realized["schema_version"] == "smii.realized_correction_operator.v1"
    assert realized["realized_operator_count"] == 2
    assert realized["companion_operator_count"] == 1
    assert realized["operator_families"] == ["gusset_corner", "stretch_zone"]
    assert (
        realized["realized_worst_fabric_violation_after"]
        < scoring["estimated_worst_fabric_violation_after"]
    )
    assert realized["realized_worst_residual_after"] <= realized["residual_gate"]
    assert receipt["fabric_metric_receipt"]["realized_correction_operator"][
        "realized_worst_fabric_violation_after"
    ] == pytest.approx(realized["realized_worst_fabric_violation_after"])
    materialization = receipt["correction_tree_materialization_receipt"]
    assert materialization["schema_version"] == "smii.correction_tree_materialization.v1"
    assert materialization["promotion"] == 1
    assert materialization["materialized_operator_count"] == 2
    assert {entry["materialization_kind"] for entry in materialization["materializations"]} == {
        "backend_hint",
        "inserted_patch",
    }
    gusset_entries = [
        entry
        for entry in materialization["materializations"]
        if entry["operator_family"] == "gusset_corner"
    ]
    assert gusset_entries
    assert gusset_entries[0]["geometry"]["creates_new_chart"] is True
    assert gusset_entries[0]["geometry"]["patch_shape"] == "diamond"
    assert "unmaterialized_correction_operator" not in receipt["panel_unwrap_blockers"]
    assert "operator_materialized_but_serialization_failed" not in receipt["panel_unwrap_blockers"]
    correction_tree = receipt["correction_tree_receipt"]
    assert correction_tree["typed_branch_count"] == 1
    assert correction_tree["diagnostic_branch_count"] == 0
    assert correction_tree["blockers"] == []


def test_unwrap_panels_reports_metric_corrected_residuals(tmp_path: Path) -> None:
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    corrections_path = tmp_path / "corrections.json"
    out_dir = tmp_path / "out"
    panel_receipt_path = out_dir / "panel_unwrap_receipt.json"

    _write_mesh(mesh_path)
    _write_seam_edges(seam_edges_path)
    corrections_path.write_text(
        json.dumps(
            {
                "panels": [
                    {"panel_label": 0, "corrected_metric_residual": 0.05},
                    {"panel_label": 1, "corrected_metric_residual": 0.10},
                ],
                "selected_corrections": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_metric_solver_receipt(solver_receipt_path, seam_edges_path, corrections_path)

    result = _run_unwrap(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-panel-receipt",
        str(panel_receipt_path),
        "--corrections",
        str(corrections_path),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(panel_receipt_path.read_text(encoding="utf-8"))
    assert receipt["per_panel_distortion"] != receipt["per_panel_corrected_residual"]
    assert receipt["per_panel_corrected_residual"] == [0.05, 0.10]
    assert receipt["worst_corrected_residual"] == 0.10
    assert receipt["mean_corrected_residual"] == 0.07500000000000001
    assert receipt["correction_payload_hash"] == _sha256_file(corrections_path)


def test_unwrap_panels_blocks_typed_operator_without_metric_receipt(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    cut_topology_receipt_path = tmp_path / "cut_topology_receipt.json"
    out_dir = tmp_path / "out"
    panel_receipt_path = out_dir / "panel_unwrap_receipt.json"

    _write_connected_square_mesh(mesh_path)
    _write_labeled_typed_branch_seam_edges(seam_edges_path)
    _write_solver_receipt(solver_receipt_path, seam_edges_path)
    _write_cut_topology_receipt(
        cut_topology_receipt_path,
        solver_receipt_path=solver_receipt_path,
        seam_edges_path=seam_edges_path,
        mesh_path=mesh_path,
    )

    result = _run_unwrap(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-panel-receipt",
        str(panel_receipt_path),
        "--cut-topology-receipt",
        str(cut_topology_receipt_path),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(panel_receipt_path.read_text(encoding="utf-8"))
    assert receipt["promotion"] == 0
    assert "missingShapingIntentReceipt" in receipt["panel_unwrap_blockers"]
    assert "missingDeltaMetricMeaning" in receipt["panel_unwrap_blockers"]
    assert "missingPanelUnwrapCompatibility" in receipt["panel_unwrap_blockers"]
    correction_tree = receipt["correction_tree_receipt"]
    assert correction_tree["branch_count"] == 1
    assert correction_tree["typed_branch_count"] == 1
    assert correction_tree["diagnostic_branch_count"] == 0
    assert correction_tree["blockers"] == []


def test_unwrap_panels_promotes_typed_operator_with_metric_receipt(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    cut_topology_receipt_path = tmp_path / "cut_topology_receipt.json"
    metric_correction_receipt_path = tmp_path / "metric_correction_receipt.json"
    out_dir = tmp_path / "out"
    panel_receipt_path = out_dir / "panel_unwrap_receipt.json"

    _write_connected_square_mesh(mesh_path)
    _write_labeled_typed_branch_seam_edges(seam_edges_path)
    _write_solver_receipt(solver_receipt_path, seam_edges_path)
    _write_cut_topology_receipt(
        cut_topology_receipt_path,
        solver_receipt_path=solver_receipt_path,
        seam_edges_path=seam_edges_path,
        mesh_path=mesh_path,
    )
    _write_metric_correction_receipt(
        metric_correction_receipt_path,
        solver_receipt_path=solver_receipt_path,
        cut_topology_receipt_path=cut_topology_receipt_path,
        seam_edges_path=seam_edges_path,
    )

    result = _run_unwrap(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-panel-receipt",
        str(panel_receipt_path),
        "--cut-topology-receipt",
        str(cut_topology_receipt_path),
        "--metric-correction-receipt",
        str(metric_correction_receipt_path),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(panel_receipt_path.read_text(encoding="utf-8"))
    assert receipt["promotion"] == 1
    assert receipt["panel_unwrap_blockers"] == []
    assert receipt["metric_correction_receipt_hash"] == _sha256_file(metric_correction_receipt_path)
    assert receipt["per_panel_corrected_residual"][0] == 0.01
    correction_tree = receipt["correction_tree_receipt"]
    assert correction_tree["branch_count"] == 1
    assert correction_tree["typed_branch_count"] == 1
    assert correction_tree["promotion"] == 1


def test_unwrap_panels_does_not_promote_degenerate_subdivision(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    out_dir = tmp_path / "out"
    panel_receipt_path = out_dir / "panel_unwrap_receipt.json"

    _write_nonplanar_mesh(mesh_path)
    _write_seam_edges(seam_edges_path)
    _write_solver_receipt(solver_receipt_path, seam_edges_path)

    result = _run_unwrap(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-panel-receipt",
        str(panel_receipt_path),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(panel_receipt_path.read_text(encoding="utf-8"))
    assert receipt["promotion"] == 0
    assert receipt["panel_count"] == 1
    assert receipt["subdivision_iterations"] == 0
    assert receipt["serialization_promoted"] is False
    assert receipt["selected_backend_per_panel"] == ["bootstrap_projection"]
    assert receipt["panel_unwrap_blockers"] == [
        "distortion_exceeds_threshold",
        "unresolved_open_boundary",
        "chart_domain_not_backend_serializable",
        "no_serialization_backend_promoted",
        "foldovers_present",
        "diagnostic_only_backend",
        "backend_skipped_invalid_chart_domain",
        "panel_serialization_blocked",
    ]
    candidates = receipt["serialization_competition_receipt"]["panels"][0]["candidates"]
    assert candidates[0]["backend"] == "bootstrap_projection"
    assert "diagnostic_only_backend" in candidates[0]["blockers"]
    for candidate in candidates[1:]:
        assert candidate["blockers"] == ["backend_skipped_invalid_chart_domain"]
        assert candidate["chart_diagnostics"]["backend_serializable"] is False
