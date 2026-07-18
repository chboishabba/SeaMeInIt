from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from smii.meshing import BodyCarrierReceiptV2, load_body_carrier_receipt
from smii.pipelines import authorized_mesh_export
from smii.pipelines.authorized_image_fit import GovernedFitResult
from smii.pipelines.fit_from_images import SMPLXRegressionFrame, SMPLXRegressionResult
from smii.pipelines.refinement_authority import build_receipt, solve_bounded_refinement
from smii.pipelines.refinement_policy import RefinementPolicy


class Row:
    def __init__(self, name: str, weights: tuple[float, ...]) -> None:
        self.name = name
        self.mean = 0.0
        self.std = 1.0
        self.weights = weights


class Report:
    coverage = 1.0

    def visualization_payload(self) -> list[object]:
        return []


def _policy() -> RefinementPolicy:
    return RefinementPolicy.from_effective_config(
        backend="test",
        num_betas=2,
        scale_measurement="height",
        models=(
            Row("height", (1.0, 0.0)),
            Row("width", (0.0, 1.0)),
        ),
        settings={
            "beta_lower": -2.0,
            "beta_upper": 2.0,
            "prior_weight": 0.01,
            "anchor_weight": 0.1,
            "max_beta_shift": 5.0,
            "max_measurement_residual": 5.0,
            "max_residual_degradation": 5.0,
            "abstain_on_warnings": ["WARN:low_view_diversity"],
        },
    )


def _result(tmp_path: Path, *, abstain: bool = False) -> SMPLXRegressionResult:
    anchor = np.zeros(2)
    policy = _policy()
    solution = solve_bounded_refinement(
        np.eye(2),
        np.array([1.0, 0.5]),
        anchor,
        policy,
    )
    warnings = ("WARN:low_view_diversity",) if abstain else ()
    receipt = build_receipt(
        policy=policy,
        measurements={"height": 1.0, "width": 0.5},
        names=("height", "width"),
        anchor_betas=anchor,
        solution=solution,
        warnings=warnings,
        severity="warn" if abstain else "pass",
        input_context={"scale": 1.0, "translation": [0.0, 0.0, 0.0]},
        candidate_context={"scale": 1.1, "translation": [0.0, 0.0, 0.0]},
    )
    selected = solution.betas if receipt.decision == "promote" else anchor
    refinement = GovernedFitResult(
        betas=selected,
        scale=1.1 if receipt.decision == "promote" else 1.0,
        translation=np.zeros(3),
        residual=(
            solution.measurement_residual
            if receipt.decision == "promote"
            else solution.anchor_measurement_residual
        ),
        measurements_used=("height", "width"),
        measurement_report=Report(),  # type: ignore[arg-type]
        refinement_receipt=receipt,
        trust_level="high",
        consistency_status="WARN" if abstain else "PASS",
        consistency_flags=warnings,
    )
    image_path = tmp_path / "source.jpg"
    image_path.write_bytes(b"source-image")
    frame = SMPLXRegressionFrame(
        image_path=image_path,
        betas=anchor,
        body_pose=np.zeros(63),
        global_orient=np.zeros(3),
        transl=np.zeros(3),
        measurements={"height": 1.0, "width": 0.5},
        confidence=0.9,
    )
    return SMPLXRegressionResult(
        betas=anchor,
        body_pose=np.zeros(63),
        global_orient=np.zeros(3),
        transl=np.zeros(3),
        measurements={"height": 1.0, "width": 0.5},
        frames=(frame,),
        measurement_fit=refinement,  # type: ignore[arg-type]
        detector="mediapipe",
        trust_level="high",
        consistency_status="WARN" if abstain else "PASS",
        consistency_flags=warnings,
    )


def _tetra(offset: float) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [offset + 0.0, 0.0, 0.0],
            [offset + 1.0, 0.0, 0.0],
            [offset + 0.0, 1.0, 0.0],
            [offset + 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=np.int32,
    )
    return vertices, faces


def _patch_mesh_generation(monkeypatch: object) -> None:
    raw = _tetra(0.0)
    candidate = _tetra(10.0)

    def create(result: object, *, use_measurement_refinement: bool, **_: object):
        if use_measurement_refinement:
            candidate_betas = np.asarray(result.measurement_fit.betas)  # type: ignore[attr-defined]
            assert not np.allclose(candidate_betas, np.zeros(2))
            return candidate
        return raw

    monkeypatch.setattr(  # type: ignore[attr-defined]
        authorized_mesh_export,
        "_create_body_mesh_from_regression",
        create,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        authorized_mesh_export,
        "repair_body_mesh_for_export",
        lambda vertices, faces: (np.asarray(vertices), np.asarray(faces)),
    )


def test_promoted_candidate_is_selected_repaired_and_receipted(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    _patch_mesh_generation(monkeypatch)
    result = _result(tmp_path)
    output = tmp_path / "person_smplx_body.npz"

    authorized_mesh_export.save_regression_mesh(result, output)

    with np.load(output) as payload:
        np.testing.assert_allclose(payload["vertices"], _tetra(10.0)[0])
    receipt = load_body_carrier_receipt(tmp_path / "body_carrier_receipt.json")
    assert isinstance(receipt, BodyCarrierReceiptV2)
    assert receipt.refinement_decision == "promote"
    assert receipt.canonical_source == "refined_candidate"
    assert receipt.body_decision == "promote"
    assert receipt.final_topology_valid
    assert receipt.selected_pre_repair_hash == receipt.refined_pre_repair_hash
    assert (tmp_path / "person_smplx_body_raw_reprojection.npz").exists()
    assert (tmp_path / "person_smplx_body_refined_pre_repair.npz").exists()


def test_abstained_candidate_exports_raw_but_keeps_warning_evidence(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    _patch_mesh_generation(monkeypatch)
    result = _result(tmp_path, abstain=True)
    output = tmp_path / "person_smplx_body.npz"

    authorized_mesh_export.save_regression_mesh(result, output)

    with np.load(output) as payload:
        np.testing.assert_allclose(payload["vertices"], _tetra(0.0)[0])
    receipt = load_body_carrier_receipt(tmp_path / "body_carrier_receipt.json")
    assert isinstance(receipt, BodyCarrierReceiptV2)
    assert receipt.refinement_decision == "abstain"
    assert receipt.canonical_source == "raw_image_fit"
    assert receipt.selected_pre_repair_hash == receipt.raw_reprojection_hash
    assert receipt.body_decision == "promote"
    assert "WARN:low_view_diversity" in receipt.warnings


def test_invalid_unrepaired_final_mesh_is_rejected(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    result = _result(tmp_path)
    open_vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    open_faces = np.array([[0, 1, 2]], dtype=np.int32)
    monkeypatch.setattr(  # type: ignore[attr-defined]
        authorized_mesh_export,
        "_create_body_mesh_from_regression",
        lambda *args, **kwargs: (open_vertices, open_faces),
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        authorized_mesh_export,
        "repair_body_mesh_for_export",
        lambda vertices, faces: None,
    )

    authorized_mesh_export.save_regression_mesh(
        result,
        tmp_path / "person_smplx_body.npz",
    )

    receipt = load_body_carrier_receipt(tmp_path / "body_carrier_receipt.json")
    assert isinstance(receipt, BodyCarrierReceiptV2)
    assert receipt.body_decision == "reject"
    assert "final_export_topology_invalid" in receipt.blockers
    assert "final_export_repair_unavailable" in receipt.warnings


def test_non_governed_fit_uses_legacy_exporter(tmp_path: Path, monkeypatch: object) -> None:
    called: dict[str, object] = {}

    def legacy(result: object, path: Path, **kwargs: object) -> Path:
        called["result"] = result
        called["kwargs"] = kwargs
        return path

    monkeypatch.setattr(  # type: ignore[attr-defined]
        authorized_mesh_export,
        "_legacy_save_regression_mesh",
        legacy,
    )
    result = SimpleNamespace(measurement_fit=None)
    path = tmp_path / "legacy.npz"

    returned = authorized_mesh_export.save_regression_mesh(result, path)

    assert returned == path
    assert called["result"] is result
