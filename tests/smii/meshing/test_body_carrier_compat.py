from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from smii.meshing import (
    BodyCarrierReceipt,
    BodyCarrierReceiptV2,
    can_consume_receipt,
    load_body_carrier_receipt,
)
from smii.pipelines.refinement_policy import canonical_hash


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_mesh(path: Path, *, non_finite: bool = False) -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    if non_finite:
        vertices[0, 0] = np.nan
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    np.savez(path, vertices=vertices, faces=faces)


def _write_governed_fit(
    directory: Path,
    *,
    decision: str,
    status: str,
    trust_level: str,
    flags: tuple[str, ...] = (),
    corrupt_hash: bool = False,
) -> None:
    refinement = {
        "schema_version": "smii.body_refinement_receipt.v1",
        "decision": decision,
        "blockers": [] if decision == "promote" else ["candidate_not_authorized"],
        "warnings": list(flags),
    }
    receipt_hash = canonical_hash(refinement)
    if corrupt_hash:
        receipt_hash = "0" * 64
    payload = {
        "refinement_receipt": refinement,
        "refinement_receipt_hash": receipt_hash,
        "consistency_status": status,
        "consistency_flags": list(flags),
        "trust_level": trust_level,
    }
    (directory / "afflec_measurement_fit.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _legacy_receipt(directory: Path, *, promotion: int) -> BodyCarrierReceipt:
    raw = directory / "afflec_body_raw_reprojection.npz"
    refined = directory / "afflec_body_refined_pre_repair.npz"
    final = directory / "afflec_body.npz"
    return BodyCarrierReceipt(
        source_hash="a" * 64,
        raw_reprojection_hash=_sha256(raw),
        refined_pre_repair_hash=_sha256(refined),
        repaired_export_hash=_sha256(final),
        vertex_count=3,
        face_count=1,
        topology_label="A_v3",
        landmark_residuals={"measurement_fit_residual": 0.1},
        skull_rigidity_residual=0.2,
        body_fit_confidence=0.9,
        promotion=promotion,
        blocked_consumers=[],
    )


def _write_standard_meshes(directory: Path, *, final_non_finite: bool = False) -> None:
    _write_mesh(directory / "afflec_body_raw_reprojection.npz")
    _write_mesh(directory / "afflec_body_refined_pre_repair.npz")
    _write_mesh(directory / "afflec_body.npz", non_finite=final_non_finite)


def test_governed_warning_selects_raw_and_emits_promoted_v2(tmp_path: Path) -> None:
    _write_standard_meshes(tmp_path)
    _write_governed_fit(
        tmp_path,
        decision="abstain",
        status="WARN",
        trust_level="high",
        flags=("WARN:low_view_diversity",),
    )
    target = tmp_path / "body_carrier_receipt.json"

    _legacy_receipt(tmp_path, promotion=0).to_json(target)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "smii.body_carrier_receipt.v2"
    assert payload["refinement_decision"] == "abstain"
    assert payload["canonical_source"] == "raw_image_fit"
    assert payload["selected_pre_repair_hash"] == payload["raw_reprojection_hash"]
    assert payload["body_decision"] == "promote"
    assert payload["promotion"] == 1
    assert payload["warnings"] == ["WARN:low_view_diversity"]
    assert payload["vertex_count"] == 3
    assert payload["face_count"] == 1

    legacy_payload = json.loads(
        (tmp_path / "body_carrier_receipt_v1.json").read_text(encoding="utf-8")
    )
    assert "schema_version" not in legacy_payload
    assert legacy_payload["promotion"] == 0

    loaded = load_body_carrier_receipt(target)
    assert isinstance(loaded, BodyCarrierReceiptV2)
    assert loaded.vertex_count == 3
    assert can_consume_receipt(loaded, "seam_cost_field")

    from smii.meshing.body_carrier_receipt import (
        can_consume_receipt as direct_can_consume,
        load_body_carrier_receipt as direct_load,
    )

    directly_loaded = direct_load(target)
    assert isinstance(directly_loaded, BodyCarrierReceiptV2)
    assert direct_can_consume(directly_loaded, "seam_cost_field")


def test_final_non_finite_mesh_rejects_body_even_when_legacy_promoted(
    tmp_path: Path,
) -> None:
    _write_standard_meshes(tmp_path, final_non_finite=True)
    _write_governed_fit(
        tmp_path,
        decision="promote",
        status="PASS",
        trust_level="high",
    )
    target = tmp_path / "body_carrier_receipt.json"

    _legacy_receipt(tmp_path, promotion=1).to_json(target)

    loaded = load_body_carrier_receipt(target)
    assert isinstance(loaded, BodyCarrierReceiptV2)
    assert loaded.canonical_source == "refined_candidate"
    assert loaded.body_decision == "reject"
    assert loaded.promotion == 0
    assert "final_export_non_finite" in loaded.blockers
    assert not can_consume_receipt(loaded, "seam_cost_field")


def test_governed_writer_rejects_refinement_hash_mismatch(tmp_path: Path) -> None:
    _write_standard_meshes(tmp_path)
    _write_governed_fit(
        tmp_path,
        decision="abstain",
        status="WARN",
        trust_level="high",
        corrupt_hash=True,
    )

    with pytest.raises(ValueError, match="Refinement receipt hash"):
        _legacy_receipt(tmp_path, promotion=0).to_json(
            tmp_path / "body_carrier_receipt.json"
        )


def test_legacy_writer_remains_available_without_governed_fit(tmp_path: Path) -> None:
    _write_standard_meshes(tmp_path)
    target = tmp_path / "body_carrier_receipt.json"

    _legacy_receipt(tmp_path, promotion=1).to_json(target)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert "schema_version" not in payload
    assert payload["promotion"] == 1
    assert not (tmp_path / "body_carrier_receipt_v1.json").exists()
