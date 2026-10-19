from __future__ import annotations

import numpy as np
import pytest

from smii.meshing.body_carrier_receipt_v2 import (
    BodyCarrierReceiptV2,

)
from smii.meshing.body_carrier_v2_builder import build_body_carrier_receipt_v2
from smii.meshing.body_carrier_v2_io import load_body_carrier_receipt_v2
from smii.pipelines.refinement_authority import build_receipt, solve_bounded_refinement
from smii.pipelines.refinement_policy import RefinementPolicy


class Row:
    name = "height"
    mean = 0.0
    std = 1.0
    weights = (1.0,)


def refinement_receipt(*, warning: str | None = None):
    policy = RefinementPolicy.from_effective_config(
        backend="test",
        num_betas=1,
        scale_measurement="height",
        models=(Row(),),
        settings={
            "beta_lower": -2.0,
            "beta_upper": 2.0,
            "prior_weight": 0.1,
            "anchor_weight": 1.0,
            "max_beta_shift": 3.0,
            "max_measurement_residual": 10.0,
            "max_residual_degradation": 10.0,
        },
    )
    anchor = np.zeros(1)
    solution = solve_bounded_refinement(np.eye(1), anchor, anchor, policy)
    warnings = (warning,) if warning else ()
    return build_receipt(
        policy=policy,
        measurements={"height": 0.0},
        names=("height",),
        anchor_betas=anchor,
        solution=solution,
        warnings=warnings,
        severity="warn" if warnings else "pass",
    )


def build(receipt, **overrides):
    values = {
        "source_hash": "a" * 64,
        "raw_reprojection_hash": "b" * 64,
        "refined_pre_repair_hash": "c" * 64,
        "repaired_export_hash": "d" * 64,
        "refinement_receipt": receipt,
        "raw_topology": {"vertex_count": 10, "face_count": 16},
        "refined_pre_repair_topology": {"vertex_count": 10, "face_count": 16},
        "final_export_topology": {"vertex_count": 9, "face_count": 14},
        "topology_label": "A_v9",
        "final_geometry_finite": True,
        "final_topology_valid": True,
        "final_landmark_residuals": {"head": 0.1},
        "final_skull_rigidity_residual": 0.2,
        "body_fit_confidence": 0.9,
        "trust_level": "high",
        "severity": "pass",
        "warnings": (),
    }
    values.update(overrides)
    return build_body_carrier_receipt_v2(**values)


def test_promoted_refinement_selects_refined_checkpoint() -> None:
    receipt = build(refinement_receipt())

    assert receipt.canonical_source == "refined_candidate"
    assert receipt.selected_pre_repair_hash == receipt.refined_pre_repair_hash
    assert receipt.body_decision == "promote"
    assert receipt.promotion == 1


def test_reference_warning_abstains_refinement_but_raw_body_can_promote() -> None:
    refinement = refinement_receipt(warning="WARN:low_view_diversity")
    receipt = build(
        refinement,
        severity="warn",
        warnings=("WARN:low_view_diversity",),
    )

    assert refinement.decision == "abstain"
    assert receipt.canonical_source == "raw_image_fit"
    assert receipt.selected_pre_repair_hash == receipt.raw_reprojection_hash
    assert receipt.body_decision == "promote"
    assert receipt.warnings == ("WARN:low_view_diversity",)


def test_final_export_invalidity_rejects_body() -> None:
    receipt = build(refinement_receipt(), final_geometry_finite=False)

    assert receipt.body_decision == "reject"
    assert "final_export_non_finite" in receipt.blockers
    assert receipt.blocked_consumers


def test_refined_source_requires_refinement_promotion() -> None:
    refinement = refinement_receipt(warning="WARN:low_view_diversity")

    with pytest.raises(ValueError, match="Canonical source"):
        BodyCarrierReceiptV2(
            source_hash="a" * 64,
            raw_reprojection_hash="b" * 64,
            refined_pre_repair_hash="c" * 64,
            repaired_export_hash="d" * 64,
            refinement_receipt_hash=refinement.receipt_hash,
            refinement_decision="abstain",
            canonical_source="refined_candidate",
            selected_pre_repair_hash="c" * 64,
            raw_topology={"vertex_count": 10, "face_count": 16},
            refined_pre_repair_topology={"vertex_count": 10, "face_count": 16},
            final_export_topology={"vertex_count": 9, "face_count": 14},
            final_geometry_finite=True,
            final_topology_valid=True,
            final_landmark_residuals={},
            final_skull_rigidity_residual=0.2,
            body_fit_confidence=0.9,
            trust_level="high",
            severity="warn",
            body_decision="promote",
            blockers=(),
            warnings=(),
            blocked_consumers=(),
            topology_label="A_v9",
        )


def test_receipt_round_trip(tmp_path) -> None:
    receipt = build(refinement_receipt())
    path = receipt.to_json(tmp_path / "body_receipt.json")

    loaded = load_body_carrier_receipt_v2(path)

    assert loaded.to_dict() == receipt.to_dict()
    assert loaded.receipt_hash == receipt.receipt_hash
