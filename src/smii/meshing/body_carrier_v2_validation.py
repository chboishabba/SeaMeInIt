"""Invariant validation for body carrier receipt v2."""

from __future__ import annotations

from typing import Any

from .body_carrier_v2_policy import finite_nonnegative, validate_hash, validate_topology

_HASHES = (
    "source_hash", "raw_reprojection_hash", "refined_pre_repair_hash",
    "repaired_export_hash", "refinement_receipt_hash", "selected_pre_repair_hash",
)
_TOPOLOGIES = ("raw_topology", "refined_pre_repair_topology", "final_export_topology")


def validate_body_carrier_receipt_v2(receipt: Any) -> None:
    if receipt.schema_version != "smii.body_carrier_receipt.v2":
        raise ValueError("Unsupported body carrier receipt schema")
    for name in _HASHES:
        object.__setattr__(receipt, name, validate_hash(getattr(receipt, name), name))
    if receipt.refinement_decision not in {"promote", "abstain", "reject"}:
        raise ValueError("Invalid refinement decision")
    if receipt.body_decision not in {"promote", "abstain", "reject"}:
        raise ValueError("Invalid body decision")
    if receipt.severity not in {"pass", "warn", "fail"}:
        raise ValueError("Invalid diagnostic severity")
    if receipt.canonical_source not in {"raw_image_fit", "refined_candidate"}:
        raise ValueError("Invalid canonical source")
    if not all(isinstance(value, bool) for value in (
        receipt.final_geometry_finite, receipt.final_topology_valid,
    )):
        raise TypeError("Final geometry and topology validity must be bools")
    if not receipt.trust_level or not receipt.topology_label:
        raise ValueError("trust_level and topology_label are required")
    for name in _TOPOLOGIES:
        object.__setattr__(receipt, name, validate_topology(getattr(receipt, name), name))

    confidence = finite_nonnegative(receipt.body_fit_confidence, "body_fit_confidence", maximum=1.0)
    skull = finite_nonnegative(receipt.final_skull_rigidity_residual, "final_skull_rigidity_residual")
    confidence_gate = finite_nonnegative(receipt.confidence_threshold, "confidence_threshold", maximum=1.0)
    skull_gate = finite_nonnegative(receipt.skull_residual_threshold, "skull_residual_threshold")
    for name, value in (
        ("body_fit_confidence", confidence),
        ("final_skull_rigidity_residual", skull),
        ("confidence_threshold", confidence_gate),
        ("skull_residual_threshold", skull_gate),
    ):
        object.__setattr__(receipt, name, value)
    object.__setattr__(receipt, "final_landmark_residuals", {
        str(key): finite_nonnegative(value, f"final_landmark_residuals[{key!r}]")
        for key, value in receipt.final_landmark_residuals.items()
    })
    for name in ("blockers", "warnings", "blocked_consumers"):
        object.__setattr__(receipt, name, tuple(dict.fromkeys(map(str, getattr(receipt, name)))))

    refined = receipt.refinement_decision == "promote"
    expected_source = "refined_candidate" if refined else "raw_image_fit"
    expected_hash = receipt.refined_pre_repair_hash if refined else receipt.raw_reprojection_hash
    if receipt.canonical_source != expected_source:
        raise ValueError("Canonical source does not match refinement decision")
    if receipt.selected_pre_repair_hash != expected_hash:
        raise ValueError("Selected pre-repair hash does not match canonical source")
    if receipt.body_decision == "promote":
        invalid = (
            receipt.blockers or receipt.blocked_consumers
            or not receipt.final_geometry_finite or not receipt.final_topology_valid
            or confidence < confidence_gate or skull > skull_gate
        )
        if invalid:
            raise ValueError("Promoted body receipt violates final-export authorization")
    elif not receipt.blockers or not receipt.blocked_consumers:
        raise ValueError("Non-promoted body receipt requires blockers and consumers")
