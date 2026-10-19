"""Builders and consumer gates for body carrier receipt v2."""

from __future__ import annotations

from typing import Mapping, Sequence, cast

from smii.pipelines.refinement_authority import RefinementReceipt

from .body_carrier_receipt_v2 import BodyCarrierReceiptV2
from .body_carrier_v2_policy import (
    DEFAULT_BLOCKED_CONSUMERS,
    BodyDecision,
    CanonicalSource,
    Severity,
    decide_body_authorization,
    validate_hash,
)


def _refinement_reference(
    *,
    refinement_receipt: RefinementReceipt | None,
    refinement_receipt_hash: str | None,
    refinement_decision: BodyDecision | None,
) -> tuple[str, BodyDecision]:
    if refinement_receipt is not None:
        receipt_hash = refinement_receipt.receipt_hash
        decision = refinement_receipt.decision
        if refinement_receipt_hash is not None and refinement_receipt_hash != receipt_hash:
            raise ValueError("Refinement receipt hash does not match the supplied receipt")
        if refinement_decision is not None and refinement_decision != decision:
            raise ValueError("Refinement decision does not match the supplied receipt")
    else:
        if refinement_receipt_hash is None or refinement_decision is None:
            raise ValueError("A refinement receipt or its hash and decision are required")
        receipt_hash = refinement_receipt_hash
        decision = refinement_decision

    if decision not in {"promote", "abstain", "reject"}:
        raise ValueError("Invalid refinement decision")
    return validate_hash(receipt_hash, "refinement_receipt_hash"), cast(BodyDecision, decision)


def build_body_carrier_receipt_v2(
    *,
    source_hash: str,
    raw_reprojection_hash: str,
    refined_pre_repair_hash: str,
    repaired_export_hash: str,
    raw_topology: Mapping[str, object],
    refined_pre_repair_topology: Mapping[str, object],
    final_export_topology: Mapping[str, object],
    topology_label: str,
    final_geometry_finite: bool,
    final_topology_valid: bool,
    final_landmark_residuals: Mapping[str, float],
    final_skull_rigidity_residual: float,
    body_fit_confidence: float,
    trust_level: str,
    severity: Severity,
    warnings: Sequence[str] = (),
    confidence_threshold: float = 0.75,
    skull_residual_threshold: float = 0.35,
    refinement_receipt: RefinementReceipt | None = None,
    refinement_receipt_hash: str | None = None,
    refinement_decision: BodyDecision | None = None,
) -> BodyCarrierReceiptV2:
    receipt_hash, decision = _refinement_reference(
        refinement_receipt=refinement_receipt,
        refinement_receipt_hash=refinement_receipt_hash,
        refinement_decision=refinement_decision,
    )
    canonical_source: CanonicalSource = (
        "refined_candidate" if decision == "promote" else "raw_image_fit"
    )
    selected_hash = (
        refined_pre_repair_hash
        if canonical_source == "refined_candidate"
        else raw_reprojection_hash
    )
    body_decision, blockers = decide_body_authorization(
        trust_level=trust_level,
        severity=severity,
        confidence=body_fit_confidence,
        skull_residual=final_skull_rigidity_residual,
        geometry_finite=final_geometry_finite,
        topology_valid=final_topology_valid,
        confidence_threshold=confidence_threshold,
        skull_threshold=skull_residual_threshold,
    )
    return BodyCarrierReceiptV2(
        source_hash=source_hash,
        raw_reprojection_hash=raw_reprojection_hash,
        refined_pre_repair_hash=refined_pre_repair_hash,
        repaired_export_hash=repaired_export_hash,
        refinement_receipt_hash=receipt_hash,
        refinement_decision=decision,
        canonical_source=canonical_source,
        selected_pre_repair_hash=selected_hash,
        raw_topology=raw_topology,
        refined_pre_repair_topology=refined_pre_repair_topology,
        final_export_topology=final_export_topology,
        final_geometry_finite=final_geometry_finite,
        final_topology_valid=final_topology_valid,
        final_landmark_residuals=final_landmark_residuals,
        final_skull_rigidity_residual=final_skull_rigidity_residual,
        body_fit_confidence=body_fit_confidence,
        trust_level=trust_level,
        confidence_threshold=confidence_threshold,
        skull_residual_threshold=skull_residual_threshold,
        severity=severity,
        body_decision=body_decision,
        blockers=blockers,
        warnings=tuple(dict.fromkeys(str(item) for item in warnings)),
        blocked_consumers=() if body_decision == "promote" else DEFAULT_BLOCKED_CONSUMERS,
        topology_label=topology_label,
    )


def can_consume_body_receipt_v2(
    receipt: BodyCarrierReceiptV2,
    consumer: str | None = None,
) -> bool:
    return receipt.body_decision == "promote" and (
        consumer is None or consumer not in receipt.blocked_consumers
    )
