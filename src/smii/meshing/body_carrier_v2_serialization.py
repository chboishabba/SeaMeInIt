"""Canonical mapping for body carrier receipt v2."""

from __future__ import annotations

from typing import Any

_HASHES = (
    "source_hash",
    "raw_reprojection_hash",
    "refined_pre_repair_hash",
    "repaired_export_hash",
    "refinement_receipt_hash",
    "selected_pre_repair_hash",
)


def body_carrier_receipt_v2_to_dict(receipt: Any) -> dict[str, object]:
    final = dict(receipt.final_export_topology)
    final.update(
        topology_label=receipt.topology_label,
        geometry_finite=receipt.final_geometry_finite,
        topology_valid=receipt.final_topology_valid,
    )
    landmark_residuals = dict(receipt.final_landmark_residuals)
    return {
        "schema_version": receipt.schema_version,
        **{name: getattr(receipt, name) for name in _HASHES},
        "refinement_decision": receipt.refinement_decision,
        "canonical_source": receipt.canonical_source,
        "raw_topology": dict(receipt.raw_topology),
        "refined_pre_repair_topology": dict(receipt.refined_pre_repair_topology),
        "final_export_topology": final,
        "vertex_count": receipt.vertex_count,
        "face_count": receipt.face_count,
        "topology_label": receipt.topology_label,
        "landmark_residuals": landmark_residuals,
        "skull_rigidity_residual": receipt.final_skull_rigidity_residual,
        "final_landmark_residuals": landmark_residuals,
        "final_skull_rigidity_residual": receipt.final_skull_rigidity_residual,
        "body_fit_confidence": receipt.body_fit_confidence,
        "body_policy": {
            "trust_level": receipt.trust_level,
            "confidence_threshold": receipt.confidence_threshold,
            "skull_residual_threshold": receipt.skull_residual_threshold,
        },
        "diagnostic_severity": receipt.severity,
        "body_decision": receipt.body_decision,
        "promotion": receipt.promotion,
        "blockers": list(receipt.blockers),
        "warnings": list(receipt.warnings),
        "blocked_consumers": list(receipt.blocked_consumers),
    }
