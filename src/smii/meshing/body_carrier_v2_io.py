"""Serialization helpers for body carrier receipt v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence, cast

from .body_carrier_receipt_v2 import BodyCarrierReceiptV2
from .body_carrier_v2_policy import BodyDecision, CanonicalSource, Severity


def load_body_carrier_receipt_v2(path: str | Path) -> BodyCarrierReceiptV2:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Body carrier v2 receipt must decode to an object")
    final = cast(Mapping[str, object], payload["final_export_topology"])
    policy = cast(Mapping[str, object], payload["body_policy"])
    return BodyCarrierReceiptV2(
        source_hash=str(payload["source_hash"]),
        raw_reprojection_hash=str(payload["raw_reprojection_hash"]),
        refined_pre_repair_hash=str(payload["refined_pre_repair_hash"]),
        repaired_export_hash=str(payload["repaired_export_hash"]),
        refinement_receipt_hash=str(payload["refinement_receipt_hash"]),
        refinement_decision=cast(BodyDecision, payload["refinement_decision"]),
        canonical_source=cast(CanonicalSource, payload["canonical_source"]),
        selected_pre_repair_hash=str(payload["selected_pre_repair_hash"]),
        raw_topology=cast(Mapping[str, object], payload["raw_topology"]),
        refined_pre_repair_topology=cast(
            Mapping[str, object],
            payload["refined_pre_repair_topology"],
        ),
        final_export_topology=final,
        final_geometry_finite=cast(bool, final["geometry_finite"]),
        final_topology_valid=cast(bool, final["topology_valid"]),
        final_landmark_residuals=cast(
            Mapping[str, float],
            payload.get("final_landmark_residuals", {}),
        ),
        final_skull_rigidity_residual=float(payload["final_skull_rigidity_residual"]),
        body_fit_confidence=float(payload["body_fit_confidence"]),
        trust_level=str(policy["trust_level"]),
        confidence_threshold=float(policy["confidence_threshold"]),
        skull_residual_threshold=float(policy["skull_residual_threshold"]),
        severity=cast(Severity, payload["diagnostic_severity"]),
        body_decision=cast(BodyDecision, payload["body_decision"]),
        blockers=tuple(cast(Sequence[str], payload.get("blockers", ()))),
        warnings=tuple(cast(Sequence[str], payload.get("warnings", ()))),
        blocked_consumers=tuple(
            cast(Sequence[str], payload.get("blocked_consumers", ()))
        ),
        topology_label=str(final["topology_label"]),
        schema_version=str(payload.get("schema_version", "smii.body_carrier_receipt.v2")),
    )
