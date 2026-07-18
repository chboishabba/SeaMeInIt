"""Final-export body authorization and lineage receipt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from smii.pipelines.refinement_policy import canonical_hash

from .body_carrier_v2_policy import BodyDecision, CanonicalSource, Severity


@dataclass(frozen=True, slots=True)
class BodyCarrierReceiptV2:
    source_hash: str
    raw_reprojection_hash: str
    refined_pre_repair_hash: str
    repaired_export_hash: str
    refinement_receipt_hash: str
    refinement_decision: BodyDecision
    canonical_source: CanonicalSource
    selected_pre_repair_hash: str
    raw_topology: Mapping[str, object]
    refined_pre_repair_topology: Mapping[str, object]
    final_export_topology: Mapping[str, object]
    final_geometry_finite: bool
    final_topology_valid: bool
    final_landmark_residuals: Mapping[str, float]
    final_skull_rigidity_residual: float
    body_fit_confidence: float
    trust_level: str
    severity: Severity
    body_decision: BodyDecision
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    blocked_consumers: tuple[str, ...]
    confidence_threshold: float = 0.75
    skull_residual_threshold: float = 0.35
    topology_label: str = "unknown"
    schema_version: str = "smii.body_carrier_receipt.v2"

    def __post_init__(self) -> None:
        from .body_carrier_v2_validation import validate_body_carrier_receipt_v2

        validate_body_carrier_receipt_v2(self)

    @property
    def promotion(self) -> int:
        return int(self.body_decision == "promote")

    @property
    def receipt_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        from .body_carrier_v2_serialization import body_carrier_receipt_v2_to_dict

        return body_carrier_receipt_v2_to_dict(self)

    def to_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target
