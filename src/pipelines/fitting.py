"""Fitting pipeline helpers that emit schema-validated body records."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Mapping, Sequence

from ..schemas.validators import DEFAULT_SCHEMA_NAME, validate_body_payload

__all__ = ["FittingPipeline", "build_body_record"]


@dataclass(slots=True)
class FittingPipeline:
    """Coordinate fitting output validation and emission."""

    solver: str
    schema_name: str = DEFAULT_SCHEMA_NAME
    metadata_overrides: Mapping[str, Any] = field(default_factory=dict)

    def emit_record(
        self,
        *,
        subject_id: str,
        metadata: Mapping[str, Any],
        anatomical_landmarks: Sequence[Mapping[str, Any]],
        measurements: Sequence[Mapping[str, Any]],
        joints: Sequence[Mapping[str, Any]],
        motion_ranges: Sequence[Mapping[str, Any]],
        mobility_constraints: Sequence[Mapping[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Return a schema-compliant record for the fitted subject."""

        record = build_body_record(
            subject_id=subject_id,
            metadata=metadata,
            anatomical_landmarks=anatomical_landmarks,
            measurements=measurements,
            joints=joints,
            motion_ranges=motion_ranges,
            mobility_constraints=mobility_constraints,
            solver=self.solver,
            schema_name=self.schema_name,
            overrides=self.metadata_overrides,
        )
        return record


def build_body_record(
    *,
    subject_id: str,
    metadata: Mapping[str, Any],
    anatomical_landmarks: Sequence[Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
    joints: Sequence[Mapping[str, Any]],
    motion_ranges: Sequence[Mapping[str, Any]],
    mobility_constraints: Sequence[Mapping[str, Any]] | None = None,
    solver: str,
    schema_name: str = DEFAULT_SCHEMA_NAME,
    overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble and validate a body record suitable for downstream tools."""

    overrides = overrides or {}
    mobility_constraints = mobility_constraints or ()
    enriched_metadata: Dict[str, Any] = {
        "schema_version": overrides.get("schema_version", metadata.get("schema_version", "v1.0.0")),
        "coordinate_frame": metadata.get("coordinate_frame", overrides.get("coordinate_frame", "smii_body_local")),
        "unit_system": metadata.get("unit_system", overrides.get("unit_system", "metric")),
        "source": metadata.get("source", solver),
        "subject_id": subject_id,
        "generated_at": metadata.get("generated_at", datetime.utcnow().isoformat()),
    }

    for key, value in metadata.items():
        if key not in enriched_metadata:
            enriched_metadata[key] = value

    record: Dict[str, Any] = {
        "metadata": enriched_metadata,
        "anatomical_landmarks": list(anatomical_landmarks),
        "measurements": list(measurements),
        "joints": list(joints),
        "motion_ranges": list(motion_ranges),
        "mobility_constraints": list(mobility_constraints),
    }

    validate_body_payload(record, schema_name=schema_name)

    return record
