"""Measure C3/BT369 transport consistency from paired observations.

A C3 phase rotation acts as ``phase + 1 mod 3``.  The six-sector tangent
orientation used by the existing BT369 serializer advances by two sectors for
the same 120-degree rotation.  A transport that commutes with this action has a
constant phase offset modulo 3 and constant orientation offset modulo 6 across
its declared correspondence slice.

The estimator below fits those two offsets by modal agreement and reports the
fraction of observations that violate the fitted equivariant transport.  It is
an empirical consistency receipt over supplied correspondences, not evidence
that garment mechanics possess exact C3 symmetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .suit_closure_receipt import TriadicTransportWitness

C3_PHASE_CARDINALITY = 3
BT369_ORIENTATION_CARDINALITY = 6
C3_ORIENTATION_STEP = 2
C3_TRANSPORT_SCHEMA_VERSION = "smii.c3_bt369_transport_measurement.v1"

__all__ = [
    "BT369_ORIENTATION_CARDINALITY",
    "C3_ORIENTATION_STEP",
    "C3_PHASE_CARDINALITY",
    "C3_TRANSPORT_SCHEMA_VERSION",
    "TriadicTransportMeasurement",
    "TriadicTransportSample",
    "measure_triadic_transport",
]


def _bounded_int(value: object, *, name: str, modulus: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if number < 0 or number >= modulus:
        raise ValueError(f"{name} must lie in [0, {modulus - 1}]")
    return number


@dataclass(frozen=True, slots=True)
class TriadicTransportSample:
    """One observed source/target phase and tangent-orientation correspondence."""

    source_phase: int
    target_phase: int
    source_orientation_bin: int
    target_orientation_bin: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_phase",
            _bounded_int(self.source_phase, name="source_phase", modulus=C3_PHASE_CARDINALITY),
        )
        object.__setattr__(
            self,
            "target_phase",
            _bounded_int(self.target_phase, name="target_phase", modulus=C3_PHASE_CARDINALITY),
        )
        object.__setattr__(
            self,
            "source_orientation_bin",
            _bounded_int(
                self.source_orientation_bin,
                name="source_orientation_bin",
                modulus=BT369_ORIENTATION_CARDINALITY,
            ),
        )
        object.__setattr__(
            self,
            "target_orientation_bin",
            _bounded_int(
                self.target_orientation_bin,
                name="target_orientation_bin",
                modulus=BT369_ORIENTATION_CARDINALITY,
            ),
        )
        if isinstance(self.weight, bool):
            raise TypeError("weight must be numeric")
        weight = float(self.weight)
        if weight <= 0.0 or not (weight < float("inf")):
            raise ValueError("weight must be finite and positive")
        object.__setattr__(self, "weight", weight)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TriadicTransportSample":
        return cls(
            source_phase=payload.get("source_phase"),  # type: ignore[arg-type]
            target_phase=payload.get("target_phase"),  # type: ignore[arg-type]
            source_orientation_bin=payload.get("source_orientation_bin"),  # type: ignore[arg-type]
            target_orientation_bin=payload.get("target_orientation_bin"),  # type: ignore[arg-type]
            weight=payload.get("weight", 1.0),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_phase": self.source_phase,
            "target_phase": self.target_phase,
            "source_orientation_bin": self.source_orientation_bin,
            "target_orientation_bin": self.target_orientation_bin,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class TriadicTransportMeasurement:
    """Fitted C3-equivariant offset model and its observed residual."""

    sample_count: int
    total_weight: float
    fitted_phase_offset: int
    fitted_orientation_offset: int
    phase_mismatch_weight: float
    orientation_mismatch_weight: float
    joint_mismatch_weight: float
    phase_error: float
    orientation_error: float
    c3_equivariance_error: float
    threshold: float

    @property
    def passes(self) -> bool:
        return self.c3_equivariance_error <= self.threshold

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": C3_TRANSPORT_SCHEMA_VERSION,
            "sample_count": self.sample_count,
            "total_weight": self.total_weight,
            "fitted_phase_offset": self.fitted_phase_offset,
            "fitted_orientation_offset": self.fitted_orientation_offset,
            "phase_mismatch_weight": self.phase_mismatch_weight,
            "orientation_mismatch_weight": self.orientation_mismatch_weight,
            "joint_mismatch_weight": self.joint_mismatch_weight,
            "phase_error": self.phase_error,
            "orientation_error": self.orientation_error,
            "c3_equivariance_error": self.c3_equivariance_error,
            "threshold": self.threshold,
            "passes": self.passes,
            "c3_action": {
                "phase_step_mod3": 1,
                "orientation_step_mod6": C3_ORIENTATION_STEP,
            },
            "claim_boundary": "measured_correspondence_consistency_not_physical_c3_symmetry_law",
        }


def _weighted_mode(values: Sequence[tuple[int, float]]) -> int:
    if not values:
        raise ValueError("at least one observation is required")
    totals: dict[int, float] = {}
    for value, weight in values:
        totals[value] = totals.get(value, 0.0) + weight
    return min(totals, key=lambda value: (-totals[value], value))


def measure_triadic_transport(
    samples: Sequence[TriadicTransportSample],
    *,
    threshold: float,
    transport_model: str,
    address_schema: str,
    phase_provenance_hash: str,
    orientation_receipt_hash: str,
) -> tuple[TriadicTransportMeasurement, TriadicTransportWitness]:
    """Fit a C3-equivariant offset transport and emit its runtime witness.

    Constant offsets commute with the regular C3 phase action and with the
    corresponding +2 action on six-sector orientation.  The residual is the
    weighted fraction of correspondences that do not agree with both fitted
    offsets.
    """

    if not samples:
        raise ValueError("at least one triadic transport sample is required")
    if isinstance(threshold, bool):
        raise TypeError("threshold must be numeric")
    threshold_value = float(threshold)
    if threshold_value < 0.0 or threshold_value > 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    for name, value in (
        ("transport_model", transport_model),
        ("address_schema", address_schema),
        ("phase_provenance_hash", phase_provenance_hash),
        ("orientation_receipt_hash", orientation_receipt_hash),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")

    phase_offsets = [
        ((sample.target_phase - sample.source_phase) % C3_PHASE_CARDINALITY, sample.weight)
        for sample in samples
    ]
    orientation_offsets = [
        (
            (sample.target_orientation_bin - sample.source_orientation_bin)
            % BT369_ORIENTATION_CARDINALITY,
            sample.weight,
        )
        for sample in samples
    ]
    fitted_phase = _weighted_mode(phase_offsets)
    fitted_orientation = _weighted_mode(orientation_offsets)
    total_weight = sum(sample.weight for sample in samples)

    phase_mismatch = 0.0
    orientation_mismatch = 0.0
    joint_mismatch = 0.0
    for sample in samples:
        phase_ok = (
            (sample.target_phase - sample.source_phase) % C3_PHASE_CARDINALITY
            == fitted_phase
        )
        orientation_ok = (
            (sample.target_orientation_bin - sample.source_orientation_bin)
            % BT369_ORIENTATION_CARDINALITY
            == fitted_orientation
        )
        if not phase_ok:
            phase_mismatch += sample.weight
        if not orientation_ok:
            orientation_mismatch += sample.weight
        if not (phase_ok and orientation_ok):
            joint_mismatch += sample.weight

    phase_error = phase_mismatch / total_weight
    orientation_error = orientation_mismatch / total_weight
    equivariance_error = joint_mismatch / total_weight
    measurement = TriadicTransportMeasurement(
        sample_count=len(samples),
        total_weight=total_weight,
        fitted_phase_offset=fitted_phase,
        fitted_orientation_offset=fitted_orientation,
        phase_mismatch_weight=phase_mismatch,
        orientation_mismatch_weight=orientation_mismatch,
        joint_mismatch_weight=joint_mismatch,
        phase_error=phase_error,
        orientation_error=orientation_error,
        c3_equivariance_error=equivariance_error,
        threshold=threshold_value,
    )
    witness = TriadicTransportWitness(
        transport_model=transport_model,
        address_schema=address_schema,
        phase_provenance_hash=phase_provenance_hash,
        orientation_receipt_hash=orientation_receipt_hash,
        c3_equivariance_error=equivariance_error,
        c3_equivariance_threshold=threshold_value,
    )
    return measurement, witness
