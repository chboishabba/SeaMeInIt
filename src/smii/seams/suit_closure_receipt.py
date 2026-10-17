"""Receipts that close the digital seam atlas back onto a physical suit.

The existing SeaMeInIt pipeline promotes a receipted digital garment through
``FinishedSeamReceipt``.  This module deliberately does *not* treat that digital
artifact as evidence that a fabricated suit behaves as predicted.  Instead it
adds three explicit physical-evidence boundaries:

``FinishedSeamReceipt -> AssemblyReceipt -> WearFitReceipt -> PhysicalROMValidationReceipt``

The final receipt can additionally carry a C3/BT369 transport witness.  This is
a provenance/equivariance check over the chosen triadic address/orientation
transport, not a claim that C3 symmetry is a law of garment physics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from .seam_derivation import FinishedSeamReceipt, can_consume_finished_seam_receipt

Promotion = Literal[-1, 0, 1]

ASSEMBLY_SCHEMA_VERSION = "smii.assembly_receipt.v1"
WEAR_FIT_SCHEMA_VERSION = "smii.wear_fit_receipt.v1"
PHYSICAL_ROM_SCHEMA_VERSION = "smii.physical_rom_validation_receipt.v1"

ASSEMBLY_CLAIM_BOUNDARY = (
    "assembly_receipt_measures_realization_against_pattern_not_physical_performance_truth"
)
WEAR_FIT_CLAIM_BOUNDARY = (
    "wear_fit_receipt_measures_declared_tasks_and_fields_not_global_clinical_or_safety_truth"
)
PHYSICAL_ROM_CLAIM_BOUNDARY = (
    "physical_rom_validation_closes_declared_measurement_loop_not_universal_garment_physics"
)

DEFAULT_ASSEMBLY_BLOCKED_CONSUMERS = ("wear_fit", "physical_rom_validation", "deployment")
DEFAULT_WEAR_FIT_BLOCKED_CONSUMERS = ("physical_rom_validation", "deployment")
DEFAULT_PHYSICAL_ROM_BLOCKED_CONSUMERS = ("deployment",)


__all__ = [
    "ASSEMBLY_SCHEMA_VERSION",
    "WEAR_FIT_SCHEMA_VERSION",
    "PHYSICAL_ROM_SCHEMA_VERSION",
    "ASSEMBLY_CLAIM_BOUNDARY",
    "WEAR_FIT_CLAIM_BOUNDARY",
    "PHYSICAL_ROM_CLAIM_BOUNDARY",
    "AssemblyReceipt",
    "WearFitReceipt",
    "TriadicTransportWitness",
    "PhysicalROMValidationReceipt",
    "derive_assembly_receipt",
    "derive_wear_fit_receipt",
    "derive_physical_rom_validation_receipt",
    "can_consume_assembly_receipt",
    "can_consume_wear_fit_receipt",
    "can_consume_physical_rom_validation_receipt",
    "load_assembly_receipt",
    "load_wear_fit_receipt",
    "load_physical_rom_validation_receipt",
]


def _normalize_promotion(value: object) -> Promotion:
    if isinstance(value, bool):
        raise ValueError("promotion must be one of -1, 0, or 1")
    if isinstance(value, int):
        promotion = value
    elif isinstance(value, float) and value.is_integer():
        promotion = int(value)
    elif isinstance(value, str) and value in {"-1", "0", "1"}:
        promotion = int(value)
    else:
        raise ValueError("promotion must be one of -1, 0, or 1")
    if promotion not in {-1, 0, 1}:
        raise ValueError("promotion must be one of -1, 0, or 1")
    return cast(Promotion, promotion)


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"field '{key}' must be a non-empty string")
    return value


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"field '{key}' must be a non-empty string when present")
    return value


def _non_negative_float(value: object, key: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"field '{key}' must be numeric")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"field '{key}' must be numeric") from exc
    if number < 0.0:
        raise ValueError(f"field '{key}' must be non-negative")
    if number != number or number in {float("inf"), float("-inf")}:
        raise ValueError(f"field '{key}' must be finite")
    return number


def _fraction(value: object, key: str) -> float:
    number = _non_negative_float(value, key)
    if number > 1.0:
        raise ValueError(f"field '{key}' must lie in [0, 1]")
    return number


def _non_negative_int(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"field '{key}' must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"field '{key}' must be an integer")
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"field '{key}' must be an integer") from exc
    if number < 0:
        raise ValueError(f"field '{key}' must be non-negative")
    return number


def _string_list(value: object, key: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"field '{key}' must be a list")
    result = [str(item) for item in value]
    if any(not item for item in result):
        raise ValueError(f"field '{key}' may not contain empty strings")
    return result


def _float_mapping(value: object, key: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"field '{key}' must be an object")
    return {
        str(name): _non_negative_float(metric, f"{key}.{name}")
        for name, metric in value.items()
    }


def _blocked(promotion: Promotion, blocked: list[str], defaults: tuple[str, ...]) -> list[str]:
    if promotion != 1 and not blocked:
        return list(defaults)
    return blocked


def _promotion_name(promotion: Promotion) -> str:
    if promotion == 1:
        return "promoted"
    if promotion == 0:
        return "diagnostic_only"
    return "rejected"


@dataclass(frozen=True, slots=True)
class AssemblyReceipt:
    """Evidence that a physical assembly realizes a promoted digital pattern."""

    finished_seam_receipt_hash: str
    manufactured_artifacts_hash: str
    assembled_geometry_hash: str
    assembly_evidence_hash: str
    piece_count: int
    realized_seam_count: int
    max_geometry_deviation: float
    geometry_deviation_threshold: float
    material_lot_hashes: list[str]
    promotion: Promotion
    blocked_consumers: list[str]
    blocker_log: list[str]
    claim_boundary: str = ASSEMBLY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        for key in (
            "finished_seam_receipt_hash",
            "manufactured_artifacts_hash",
            "assembled_geometry_hash",
            "assembly_evidence_hash",
            "claim_boundary",
        ):
            value = getattr(self, key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"field '{key}' must be a non-empty string")
        object.__setattr__(self, "piece_count", _non_negative_int(self.piece_count, "piece_count"))
        object.__setattr__(
            self,
            "realized_seam_count",
            _non_negative_int(self.realized_seam_count, "realized_seam_count"),
        )
        object.__setattr__(
            self,
            "max_geometry_deviation",
            _non_negative_float(self.max_geometry_deviation, "max_geometry_deviation"),
        )
        object.__setattr__(
            self,
            "geometry_deviation_threshold",
            _non_negative_float(
                self.geometry_deviation_threshold,
                "geometry_deviation_threshold",
            ),
        )
        object.__setattr__(
            self,
            "material_lot_hashes",
            _string_list(self.material_lot_hashes, "material_lot_hashes"),
        )
        promotion = _normalize_promotion(self.promotion)
        object.__setattr__(self, "promotion", promotion)
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked(
                promotion,
                _string_list(self.blocked_consumers, "blocked_consumers"),
                DEFAULT_ASSEMBLY_BLOCKED_CONSUMERS,
            ),
        )
        object.__setattr__(self, "blocker_log", _string_list(self.blocker_log, "blocker_log"))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ASSEMBLY_SCHEMA_VERSION,
            "finished_seam_receipt_hash": self.finished_seam_receipt_hash,
            "manufactured_artifacts_hash": self.manufactured_artifacts_hash,
            "assembled_geometry_hash": self.assembled_geometry_hash,
            "assembly_evidence_hash": self.assembly_evidence_hash,
            "piece_count": self.piece_count,
            "realized_seam_count": self.realized_seam_count,
            "max_geometry_deviation": self.max_geometry_deviation,
            "geometry_deviation_threshold": self.geometry_deviation_threshold,
            "material_lot_hashes": list(self.material_lot_hashes),
            "promotion": int(self.promotion),
            "promotion_state": _promotion_name(self.promotion),
            "blocked_consumers": list(self.blocked_consumers),
            "blocker_log": list(self.blocker_log),
            "claim_boundary": self.claim_boundary,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AssemblyReceipt":
        return cls(
            finished_seam_receipt_hash=_required_str(payload, "finished_seam_receipt_hash"),
            manufactured_artifacts_hash=_required_str(payload, "manufactured_artifacts_hash"),
            assembled_geometry_hash=_required_str(payload, "assembled_geometry_hash"),
            assembly_evidence_hash=_required_str(payload, "assembly_evidence_hash"),
            piece_count=_non_negative_int(payload.get("piece_count"), "piece_count"),
            realized_seam_count=_non_negative_int(
                payload.get("realized_seam_count"),
                "realized_seam_count",
            ),
            max_geometry_deviation=_non_negative_float(
                payload.get("max_geometry_deviation"),
                "max_geometry_deviation",
            ),
            geometry_deviation_threshold=_non_negative_float(
                payload.get("geometry_deviation_threshold"),
                "geometry_deviation_threshold",
            ),
            material_lot_hashes=_string_list(payload.get("material_lot_hashes", []), "material_lot_hashes"),
            promotion=_normalize_promotion(payload.get("promotion")),
            blocked_consumers=_string_list(payload.get("blocked_consumers", []), "blocked_consumers"),
            blocker_log=_string_list(payload.get("blocker_log", []), "blocker_log"),
            claim_boundary=str(payload.get("claim_boundary", ASSEMBLY_CLAIM_BOUNDARY)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "AssemblyReceipt":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("AssemblyReceipt JSON must contain an object")
        return cls.from_mapping(payload)

    def to_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


@dataclass(frozen=True, slots=True)
class WearFitReceipt:
    """Measured-vs-predicted garment field residuals over a declared task slice."""

    assembly_receipt_hash: str
    rom_receipt_hash: str
    task_profile_hash: str
    predicted_field_hash: str
    measured_field_hash: str
    measurement_evidence_hash: str
    sample_count: int
    task_coverage: float
    field_residuals: dict[str, float]
    residual_thresholds: dict[str, float]
    promotion: Promotion
    blocked_consumers: list[str]
    blocker_log: list[str]
    claim_boundary: str = WEAR_FIT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        for key in (
            "assembly_receipt_hash",
            "rom_receipt_hash",
            "task_profile_hash",
            "predicted_field_hash",
            "measured_field_hash",
            "measurement_evidence_hash",
            "claim_boundary",
        ):
            value = getattr(self, key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"field '{key}' must be a non-empty string")
        object.__setattr__(self, "sample_count", _non_negative_int(self.sample_count, "sample_count"))
        object.__setattr__(self, "task_coverage", _fraction(self.task_coverage, "task_coverage"))
        object.__setattr__(
            self,
            "field_residuals",
            _float_mapping(self.field_residuals, "field_residuals"),
        )
        object.__setattr__(
            self,
            "residual_thresholds",
            _float_mapping(self.residual_thresholds, "residual_thresholds"),
        )
        promotion = _normalize_promotion(self.promotion)
        object.__setattr__(self, "promotion", promotion)
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked(
                promotion,
                _string_list(self.blocked_consumers, "blocked_consumers"),
                DEFAULT_WEAR_FIT_BLOCKED_CONSUMERS,
            ),
        )
        object.__setattr__(self, "blocker_log", _string_list(self.blocker_log, "blocker_log"))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": WEAR_FIT_SCHEMA_VERSION,
            "assembly_receipt_hash": self.assembly_receipt_hash,
            "rom_receipt_hash": self.rom_receipt_hash,
            "task_profile_hash": self.task_profile_hash,
            "predicted_field_hash": self.predicted_field_hash,
            "measured_field_hash": self.measured_field_hash,
            "measurement_evidence_hash": self.measurement_evidence_hash,
            "sample_count": self.sample_count,
            "task_coverage": self.task_coverage,
            "field_residuals": dict(self.field_residuals),
            "residual_thresholds": dict(self.residual_thresholds),
            "promotion": int(self.promotion),
            "promotion_state": _promotion_name(self.promotion),
            "blocked_consumers": list(self.blocked_consumers),
            "blocker_log": list(self.blocker_log),
            "claim_boundary": self.claim_boundary,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "WearFitReceipt":
        return cls(
            assembly_receipt_hash=_required_str(payload, "assembly_receipt_hash"),
            rom_receipt_hash=_required_str(payload, "rom_receipt_hash"),
            task_profile_hash=_required_str(payload, "task_profile_hash"),
            predicted_field_hash=_required_str(payload, "predicted_field_hash"),
            measured_field_hash=_required_str(payload, "measured_field_hash"),
            measurement_evidence_hash=_required_str(payload, "measurement_evidence_hash"),
            sample_count=_non_negative_int(payload.get("sample_count"), "sample_count"),
            task_coverage=_fraction(payload.get("task_coverage"), "task_coverage"),
            field_residuals=_float_mapping(payload.get("field_residuals", {}), "field_residuals"),
            residual_thresholds=_float_mapping(
                payload.get("residual_thresholds", {}),
                "residual_thresholds",
            ),
            promotion=_normalize_promotion(payload.get("promotion")),
            blocked_consumers=_string_list(payload.get("blocked_consumers", []), "blocked_consumers"),
            blocker_log=_string_list(payload.get("blocker_log", []), "blocker_log"),
            claim_boundary=str(payload.get("claim_boundary", WEAR_FIT_CLAIM_BOUNDARY)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "WearFitReceipt":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("WearFitReceipt JSON must contain an object")
        return cls.from_mapping(payload)

    def to_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


@dataclass(frozen=True, slots=True)
class TriadicTransportWitness:
    """Measured provenance/equivariance witness for C3/BT369 transport."""

    transport_model: str
    address_schema: str
    phase_provenance_hash: str
    orientation_receipt_hash: str
    c3_equivariance_error: float
    c3_equivariance_threshold: float

    def __post_init__(self) -> None:
        for key in (
            "transport_model",
            "address_schema",
            "phase_provenance_hash",
            "orientation_receipt_hash",
        ):
            value = getattr(self, key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"field '{key}' must be a non-empty string")
        object.__setattr__(
            self,
            "c3_equivariance_error",
            _non_negative_float(self.c3_equivariance_error, "c3_equivariance_error"),
        )
        object.__setattr__(
            self,
            "c3_equivariance_threshold",
            _non_negative_float(self.c3_equivariance_threshold, "c3_equivariance_threshold"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "transport_model": self.transport_model,
            "address_schema": self.address_schema,
            "phase_provenance_hash": self.phase_provenance_hash,
            "orientation_receipt_hash": self.orientation_receipt_hash,
            "c3_equivariance_error": self.c3_equivariance_error,
            "c3_equivariance_threshold": self.c3_equivariance_threshold,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TriadicTransportWitness":
        return cls(
            transport_model=_required_str(payload, "transport_model"),
            address_schema=_required_str(payload, "address_schema"),
            phase_provenance_hash=_required_str(payload, "phase_provenance_hash"),
            orientation_receipt_hash=_required_str(payload, "orientation_receipt_hash"),
            c3_equivariance_error=_non_negative_float(
                payload.get("c3_equivariance_error"),
                "c3_equivariance_error",
            ),
            c3_equivariance_threshold=_non_negative_float(
                payload.get("c3_equivariance_threshold"),
                "c3_equivariance_threshold",
            ),
        )


@dataclass(frozen=True, slots=True)
class PhysicalROMValidationReceipt:
    """Closure of declared physical wear evidence back onto the ROM model."""

    wear_fit_receipt_hash: str
    source_rom_receipt_hash: str
    validation_evidence_hash: str
    field_residual_hash: str
    closure_iteration: int
    calibrated_rom_receipt_hash: str | None
    triadic_transport: TriadicTransportWitness | None
    promotion: Promotion
    blocked_consumers: list[str]
    blocker_log: list[str]
    claim_boundary: str = PHYSICAL_ROM_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        for key in (
            "wear_fit_receipt_hash",
            "source_rom_receipt_hash",
            "validation_evidence_hash",
            "field_residual_hash",
            "claim_boundary",
        ):
            value = getattr(self, key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"field '{key}' must be a non-empty string")
        object.__setattr__(
            self,
            "closure_iteration",
            _non_negative_int(self.closure_iteration, "closure_iteration"),
        )
        if self.calibrated_rom_receipt_hash is not None and not self.calibrated_rom_receipt_hash:
            raise ValueError("calibrated_rom_receipt_hash must be non-empty when present")
        promotion = _normalize_promotion(self.promotion)
        object.__setattr__(self, "promotion", promotion)
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked(
                promotion,
                _string_list(self.blocked_consumers, "blocked_consumers"),
                DEFAULT_PHYSICAL_ROM_BLOCKED_CONSUMERS,
            ),
        )
        object.__setattr__(self, "blocker_log", _string_list(self.blocker_log, "blocker_log"))

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": PHYSICAL_ROM_SCHEMA_VERSION,
            "wear_fit_receipt_hash": self.wear_fit_receipt_hash,
            "source_rom_receipt_hash": self.source_rom_receipt_hash,
            "validation_evidence_hash": self.validation_evidence_hash,
            "field_residual_hash": self.field_residual_hash,
            "closure_iteration": self.closure_iteration,
            "calibrated_rom_receipt_hash": self.calibrated_rom_receipt_hash,
            "promotion": int(self.promotion),
            "promotion_state": _promotion_name(self.promotion),
            "blocked_consumers": list(self.blocked_consumers),
            "blocker_log": list(self.blocker_log),
            "claim_boundary": self.claim_boundary,
        }
        payload["triadic_transport"] = (
            self.triadic_transport.to_dict() if self.triadic_transport is not None else None
        )
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PhysicalROMValidationReceipt":
        transport_payload = payload.get("triadic_transport")
        if transport_payload is None:
            transport = None
        elif isinstance(transport_payload, Mapping):
            transport = TriadicTransportWitness.from_mapping(transport_payload)
        else:
            raise TypeError("field 'triadic_transport' must be an object or null")
        return cls(
            wear_fit_receipt_hash=_required_str(payload, "wear_fit_receipt_hash"),
            source_rom_receipt_hash=_required_str(payload, "source_rom_receipt_hash"),
            validation_evidence_hash=_required_str(payload, "validation_evidence_hash"),
            field_residual_hash=_required_str(payload, "field_residual_hash"),
            closure_iteration=_non_negative_int(
                payload.get("closure_iteration"),
                "closure_iteration",
            ),
            calibrated_rom_receipt_hash=_optional_str(payload, "calibrated_rom_receipt_hash"),
            triadic_transport=transport,
            promotion=_normalize_promotion(payload.get("promotion")),
            blocked_consumers=_string_list(payload.get("blocked_consumers", []), "blocked_consumers"),
            blocker_log=_string_list(payload.get("blocker_log", []), "blocker_log"),
            claim_boundary=str(payload.get("claim_boundary", PHYSICAL_ROM_CLAIM_BOUNDARY)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PhysicalROMValidationReceipt":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("PhysicalROMValidationReceipt JSON must contain an object")
        return cls.from_mapping(payload)

    def to_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


def derive_assembly_receipt(
    *,
    finished_seam_receipt_hash: str,
    finished_seam_receipt: FinishedSeamReceipt,
    manufactured_artifacts_hash: str,
    assembled_geometry_hash: str,
    assembly_evidence_hash: str,
    piece_count: int,
    realized_seam_count: int,
    max_geometry_deviation: float,
    geometry_deviation_threshold: float,
    material_lot_hashes: list[str] | None = None,
) -> AssemblyReceipt:
    blockers: list[str] = []
    hard_reject = False
    if not can_consume_finished_seam_receipt(finished_seam_receipt, "assembly"):
        blockers.append("finished_seam_receipt_not_promoted_for_assembly")
    if piece_count <= 0:
        blockers.append("assembly_piece_count_missing")
    if realized_seam_count != finished_seam_receipt.selected_seam_count:
        blockers.append("realized_seam_count_mismatch")
    if max_geometry_deviation > geometry_deviation_threshold:
        blockers.append("assembly_geometry_deviation_exceeds_threshold")
        hard_reject = True
    lots = material_lot_hashes or []
    if not lots:
        blockers.append("material_lot_provenance_missing")

    promotion: Promotion
    if hard_reject:
        promotion = -1
    elif blockers:
        promotion = 0
    else:
        promotion = 1
    return AssemblyReceipt(
        finished_seam_receipt_hash=finished_seam_receipt_hash,
        manufactured_artifacts_hash=manufactured_artifacts_hash,
        assembled_geometry_hash=assembled_geometry_hash,
        assembly_evidence_hash=assembly_evidence_hash,
        piece_count=piece_count,
        realized_seam_count=realized_seam_count,
        max_geometry_deviation=max_geometry_deviation,
        geometry_deviation_threshold=geometry_deviation_threshold,
        material_lot_hashes=lots,
        promotion=promotion,
        blocked_consumers=[],
        blocker_log=blockers,
    )


def derive_wear_fit_receipt(
    *,
    assembly_receipt_hash: str,
    assembly_receipt: AssemblyReceipt,
    rom_receipt_hash: str,
    task_profile_hash: str,
    predicted_field_hash: str,
    measured_field_hash: str,
    measurement_evidence_hash: str,
    sample_count: int,
    task_coverage: float,
    field_residuals: Mapping[str, float],
    residual_thresholds: Mapping[str, float],
    minimum_task_coverage: float = 0.95,
) -> WearFitReceipt:
    blockers: list[str] = []
    hard_reject = False
    if not can_consume_assembly_receipt(assembly_receipt, "wear_fit"):
        blockers.append("assembly_receipt_not_promoted_for_wear_fit")
    if sample_count <= 0:
        blockers.append("wear_fit_samples_missing")
    if task_coverage < minimum_task_coverage:
        blockers.append("task_coverage_below_threshold")
    residuals = {str(name): float(value) for name, value in field_residuals.items()}
    thresholds = {str(name): float(value) for name, value in residual_thresholds.items()}
    missing_thresholds = sorted(set(residuals) - set(thresholds))
    if missing_thresholds:
        blockers.extend(f"missing_residual_threshold:{name}" for name in missing_thresholds)
    if not residuals:
        blockers.append("field_residuals_missing")
    for name, threshold in thresholds.items():
        if name in residuals and residuals[name] > threshold:
            blockers.append(f"field_residual_exceeds_threshold:{name}")
            hard_reject = True

    promotion: Promotion
    if hard_reject:
        promotion = -1
    elif blockers:
        promotion = 0
    else:
        promotion = 1
    return WearFitReceipt(
        assembly_receipt_hash=assembly_receipt_hash,
        rom_receipt_hash=rom_receipt_hash,
        task_profile_hash=task_profile_hash,
        predicted_field_hash=predicted_field_hash,
        measured_field_hash=measured_field_hash,
        measurement_evidence_hash=measurement_evidence_hash,
        sample_count=sample_count,
        task_coverage=task_coverage,
        field_residuals=residuals,
        residual_thresholds=thresholds,
        promotion=promotion,
        blocked_consumers=[],
        blocker_log=blockers,
    )


def derive_physical_rom_validation_receipt(
    *,
    wear_fit_receipt_hash: str,
    wear_fit_receipt: WearFitReceipt,
    source_rom_receipt_hash: str,
    validation_evidence_hash: str,
    field_residual_hash: str,
    closure_iteration: int,
    calibrated_rom_receipt_hash: str | None = None,
    triadic_transport: TriadicTransportWitness | None = None,
    require_triadic_transport: bool = True,
) -> PhysicalROMValidationReceipt:
    blockers: list[str] = []
    hard_reject = False
    if not can_consume_wear_fit_receipt(wear_fit_receipt, "physical_rom_validation"):
        blockers.append("wear_fit_receipt_not_promoted_for_physical_rom_validation")
    if triadic_transport is None:
        if require_triadic_transport:
            blockers.append("triadic_transport_witness_missing")
    elif triadic_transport.c3_equivariance_error > triadic_transport.c3_equivariance_threshold:
        blockers.append("c3_equivariance_error_exceeds_threshold")
        hard_reject = True

    promotion: Promotion
    if hard_reject:
        promotion = -1
    elif blockers:
        promotion = 0
    else:
        promotion = 1
    return PhysicalROMValidationReceipt(
        wear_fit_receipt_hash=wear_fit_receipt_hash,
        source_rom_receipt_hash=source_rom_receipt_hash,
        validation_evidence_hash=validation_evidence_hash,
        field_residual_hash=field_residual_hash,
        closure_iteration=closure_iteration,
        calibrated_rom_receipt_hash=calibrated_rom_receipt_hash,
        triadic_transport=triadic_transport,
        promotion=promotion,
        blocked_consumers=[],
        blocker_log=blockers,
    )


def can_consume_assembly_receipt(receipt: AssemblyReceipt, consumer: str | None = None) -> bool:
    if receipt.promotion != 1:
        return False
    if consumer is None:
        return not receipt.blocked_consumers
    return consumer not in receipt.blocked_consumers


def can_consume_wear_fit_receipt(receipt: WearFitReceipt, consumer: str | None = None) -> bool:
    if receipt.promotion != 1:
        return False
    if consumer is None:
        return not receipt.blocked_consumers
    return consumer not in receipt.blocked_consumers


def can_consume_physical_rom_validation_receipt(
    receipt: PhysicalROMValidationReceipt,
    consumer: str | None = None,
) -> bool:
    if receipt.promotion != 1:
        return False
    if consumer is None:
        return not receipt.blocked_consumers
    return consumer not in receipt.blocked_consumers


def load_assembly_receipt(path: str | Path) -> AssemblyReceipt:
    return AssemblyReceipt.from_json(path)


def load_wear_fit_receipt(path: str | Path) -> WearFitReceipt:
    return WearFitReceipt.from_json(path)


def load_physical_rom_validation_receipt(path: str | Path) -> PhysicalROMValidationReceipt:
    return PhysicalROMValidationReceipt.from_json(path)
