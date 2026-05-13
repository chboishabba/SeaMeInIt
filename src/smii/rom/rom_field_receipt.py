"""Typed receipts for ROM field aggregation promotion decisions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

Promotion = Literal[-1, 0, 1]
DEFAULT_ROM_FIELD_BLOCKED_CONSUMERS = (
    "seam_cost_field",
    "solver_promotion",
    "panel_unwrap",
)

__all__ = [
    "DEFAULT_ROM_FIELD_BLOCKED_CONSUMERS",
    "Promotion",
    "ROMFieldReceipt",
    "can_consume_rom_field_receipt",
    "load_rom_field_receipt",
    "normalize_promotion",
    "with_promotion",
]


def normalize_promotion(value: object) -> Promotion:
    """Coerce and validate a receipt promotion state."""

    if isinstance(value, bool):
        raise ValueError("ROMFieldReceipt promotion must be one of -1, 0, or 1.")
    if isinstance(value, int):
        promotion = value
    elif isinstance(value, float) and value.is_integer():
        promotion = int(value)
    elif isinstance(value, str) and value in {"-1", "0", "1"}:
        promotion = int(value)
    else:
        raise ValueError("ROMFieldReceipt promotion must be one of -1, 0, or 1.")
    if promotion not in {-1, 0, 1}:
        raise ValueError("ROMFieldReceipt promotion must be one of -1, 0, or 1.")
    return cast(Promotion, promotion)


def _coerce_required_str(payload: Mapping[str, Any], key: str) -> str:
    try:
        value = payload[key]
    except KeyError as exc:
        raise KeyError(f"ROMFieldReceipt is missing required field '{key}'.") from exc
    return _coerce_str_value(value, key)


def _coerce_str_value(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"ROMFieldReceipt field '{key}' must be a string.")
    if not value:
        raise ValueError(f"ROMFieldReceipt field '{key}' must be non-empty.")
    return value


def _coerce_non_negative_finite_float_value(value: object, key: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"ROMFieldReceipt field '{key}' must be numeric.")
    try:
        coerced = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"ROMFieldReceipt field '{key}' must be numeric.") from exc
    if not math.isfinite(coerced):
        raise ValueError(f"ROMFieldReceipt field '{key}' must be finite.")
    if coerced < 0.0:
        raise ValueError(f"ROMFieldReceipt field '{key}' must be non-negative.")
    return coerced


def _coerce_non_negative_finite_float(payload: Mapping[str, Any], key: str) -> float:
    try:
        value = payload[key]
    except KeyError as exc:
        raise KeyError(f"ROMFieldReceipt is missing required field '{key}'.") from exc
    return _coerce_non_negative_finite_float_value(value, key)


def _coerce_unit_interval_float(payload: Mapping[str, Any], key: str) -> float:
    value = _coerce_non_negative_finite_float(payload, key)
    if value > 1.0:
        raise ValueError(f"ROMFieldReceipt field '{key}' must be <= 1.0.")
    return value


def _coerce_positive_int_value(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"ROMFieldReceipt field '{key}' must be an integer.")
    try:
        coerced = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"ROMFieldReceipt field '{key}' must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"ROMFieldReceipt field '{key}' must be an integer.")
    if coerced <= 0:
        raise ValueError(f"ROMFieldReceipt field '{key}' must be positive.")
    return coerced


def _coerce_positive_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        value = payload[key]
    except KeyError as exc:
        raise KeyError(f"ROMFieldReceipt is missing required field '{key}'.") from exc
    return _coerce_positive_int_value(value, key)


def _coerce_string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    try:
        value = payload[key]
    except KeyError as exc:
        raise KeyError(f"ROMFieldReceipt is missing required field '{key}'.") from exc
    if not isinstance(value, list):
        raise TypeError(f"ROMFieldReceipt field '{key}' must be a list.")
    fields = [str(field) for field in value]
    if not fields:
        raise ValueError(f"ROMFieldReceipt field '{key}' must be non-empty.")
    return fields


def _coerce_bool(payload: Mapping[str, Any], key: str) -> bool:
    try:
        value = payload[key]
    except KeyError as exc:
        raise KeyError(f"ROMFieldReceipt is missing required field '{key}'.") from exc
    if not isinstance(value, bool):
        raise TypeError(f"ROMFieldReceipt field '{key}' must be boolean.")
    return value


def _coerce_promotion(payload: Mapping[str, Any]) -> Promotion:
    try:
        value = payload["promotion"]
    except KeyError as exc:
        raise KeyError("ROMFieldReceipt is missing required field 'promotion'.") from exc
    return normalize_promotion(value)


def _coerce_blocked_consumers(payload: Mapping[str, Any]) -> list[str]:
    blocked_consumers = payload.get("blocked_consumers", [])
    return _coerce_blocked_consumer_values(blocked_consumers)


def _coerce_blocked_consumer_values(blocked_consumers: object) -> list[str]:
    if not isinstance(blocked_consumers, list):
        raise TypeError("ROMFieldReceipt field 'blocked_consumers' must be a list.")
    return [str(consumer) for consumer in blocked_consumers]


def _blocked_consumers_for_promotion(
    promotion: Promotion, blocked_consumers: list[str]
) -> list[str]:
    if promotion != 1 and not blocked_consumers:
        return list(DEFAULT_ROM_FIELD_BLOCKED_CONSUMERS)
    return blocked_consumers


@dataclass(frozen=True, slots=True)
class ROMFieldReceipt:
    """Hash-linked receipt for vertex-aligned ROM field aggregation."""

    basis_receipt_hash: str
    samples_hash: str
    aggregation_summary_hash: str
    fields_hash: str
    pose_count: int
    total_samples: int
    pose_source: str
    fields_computed: list[str]
    vertex_count: int
    peak_pressure_max: float
    peak_pressure_percentile95: float
    field_uniformity: float
    synthetic: bool
    promotion: Promotion
    blocked_consumers: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "basis_receipt_hash",
            _coerce_str_value(self.basis_receipt_hash, "basis_receipt_hash"),
        )
        object.__setattr__(
            self,
            "samples_hash",
            _coerce_str_value(self.samples_hash, "samples_hash"),
        )
        object.__setattr__(
            self,
            "aggregation_summary_hash",
            _coerce_str_value(
                self.aggregation_summary_hash,
                "aggregation_summary_hash",
            ),
        )
        object.__setattr__(
            self,
            "fields_hash",
            _coerce_str_value(self.fields_hash, "fields_hash"),
        )
        object.__setattr__(
            self,
            "pose_count",
            _coerce_positive_int_value(self.pose_count, "pose_count"),
        )
        object.__setattr__(
            self,
            "total_samples",
            _coerce_positive_int_value(self.total_samples, "total_samples"),
        )
        object.__setattr__(self, "pose_source", _coerce_str_value(self.pose_source, "pose_source"))
        fields_computed = [str(field) for field in self.fields_computed]
        if not fields_computed:
            raise ValueError("ROMFieldReceipt field 'fields_computed' must be non-empty.")
        object.__setattr__(self, "fields_computed", fields_computed)
        object.__setattr__(
            self,
            "vertex_count",
            _coerce_positive_int_value(self.vertex_count, "vertex_count"),
        )
        object.__setattr__(
            self,
            "peak_pressure_max",
            _coerce_non_negative_finite_float_value(
                self.peak_pressure_max,
                "peak_pressure_max",
            ),
        )
        object.__setattr__(
            self,
            "peak_pressure_percentile95",
            _coerce_non_negative_finite_float_value(
                self.peak_pressure_percentile95,
                "peak_pressure_percentile95",
            ),
        )
        uniformity = _coerce_non_negative_finite_float_value(
            self.field_uniformity,
            "field_uniformity",
        )
        if uniformity > 1.0:
            raise ValueError("ROMFieldReceipt field 'field_uniformity' must be <= 1.0.")
        object.__setattr__(self, "field_uniformity", uniformity)
        if not isinstance(self.synthetic, bool):
            raise TypeError("ROMFieldReceipt field 'synthetic' must be boolean.")
        object.__setattr__(self, "promotion", normalize_promotion(self.promotion))
        blocked_consumers = _coerce_blocked_consumer_values(self.blocked_consumers)
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked_consumers_for_promotion(self.promotion, blocked_consumers),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ROMFieldReceipt":
        """Build a validated receipt from a JSON-like mapping."""

        return cls(
            basis_receipt_hash=_coerce_required_str(payload, "basis_receipt_hash"),
            samples_hash=_coerce_required_str(payload, "samples_hash"),
            aggregation_summary_hash=_coerce_required_str(
                payload,
                "aggregation_summary_hash",
            ),
            fields_hash=_coerce_required_str(payload, "fields_hash"),
            pose_count=_coerce_positive_int(payload, "pose_count"),
            total_samples=_coerce_positive_int(payload, "total_samples"),
            pose_source=_coerce_required_str(payload, "pose_source"),
            fields_computed=_coerce_string_list(payload, "fields_computed"),
            vertex_count=_coerce_positive_int(payload, "vertex_count"),
            peak_pressure_max=_coerce_non_negative_finite_float(
                payload,
                "peak_pressure_max",
            ),
            peak_pressure_percentile95=_coerce_non_negative_finite_float(
                payload,
                "peak_pressure_percentile95",
            ),
            field_uniformity=_coerce_unit_interval_float(payload, "field_uniformity"),
            synthetic=_coerce_bool(payload, "synthetic"),
            promotion=_coerce_promotion(payload),
            blocked_consumers=_coerce_blocked_consumers(payload),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ROMFieldReceipt":
        """Load a receipt from a JSON document."""

        receipt_path = Path(path)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("ROMFieldReceipt JSON must contain an object.")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable receipt payload."""

        return {
            "basis_receipt_hash": self.basis_receipt_hash,
            "samples_hash": self.samples_hash,
            "aggregation_summary_hash": self.aggregation_summary_hash,
            "fields_hash": self.fields_hash,
            "pose_count": int(self.pose_count),
            "total_samples": int(self.total_samples),
            "pose_source": self.pose_source,
            "fields_computed": list(self.fields_computed),
            "vertex_count": int(self.vertex_count),
            "peak_pressure_max": float(self.peak_pressure_max),
            "peak_pressure_percentile95": float(self.peak_pressure_percentile95),
            "field_uniformity": float(self.field_uniformity),
            "synthetic": bool(self.synthetic),
            "promotion": int(self.promotion),
            "blocked_consumers": list(self.blocked_consumers),
        }

    def to_json(self, path: str | Path) -> Path:
        """Write the receipt as stable JSON and return the target path."""

        receipt_path = Path(path)
        receipt_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return receipt_path


def load_rom_field_receipt(path: str | Path) -> ROMFieldReceipt:
    """Load a ROM field receipt from JSON."""

    return ROMFieldReceipt.from_json(path)


def with_promotion(receipt: ROMFieldReceipt, promotion: object) -> ROMFieldReceipt:
    """Return a copy of a receipt with a validated promotion value."""

    next_promotion = normalize_promotion(promotion)
    return replace(
        receipt,
        promotion=next_promotion,
        blocked_consumers=_blocked_consumers_for_promotion(
            next_promotion,
            receipt.blocked_consumers,
        ),
    )


def can_consume_rom_field_receipt(
    receipt: ROMFieldReceipt, consumer: str | None = None
) -> bool:
    """Return whether a ROM field receipt is promoted for downstream consumers."""

    if receipt.promotion != 1:
        return False
    if consumer is None:
        return not receipt.blocked_consumers
    return consumer not in receipt.blocked_consumers
