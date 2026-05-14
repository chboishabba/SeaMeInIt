"""Typed receipts for promoted manufacturing artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

Promotion = Literal[-1, 0, 1]
MANUFACTURING_METHODS = (
    "home_sewing",
    "overlock",
    "flatlock",
    "bonded",
    "welded",
    "laser_cut",
    "3d_print",
    "eva_foam_cut",
)
ACCESSIBILITY_LEVELS = ("consumer", "industrial", "advanced")
GRAIN_DIRECTIONS = ("warp", "weft", "bias")

__all__ = [
    "ACCESSIBILITY_LEVELS",
    "GRAIN_DIRECTIONS",
    "MANUFACTURING_METHODS",
    "ManufacturingReceipt",
    "Promotion",
    "can_consume_manufacturing_receipt",
    "load_manufacturing_receipt",
    "normalize_promotion",
    "with_promotion",
]


def normalize_promotion(value: object) -> Promotion:
    """Coerce and validate a receipt promotion state."""

    if isinstance(value, bool):
        raise ValueError("ManufacturingReceipt promotion must be one of -1, 0, or 1.")
    if isinstance(value, int):
        promotion = value
    elif isinstance(value, float) and value.is_integer():
        promotion = int(value)
    elif isinstance(value, str) and value in {"-1", "0", "1"}:
        promotion = int(value)
    else:
        raise ValueError("ManufacturingReceipt promotion must be one of -1, 0, or 1.")
    if promotion not in {-1, 0, 1}:
        raise ValueError("ManufacturingReceipt promotion must be one of -1, 0, or 1.")
    return cast(Promotion, promotion)


def _missing(key: str) -> KeyError:
    return KeyError(f"ManufacturingReceipt is missing required field '{key}'.")


def _coerce_required_str(payload: Mapping[str, Any], key: str) -> str:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_str_value(value, key)


def _coerce_str_value(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"ManufacturingReceipt field '{key}' must be a string.")
    if not value:
        raise ValueError(f"ManufacturingReceipt field '{key}' must be non-empty.")
    return value


def _coerce_optional_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"ManufacturingReceipt field '{key}' must be a string.")
    return value


def _coerce_bool(payload: Mapping[str, Any], key: str) -> bool:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    if not isinstance(value, bool):
        raise TypeError(f"ManufacturingReceipt field '{key}' must be a boolean.")
    return value


def _coerce_non_negative_int_value(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"ManufacturingReceipt field '{key}' must be an integer.")
    try:
        coerced = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"ManufacturingReceipt field '{key}' must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"ManufacturingReceipt field '{key}' must be an integer.")
    if coerced < 0:
        raise ValueError(f"ManufacturingReceipt field '{key}' must be non-negative.")
    return coerced


def _coerce_non_negative_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_non_negative_int_value(value, key)


def _coerce_finite_float_value(value: object, key: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"ManufacturingReceipt field '{key}' must be numeric.")
    try:
        coerced = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"ManufacturingReceipt field '{key}' must be numeric.") from exc
    if not math.isfinite(coerced):
        raise ValueError(f"ManufacturingReceipt field '{key}' must be finite.")
    return coerced


def _coerce_non_negative_float_value(value: object, key: str) -> float:
    coerced = _coerce_finite_float_value(value, key)
    if coerced < 0.0:
        raise ValueError(f"ManufacturingReceipt field '{key}' must be non-negative.")
    return coerced


def _coerce_non_negative_float(payload: Mapping[str, Any], key: str) -> float:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_non_negative_float_value(value, key)


def _coerce_str_list(payload: Mapping[str, Any], key: str) -> list[str]:
    try:
        values = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_str_list_value(values, key)


def _coerce_str_list_value(values: object, key: str) -> list[str]:
    if not isinstance(values, list):
        raise TypeError(f"ManufacturingReceipt field '{key}' must be a list.")
    return [str(value) for value in values]


def _coerce_grain_directions(payload: Mapping[str, Any]) -> list[str]:
    directions = _coerce_str_list(payload, "grain_directions")
    invalid = [direction for direction in directions if direction not in GRAIN_DIRECTIONS]
    if invalid:
        raise ValueError(
            "ManufacturingReceipt field 'grain_directions' entries must be one of "
            f"{', '.join(GRAIN_DIRECTIONS)}."
        )
    return directions


def _coerce_blocked_consumers(payload: Mapping[str, Any]) -> list[str]:
    return _coerce_str_list_value(payload.get("blocked_consumers", []), "blocked_consumers")


def _coerce_promotion(payload: Mapping[str, Any]) -> Promotion:
    try:
        value = payload["promotion"]
    except KeyError as exc:
        raise _missing("promotion") from exc
    return normalize_promotion(value)


@dataclass(frozen=True, slots=True)
class ManufacturingReceipt:
    """Hash-linked receipt for final fabrication artifact promotion."""

    panel_unwrap_receipt_hash: str
    panel_count: int
    manufacturing_method: str
    accessibility_level: str
    seam_allowance_hash: str
    seam_allowance_mean: float
    seam_allowance_min: float
    seam_allowance_max: float
    allowance_varies: bool
    grain_directions: list[str]
    panel_hashes: list[str]
    cutting_artifacts_hash: str
    notches_present: bool
    labels_present: bool
    promotion: Promotion
    blocked_consumers: list[str]
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "panel_unwrap_receipt_hash",
            _coerce_str_value(
                self.panel_unwrap_receipt_hash,
                "panel_unwrap_receipt_hash",
            ),
        )
        panel_count = _coerce_non_negative_int_value(self.panel_count, "panel_count")
        object.__setattr__(self, "panel_count", panel_count)
        method = _coerce_str_value(self.manufacturing_method, "manufacturing_method")
        if method not in MANUFACTURING_METHODS:
            raise ValueError(
                "ManufacturingReceipt field 'manufacturing_method' must be one of "
                f"{', '.join(MANUFACTURING_METHODS)}."
            )
        object.__setattr__(self, "manufacturing_method", method)
        accessibility = _coerce_str_value(
            self.accessibility_level,
            "accessibility_level",
        )
        if accessibility not in ACCESSIBILITY_LEVELS:
            raise ValueError(
                "ManufacturingReceipt field 'accessibility_level' must be one of "
                f"{', '.join(ACCESSIBILITY_LEVELS)}."
            )
        object.__setattr__(self, "accessibility_level", accessibility)
        object.__setattr__(
            self,
            "seam_allowance_hash",
            _coerce_str_value(self.seam_allowance_hash, "seam_allowance_hash"),
        )
        allowance_mean = _coerce_non_negative_float_value(
            self.seam_allowance_mean,
            "seam_allowance_mean",
        )
        allowance_min = _coerce_non_negative_float_value(
            self.seam_allowance_min,
            "seam_allowance_min",
        )
        allowance_max = _coerce_non_negative_float_value(
            self.seam_allowance_max,
            "seam_allowance_max",
        )
        if allowance_min > allowance_mean or allowance_mean > allowance_max:
            raise ValueError(
                "ManufacturingReceipt seam allowance summary must satisfy "
                "min <= mean <= max."
            )
        object.__setattr__(self, "seam_allowance_mean", allowance_mean)
        object.__setattr__(self, "seam_allowance_min", allowance_min)
        object.__setattr__(self, "seam_allowance_max", allowance_max)
        if not isinstance(self.allowance_varies, bool):
            raise TypeError(
                "ManufacturingReceipt field 'allowance_varies' must be a boolean."
            )
        directions = _coerce_str_list_value(self.grain_directions, "grain_directions")
        invalid = [direction for direction in directions if direction not in GRAIN_DIRECTIONS]
        if invalid:
            raise ValueError(
                "ManufacturingReceipt field 'grain_directions' entries must be one of "
                f"{', '.join(GRAIN_DIRECTIONS)}."
            )
        if len(directions) != panel_count:
            raise ValueError(
                "ManufacturingReceipt field 'grain_directions' length must match "
                "panel_count."
            )
        object.__setattr__(self, "grain_directions", directions)
        panel_hashes = _coerce_str_list_value(self.panel_hashes, "panel_hashes")
        if len(panel_hashes) != panel_count:
            raise ValueError(
                "ManufacturingReceipt field 'panel_hashes' length must match panel_count."
            )
        object.__setattr__(self, "panel_hashes", panel_hashes)
        object.__setattr__(
            self,
            "cutting_artifacts_hash",
            _coerce_str_value(
                self.cutting_artifacts_hash,
                "cutting_artifacts_hash",
            ),
        )
        if not isinstance(self.notches_present, bool):
            raise TypeError(
                "ManufacturingReceipt field 'notches_present' must be a boolean."
            )
        if not isinstance(self.labels_present, bool):
            raise TypeError(
                "ManufacturingReceipt field 'labels_present' must be a boolean."
            )
        object.__setattr__(self, "promotion", normalize_promotion(self.promotion))
        blocked_consumers = _coerce_str_list_value(
            self.blocked_consumers,
            "blocked_consumers",
        )
        if self.promotion == 1 and blocked_consumers:
            raise ValueError(
                "ManufacturingReceipt is the end of chain; promoted receipts must not "
                "block consumers."
            )
        object.__setattr__(self, "blocked_consumers", blocked_consumers)
        if not isinstance(self.notes, str):
            raise TypeError("ManufacturingReceipt field 'notes' must be a string.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ManufacturingReceipt":
        """Build a validated receipt from a JSON-like mapping."""

        return cls(
            panel_unwrap_receipt_hash=_coerce_required_str(
                payload,
                "panel_unwrap_receipt_hash",
            ),
            panel_count=_coerce_non_negative_int(payload, "panel_count"),
            manufacturing_method=_coerce_required_str(payload, "manufacturing_method"),
            accessibility_level=_coerce_required_str(payload, "accessibility_level"),
            seam_allowance_hash=_coerce_required_str(payload, "seam_allowance_hash"),
            seam_allowance_mean=_coerce_non_negative_float(
                payload,
                "seam_allowance_mean",
            ),
            seam_allowance_min=_coerce_non_negative_float(
                payload,
                "seam_allowance_min",
            ),
            seam_allowance_max=_coerce_non_negative_float(
                payload,
                "seam_allowance_max",
            ),
            allowance_varies=_coerce_bool(payload, "allowance_varies"),
            grain_directions=_coerce_grain_directions(payload),
            panel_hashes=_coerce_str_list(payload, "panel_hashes"),
            cutting_artifacts_hash=_coerce_required_str(payload, "cutting_artifacts_hash"),
            notches_present=_coerce_bool(payload, "notches_present"),
            labels_present=_coerce_bool(payload, "labels_present"),
            promotion=_coerce_promotion(payload),
            blocked_consumers=_coerce_blocked_consumers(payload),
            notes=_coerce_optional_str(payload, "notes"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ManufacturingReceipt":
        """Load a receipt from a JSON document."""

        receipt_path = Path(path)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("ManufacturingReceipt JSON must contain an object.")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable receipt payload."""

        return {
            "panel_unwrap_receipt_hash": self.panel_unwrap_receipt_hash,
            "panel_count": int(self.panel_count),
            "manufacturing_method": self.manufacturing_method,
            "accessibility_level": self.accessibility_level,
            "seam_allowance_hash": self.seam_allowance_hash,
            "seam_allowance_mean": float(self.seam_allowance_mean),
            "seam_allowance_min": float(self.seam_allowance_min),
            "seam_allowance_max": float(self.seam_allowance_max),
            "allowance_varies": bool(self.allowance_varies),
            "grain_directions": list(self.grain_directions),
            "panel_hashes": list(self.panel_hashes),
            "cutting_artifacts_hash": self.cutting_artifacts_hash,
            "notches_present": bool(self.notches_present),
            "labels_present": bool(self.labels_present),
            "promotion": int(self.promotion),
            "blocked_consumers": list(self.blocked_consumers),
            "notes": self.notes,
        }

    def to_json(self, path: str | Path) -> Path:
        """Write the receipt as stable JSON and return the target path."""

        receipt_path = Path(path)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return receipt_path


def load_manufacturing_receipt(path: str | Path) -> ManufacturingReceipt:
    """Load a manufacturing receipt from JSON."""

    return ManufacturingReceipt.from_json(path)


def with_promotion(
    receipt: ManufacturingReceipt,
    promotion: object,
) -> ManufacturingReceipt:
    """Return a copy of a receipt with a validated promotion value."""

    return replace(receipt, promotion=normalize_promotion(promotion))


def can_consume_manufacturing_receipt(
    receipt: ManufacturingReceipt,
    consumer: str | None = None,
) -> bool:
    """Return whether final manufacturing artifacts are promoted."""

    if receipt.promotion != 1:
        return False
    if consumer is None:
        return not receipt.blocked_consumers
    return consumer not in receipt.blocked_consumers
