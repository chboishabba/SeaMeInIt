"""Typed receipts for canonical field-basis provenance."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

Promotion = Literal[-1, 0, 1]
DEFAULT_BASIS_BLOCKED_CONSUMERS = (
    "rom_field_aggregation",
    "seam_cost_field",
    "solver_promotion",
)

__all__ = [
    "BasisReceipt",
    "DEFAULT_BASIS_BLOCKED_CONSUMERS",
    "Promotion",
    "can_consume_basis_receipt",
    "load_basis_receipt",
    "normalize_promotion",
    "with_promotion",
]


def normalize_promotion(value: object) -> Promotion:
    """Coerce and validate a receipt promotion state."""

    if isinstance(value, bool):
        raise ValueError("BasisReceipt promotion must be one of -1, 0, or 1.")
    if isinstance(value, int):
        promotion = value
    elif isinstance(value, float) and value.is_integer():
        promotion = int(value)
    elif isinstance(value, str) and value in {"-1", "0", "1"}:
        promotion = int(value)
    else:
        raise ValueError("BasisReceipt promotion must be one of -1, 0, or 1.")
    if promotion not in {-1, 0, 1}:
        raise ValueError("BasisReceipt promotion must be one of -1, 0, or 1.")
    return cast(Promotion, promotion)


def _coerce_required_str(payload: Mapping[str, Any], key: str) -> str:
    try:
        value = payload[key]
    except KeyError as exc:
        raise KeyError(f"BasisReceipt is missing required field '{key}'.") from exc
    return _coerce_str_value(value, key)


def _coerce_str_value(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"BasisReceipt field '{key}' must be a string.")
    if not value:
        raise ValueError(f"BasisReceipt field '{key}' must be non-empty.")
    return value


def _coerce_positive_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        value = payload[key]
    except KeyError as exc:
        raise KeyError(f"BasisReceipt is missing required field '{key}'.") from exc
    return _coerce_positive_int_value(value, key)


def _coerce_positive_int_value(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"BasisReceipt field '{key}' must be an integer.")
    try:
        coerced = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"BasisReceipt field '{key}' must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"BasisReceipt field '{key}' must be an integer.")
    if coerced <= 0:
        raise ValueError(f"BasisReceipt field '{key}' must be positive.")
    return coerced


def _coerce_non_negative_finite_float(payload: Mapping[str, Any], key: str) -> float:
    try:
        value = payload[key]
    except KeyError as exc:
        raise KeyError(f"BasisReceipt is missing required field '{key}'.") from exc
    return _coerce_non_negative_finite_float_value(value, key)


def _coerce_non_negative_finite_float_value(value: object, key: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"BasisReceipt field '{key}' must be numeric.")
    try:
        coerced = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"BasisReceipt field '{key}' must be numeric.") from exc
    if not math.isfinite(coerced):
        raise ValueError(f"BasisReceipt field '{key}' must be finite.")
    if coerced < 0.0:
        raise ValueError(f"BasisReceipt field '{key}' must be non-negative.")
    return coerced


def _coerce_promotion(payload: Mapping[str, Any]) -> Promotion:
    try:
        value = payload["promotion"]
    except KeyError as exc:
        raise KeyError("BasisReceipt is missing required field 'promotion'.") from exc
    return normalize_promotion(value)


def _coerce_blocked_consumers(payload: Mapping[str, Any]) -> list[str]:
    blocked_consumers = payload.get("blocked_consumers", [])
    return _coerce_blocked_consumer_values(blocked_consumers)


def _coerce_blocked_consumer_values(blocked_consumers: object) -> list[str]:
    if not isinstance(blocked_consumers, list):
        raise TypeError("BasisReceipt field 'blocked_consumers' must be a list.")
    return [str(consumer) for consumer in blocked_consumers]


def _blocked_consumers_for_promotion(
    promotion: Promotion, blocked_consumers: list[str]
) -> list[str]:
    if promotion != 1 and not blocked_consumers:
        return list(DEFAULT_BASIS_BLOCKED_CONSUMERS)
    return blocked_consumers


@dataclass(frozen=True, slots=True)
class BasisReceipt:
    """Hash-linked B0 field-basis receipt used to gate downstream consumers."""

    carrier_receipt_hash: str
    basis_vertex_count: int
    basis_dimension: int
    construction_method: str
    reconstruction_error: float
    promotion: Promotion
    blocked_consumers: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "carrier_receipt_hash",
            _coerce_str_value(self.carrier_receipt_hash, "carrier_receipt_hash"),
        )
        object.__setattr__(
            self,
            "basis_vertex_count",
            _coerce_positive_int_value(self.basis_vertex_count, "basis_vertex_count"),
        )
        object.__setattr__(
            self,
            "basis_dimension",
            _coerce_positive_int_value(self.basis_dimension, "basis_dimension"),
        )
        object.__setattr__(
            self,
            "construction_method",
            _coerce_str_value(self.construction_method, "construction_method"),
        )
        object.__setattr__(
            self,
            "reconstruction_error",
            _coerce_non_negative_finite_float_value(
                self.reconstruction_error,
                "reconstruction_error",
            ),
        )
        object.__setattr__(self, "promotion", normalize_promotion(self.promotion))
        blocked_consumers = _coerce_blocked_consumer_values(self.blocked_consumers)
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked_consumers_for_promotion(self.promotion, blocked_consumers),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BasisReceipt":
        """Build a validated receipt from a JSON-like mapping."""

        return cls(
            carrier_receipt_hash=_coerce_required_str(payload, "carrier_receipt_hash"),
            basis_vertex_count=_coerce_positive_int(payload, "basis_vertex_count"),
            basis_dimension=_coerce_positive_int(payload, "basis_dimension"),
            construction_method=_coerce_required_str(payload, "construction_method"),
            reconstruction_error=_coerce_non_negative_finite_float(
                payload,
                "reconstruction_error",
            ),
            promotion=_coerce_promotion(payload),
            blocked_consumers=_coerce_blocked_consumers(payload),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "BasisReceipt":
        """Load a receipt from a JSON document."""

        receipt_path = Path(path)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("BasisReceipt JSON must contain an object.")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable receipt payload."""

        return {
            "carrier_receipt_hash": self.carrier_receipt_hash,
            "basis_vertex_count": int(self.basis_vertex_count),
            "basis_dimension": int(self.basis_dimension),
            "construction_method": self.construction_method,
            "reconstruction_error": float(self.reconstruction_error),
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


def load_basis_receipt(path: str | Path) -> BasisReceipt:
    """Load a basis receipt from JSON."""

    return BasisReceipt.from_json(path)


def with_promotion(receipt: BasisReceipt, promotion: object) -> BasisReceipt:
    """Return a copy of a receipt with a validated promotion value."""

    next_promotion = normalize_promotion(promotion)
    return replace(
        receipt,
        promotion=next_promotion,
        blocked_consumers=_blocked_consumers_for_promotion(
            next_promotion, receipt.blocked_consumers
        ),
    )


def can_consume_basis_receipt(
    receipt: BasisReceipt, consumer: str | None = None
) -> bool:
    """Return whether a basis receipt is promoted for downstream consumers."""

    if receipt.promotion != 1:
        return False
    if consumer is None:
        return not receipt.blocked_consumers
    return consumer not in receipt.blocked_consumers
