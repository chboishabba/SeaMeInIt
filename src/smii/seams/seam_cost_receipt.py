"""Typed receipts for promoted seam-cost artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

Promotion = Literal[-1, 0, 1]
DEFAULT_SEAM_COST_BLOCKED_CONSUMERS = (
    "solver_promotion",
    "panel_unwrap",
    "manufacturing",
)
SOLVE_DOMAINS = ("A_v3240", "B_v9438")

__all__ = [
    "DEFAULT_SEAM_COST_BLOCKED_CONSUMERS",
    "Promotion",
    "SOLVE_DOMAINS",
    "SeamCostReceipt",
    "can_consume_seam_cost_receipt",
    "load_seam_cost_receipt",
    "normalize_promotion",
    "with_promotion",
]


def normalize_promotion(value: object) -> Promotion:
    """Coerce and validate a receipt promotion state."""

    if isinstance(value, bool):
        raise ValueError("SeamCostReceipt promotion must be one of -1, 0, or 1.")
    if isinstance(value, int):
        promotion = value
    elif isinstance(value, float) and value.is_integer():
        promotion = int(value)
    elif isinstance(value, str) and value in {"-1", "0", "1"}:
        promotion = int(value)
    else:
        raise ValueError("SeamCostReceipt promotion must be one of -1, 0, or 1.")
    if promotion not in {-1, 0, 1}:
        raise ValueError("SeamCostReceipt promotion must be one of -1, 0, or 1.")
    return cast(Promotion, promotion)


def _missing(key: str) -> KeyError:
    return KeyError(f"SeamCostReceipt is missing required field '{key}'.")


def _coerce_required_str(payload: Mapping[str, Any], key: str) -> str:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_str_value(value, key)


def _coerce_str_value(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"SeamCostReceipt field '{key}' must be a string.")
    if not value:
        raise ValueError(f"SeamCostReceipt field '{key}' must be non-empty.")
    return value


def _coerce_optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    if key not in payload:
        raise _missing(key)
    value = payload[key]
    if value is None:
        return None
    return _coerce_str_value(value, key)


def _coerce_positive_int_value(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"SeamCostReceipt field '{key}' must be an integer.")
    try:
        coerced = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"SeamCostReceipt field '{key}' must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"SeamCostReceipt field '{key}' must be an integer.")
    if coerced <= 0:
        raise ValueError(f"SeamCostReceipt field '{key}' must be positive.")
    return coerced


def _coerce_non_negative_int_value(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"SeamCostReceipt field '{key}' must be an integer.")
    try:
        coerced = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"SeamCostReceipt field '{key}' must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"SeamCostReceipt field '{key}' must be an integer.")
    if coerced < 0:
        raise ValueError(f"SeamCostReceipt field '{key}' must be non-negative.")
    return coerced


def _coerce_positive_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_positive_int_value(value, key)


def _coerce_non_negative_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_non_negative_int_value(value, key)


def _coerce_finite_float_value(value: object, key: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"SeamCostReceipt field '{key}' must be numeric.")
    try:
        coerced = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"SeamCostReceipt field '{key}' must be numeric.") from exc
    if not math.isfinite(coerced):
        raise ValueError(f"SeamCostReceipt field '{key}' must be finite.")
    return coerced


def _coerce_non_negative_float(payload: Mapping[str, Any], key: str) -> float:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    coerced = _coerce_finite_float_value(value, key)
    if coerced < 0.0:
        raise ValueError(f"SeamCostReceipt field '{key}' must be non-negative.")
    return coerced


def _coerce_unit_interval_float(payload: Mapping[str, Any], key: str) -> float:
    coerced = _coerce_non_negative_float(payload, key)
    if coerced > 1.0:
        raise ValueError(f"SeamCostReceipt field '{key}' must be <= 1.0.")
    return coerced


def _coerce_weight_vector(payload: Mapping[str, Any]) -> dict[str, float]:
    try:
        weights = payload["weight_vector"]
    except KeyError as exc:
        raise _missing("weight_vector") from exc
    return _coerce_weight_values(weights)


def _coerce_weight_values(weights: object) -> dict[str, float]:
    if not isinstance(weights, Mapping):
        raise TypeError("SeamCostReceipt field 'weight_vector' must be an object.")
    coerced: dict[str, float] = {}
    for key, value in weights.items():
        coerced[str(key)] = _coerce_finite_float_value(value, f"weight_vector.{key}")
    if not coerced:
        raise ValueError("SeamCostReceipt field 'weight_vector' must be non-empty.")
    return coerced


def _coerce_blocked_consumers(payload: Mapping[str, Any]) -> list[str]:
    blocked_consumers = payload.get("blocked_consumers", [])
    return _coerce_blocked_consumer_values(blocked_consumers)


def _coerce_blocked_consumer_values(blocked_consumers: object) -> list[str]:
    if not isinstance(blocked_consumers, list):
        raise TypeError("SeamCostReceipt field 'blocked_consumers' must be a list.")
    return [str(consumer) for consumer in blocked_consumers]


def _coerce_promotion(payload: Mapping[str, Any]) -> Promotion:
    try:
        value = payload["promotion"]
    except KeyError as exc:
        raise _missing("promotion") from exc
    return normalize_promotion(value)


def _blocked_consumers_for_promotion(
    promotion: Promotion, blocked_consumers: list[str]
) -> list[str]:
    if promotion != 1 and not blocked_consumers:
        return list(DEFAULT_SEAM_COST_BLOCKED_CONSUMERS)
    return blocked_consumers


@dataclass(frozen=True, slots=True)
class SeamCostReceipt:
    """Hash-linked receipt for seam-cost promotion decisions."""

    rom_field_receipt_hash: str
    body_receipt_hash: str
    correspondence_receipt_hash: str | None
    solve_domain: str
    vertex_count: int
    edge_count: int
    finite_cost_coverage: float
    cost_uniformity: float
    peak_cost: float
    mean_cost: float
    weight_vector: dict[str, float]
    costs_hash: str
    promotion: Promotion
    blocked_consumers: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rom_field_receipt_hash",
            _coerce_str_value(self.rom_field_receipt_hash, "rom_field_receipt_hash"),
        )
        object.__setattr__(
            self,
            "body_receipt_hash",
            _coerce_str_value(self.body_receipt_hash, "body_receipt_hash"),
        )
        if self.correspondence_receipt_hash is not None:
            object.__setattr__(
                self,
                "correspondence_receipt_hash",
                _coerce_str_value(
                    self.correspondence_receipt_hash,
                    "correspondence_receipt_hash",
                ),
            )
        solve_domain = _coerce_str_value(self.solve_domain, "solve_domain")
        if solve_domain not in SOLVE_DOMAINS:
            raise ValueError(
                "SeamCostReceipt field 'solve_domain' must be one of "
                f"{', '.join(SOLVE_DOMAINS)}."
            )
        object.__setattr__(self, "solve_domain", solve_domain)
        object.__setattr__(
            self,
            "vertex_count",
            _coerce_positive_int_value(self.vertex_count, "vertex_count"),
        )
        object.__setattr__(
            self,
            "edge_count",
            _coerce_non_negative_int_value(self.edge_count, "edge_count"),
        )
        coverage = _coerce_finite_float_value(
            self.finite_cost_coverage,
            "finite_cost_coverage",
        )
        if not 0.0 <= coverage <= 1.0:
            raise ValueError(
                "SeamCostReceipt field 'finite_cost_coverage' must be between 0 and 1."
            )
        object.__setattr__(self, "finite_cost_coverage", coverage)
        uniformity = _coerce_finite_float_value(self.cost_uniformity, "cost_uniformity")
        if not 0.0 <= uniformity <= 1.0:
            raise ValueError("SeamCostReceipt field 'cost_uniformity' must be between 0 and 1.")
        object.__setattr__(self, "cost_uniformity", uniformity)
        object.__setattr__(
            self,
            "peak_cost",
            _coerce_non_negative_float({"peak_cost": self.peak_cost}, "peak_cost"),
        )
        object.__setattr__(
            self,
            "mean_cost",
            _coerce_non_negative_float({"mean_cost": self.mean_cost}, "mean_cost"),
        )
        object.__setattr__(self, "weight_vector", _coerce_weight_values(self.weight_vector))
        object.__setattr__(self, "costs_hash", _coerce_str_value(self.costs_hash, "costs_hash"))
        object.__setattr__(self, "promotion", normalize_promotion(self.promotion))
        blocked_consumers = _coerce_blocked_consumer_values(self.blocked_consumers)
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked_consumers_for_promotion(self.promotion, blocked_consumers),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SeamCostReceipt":
        """Build a validated receipt from a JSON-like mapping."""

        return cls(
            rom_field_receipt_hash=_coerce_required_str(payload, "rom_field_receipt_hash"),
            body_receipt_hash=_coerce_required_str(payload, "body_receipt_hash"),
            correspondence_receipt_hash=_coerce_optional_str(
                payload,
                "correspondence_receipt_hash",
            ),
            solve_domain=_coerce_required_str(payload, "solve_domain"),
            vertex_count=_coerce_positive_int(payload, "vertex_count"),
            edge_count=_coerce_non_negative_int(payload, "edge_count"),
            finite_cost_coverage=_coerce_unit_interval_float(
                payload,
                "finite_cost_coverage",
            ),
            cost_uniformity=_coerce_unit_interval_float(payload, "cost_uniformity"),
            peak_cost=_coerce_non_negative_float(payload, "peak_cost"),
            mean_cost=_coerce_non_negative_float(payload, "mean_cost"),
            weight_vector=_coerce_weight_vector(payload),
            costs_hash=_coerce_required_str(payload, "costs_hash"),
            promotion=_coerce_promotion(payload),
            blocked_consumers=_coerce_blocked_consumers(payload),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SeamCostReceipt":
        """Load a receipt from a JSON document."""

        receipt_path = Path(path)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("SeamCostReceipt JSON must contain an object.")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable receipt payload."""

        return {
            "rom_field_receipt_hash": self.rom_field_receipt_hash,
            "body_receipt_hash": self.body_receipt_hash,
            "correspondence_receipt_hash": self.correspondence_receipt_hash,
            "solve_domain": self.solve_domain,
            "vertex_count": int(self.vertex_count),
            "edge_count": int(self.edge_count),
            "finite_cost_coverage": float(self.finite_cost_coverage),
            "cost_uniformity": float(self.cost_uniformity),
            "peak_cost": float(self.peak_cost),
            "mean_cost": float(self.mean_cost),
            "weight_vector": dict(self.weight_vector),
            "costs_hash": self.costs_hash,
            "promotion": int(self.promotion),
            "blocked_consumers": list(self.blocked_consumers),
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


def load_seam_cost_receipt(path: str | Path) -> SeamCostReceipt:
    """Load a seam-cost receipt from JSON."""

    return SeamCostReceipt.from_json(path)


def with_promotion(receipt: SeamCostReceipt, promotion: object) -> SeamCostReceipt:
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


def can_consume_seam_cost_receipt(
    receipt: SeamCostReceipt, consumer: str | None = None
) -> bool:
    """Return whether a seam-cost receipt is promoted for downstream consumers."""

    if receipt.promotion != 1:
        return False
    if consumer is None:
        return not receipt.blocked_consumers
    return consumer not in receipt.blocked_consumers
