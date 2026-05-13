"""Typed receipts for promoted seam-solver artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

Promotion = Literal[-1, 0, 1]
SOLVER_MODES = ("shortest_path", "min_cut", "pda_mst")
ANCHOR_SOURCES = ("field_minima", "geometric", "manual")
DEFAULT_SOLVER_BLOCKED_CONSUMERS = (
    "panel_unwrap",
    "manufacturing",
)

__all__ = [
    "ANCHOR_SOURCES",
    "DEFAULT_SOLVER_BLOCKED_CONSUMERS",
    "Promotion",
    "SOLVER_MODES",
    "SolverPromotionReceipt",
    "can_consume_solver_promotion_receipt",
    "load_solver_promotion_receipt",
    "normalize_promotion",
    "with_promotion",
]


def normalize_promotion(value: object) -> Promotion:
    """Coerce and validate a receipt promotion state."""

    if isinstance(value, bool):
        raise ValueError("SolverPromotionReceipt promotion must be one of -1, 0, or 1.")
    if isinstance(value, int):
        promotion = value
    elif isinstance(value, float) and value.is_integer():
        promotion = int(value)
    elif isinstance(value, str) and value in {"-1", "0", "1"}:
        promotion = int(value)
    else:
        raise ValueError("SolverPromotionReceipt promotion must be one of -1, 0, or 1.")
    if promotion not in {-1, 0, 1}:
        raise ValueError("SolverPromotionReceipt promotion must be one of -1, 0, or 1.")
    return cast(Promotion, promotion)


def _missing(key: str) -> KeyError:
    return KeyError(f"SolverPromotionReceipt is missing required field '{key}'.")


def _coerce_required_str(payload: Mapping[str, Any], key: str) -> str:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_str_value(value, key)


def _coerce_str_value(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"SolverPromotionReceipt field '{key}' must be a string.")
    if not value:
        raise ValueError(f"SolverPromotionReceipt field '{key}' must be non-empty.")
    return value


def _coerce_bool(payload: Mapping[str, Any], key: str) -> bool:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    if not isinstance(value, bool):
        raise TypeError(f"SolverPromotionReceipt field '{key}' must be a boolean.")
    return value


def _coerce_non_negative_int_value(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"SolverPromotionReceipt field '{key}' must be an integer.")
    try:
        coerced = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"SolverPromotionReceipt field '{key}' must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"SolverPromotionReceipt field '{key}' must be an integer.")
    if coerced < 0:
        raise ValueError(f"SolverPromotionReceipt field '{key}' must be non-negative.")
    return coerced


def _coerce_non_negative_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_non_negative_int_value(value, key)


def _coerce_positive_int_value(value: object, key: str) -> int:
    coerced = _coerce_non_negative_int_value(value, key)
    if coerced <= 0:
        raise ValueError(f"SolverPromotionReceipt field '{key}' must be positive.")
    return coerced


def _coerce_positive_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_positive_int_value(value, key)


def _coerce_finite_float_value(value: object, key: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"SolverPromotionReceipt field '{key}' must be numeric.")
    try:
        coerced = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"SolverPromotionReceipt field '{key}' must be numeric.") from exc
    if not math.isfinite(coerced):
        raise ValueError(f"SolverPromotionReceipt field '{key}' must be finite.")
    return coerced


def _coerce_non_negative_float(payload: Mapping[str, Any], key: str) -> float:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    coerced = _coerce_finite_float_value(value, key)
    if coerced < 0.0:
        raise ValueError(f"SolverPromotionReceipt field '{key}' must be non-negative.")
    return coerced


def _coerce_blocked_consumers(payload: Mapping[str, Any]) -> list[str]:
    blocked_consumers = payload.get("blocked_consumers", [])
    return _coerce_blocked_consumer_values(blocked_consumers)


def _coerce_blocked_consumer_values(blocked_consumers: object) -> list[str]:
    if not isinstance(blocked_consumers, list):
        raise TypeError("SolverPromotionReceipt field 'blocked_consumers' must be a list.")
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
        return list(DEFAULT_SOLVER_BLOCKED_CONSUMERS)
    return blocked_consumers


@dataclass(frozen=True, slots=True)
class SolverPromotionReceipt:
    """Hash-linked receipt for seam-solver promotion decisions."""

    seam_cost_receipt_hash: str
    solver_mode: str
    anchor_count: int
    anchor_source: str
    connected_component_count: int
    anchor_fallback_used: bool
    seam_edge_count: int
    seam_vertex_count: int
    total_seam_cost: float
    panel_count: int
    panels_are_disks: bool
    seam_hash: str
    promotion: Promotion
    blocked_consumers: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "seam_cost_receipt_hash",
            _coerce_str_value(self.seam_cost_receipt_hash, "seam_cost_receipt_hash"),
        )
        solver_mode = _coerce_str_value(self.solver_mode, "solver_mode")
        if solver_mode not in SOLVER_MODES:
            raise ValueError(
                "SolverPromotionReceipt field 'solver_mode' must be one of "
                f"{', '.join(SOLVER_MODES)}."
            )
        object.__setattr__(self, "solver_mode", solver_mode)
        object.__setattr__(
            self,
            "anchor_count",
            _coerce_non_negative_int_value(self.anchor_count, "anchor_count"),
        )
        anchor_source = _coerce_str_value(self.anchor_source, "anchor_source")
        if anchor_source not in ANCHOR_SOURCES:
            raise ValueError(
                "SolverPromotionReceipt field 'anchor_source' must be one of "
                f"{', '.join(ANCHOR_SOURCES)}."
            )
        object.__setattr__(self, "anchor_source", anchor_source)
        object.__setattr__(
            self,
            "connected_component_count",
            _coerce_non_negative_int_value(
                self.connected_component_count,
                "connected_component_count",
            ),
        )
        if not isinstance(self.anchor_fallback_used, bool):
            raise TypeError(
                "SolverPromotionReceipt field 'anchor_fallback_used' must be a boolean."
            )
        object.__setattr__(
            self,
            "seam_edge_count",
            _coerce_non_negative_int_value(self.seam_edge_count, "seam_edge_count"),
        )
        object.__setattr__(
            self,
            "seam_vertex_count",
            _coerce_non_negative_int_value(self.seam_vertex_count, "seam_vertex_count"),
        )
        object.__setattr__(
            self,
            "total_seam_cost",
            _coerce_finite_float_value(self.total_seam_cost, "total_seam_cost"),
        )
        if self.total_seam_cost < 0.0:
            raise ValueError("SolverPromotionReceipt field 'total_seam_cost' must be non-negative.")
        object.__setattr__(
            self,
            "panel_count",
            _coerce_non_negative_int_value(self.panel_count, "panel_count"),
        )
        if not isinstance(self.panels_are_disks, bool):
            raise TypeError("SolverPromotionReceipt field 'panels_are_disks' must be a boolean.")
        object.__setattr__(self, "seam_hash", _coerce_str_value(self.seam_hash, "seam_hash"))
        object.__setattr__(self, "promotion", normalize_promotion(self.promotion))
        blocked_consumers = _coerce_blocked_consumer_values(self.blocked_consumers)
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked_consumers_for_promotion(self.promotion, blocked_consumers),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SolverPromotionReceipt":
        """Build a validated receipt from a JSON-like mapping."""

        return cls(
            seam_cost_receipt_hash=_coerce_required_str(payload, "seam_cost_receipt_hash"),
            solver_mode=_coerce_required_str(payload, "solver_mode"),
            anchor_count=_coerce_non_negative_int(payload, "anchor_count"),
            anchor_source=_coerce_required_str(payload, "anchor_source"),
            connected_component_count=_coerce_non_negative_int(
                payload,
                "connected_component_count",
            ),
            anchor_fallback_used=_coerce_bool(payload, "anchor_fallback_used"),
            seam_edge_count=_coerce_non_negative_int(payload, "seam_edge_count"),
            seam_vertex_count=_coerce_non_negative_int(payload, "seam_vertex_count"),
            total_seam_cost=_coerce_non_negative_float(payload, "total_seam_cost"),
            panel_count=_coerce_non_negative_int(payload, "panel_count"),
            panels_are_disks=_coerce_bool(payload, "panels_are_disks"),
            seam_hash=_coerce_required_str(payload, "seam_hash"),
            promotion=_coerce_promotion(payload),
            blocked_consumers=_coerce_blocked_consumers(payload),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SolverPromotionReceipt":
        """Load a receipt from a JSON document."""

        receipt_path = Path(path)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("SolverPromotionReceipt JSON must contain an object.")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable receipt payload."""

        return {
            "seam_cost_receipt_hash": self.seam_cost_receipt_hash,
            "solver_mode": self.solver_mode,
            "anchor_count": int(self.anchor_count),
            "anchor_source": self.anchor_source,
            "connected_component_count": int(self.connected_component_count),
            "anchor_fallback_used": bool(self.anchor_fallback_used),
            "seam_edge_count": int(self.seam_edge_count),
            "seam_vertex_count": int(self.seam_vertex_count),
            "total_seam_cost": float(self.total_seam_cost),
            "panel_count": int(self.panel_count),
            "panels_are_disks": bool(self.panels_are_disks),
            "seam_hash": self.seam_hash,
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


def load_solver_promotion_receipt(path: str | Path) -> SolverPromotionReceipt:
    """Load a solver promotion receipt from JSON."""

    return SolverPromotionReceipt.from_json(path)


def with_promotion(
    receipt: SolverPromotionReceipt, promotion: object
) -> SolverPromotionReceipt:
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


def can_consume_solver_promotion_receipt(
    receipt: SolverPromotionReceipt, consumer: str | None = None
) -> bool:
    """Return whether a solver receipt is promoted for downstream consumers."""

    if receipt.promotion != 1:
        return False
    if consumer is None:
        return not receipt.blocked_consumers
    return consumer not in receipt.blocked_consumers
