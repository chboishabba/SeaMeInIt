"""Typed receipts for promoted panel-unwrap artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

Promotion = Literal[-1, 0, 1]
GRAIN_DIRECTIONS = ("warp", "weft", "bias")
DEFAULT_PANEL_UNWRAP_BLOCKED_CONSUMERS = ("manufacturing",)

__all__ = [
    "DEFAULT_PANEL_UNWRAP_BLOCKED_CONSUMERS",
    "GRAIN_DIRECTIONS",
    "PanelUnwrapReceipt",
    "Promotion",
    "can_consume_panel_unwrap_receipt",
    "load_panel_unwrap_receipt",
    "normalize_promotion",
    "with_promotion",
]


def normalize_promotion(value: object) -> Promotion:
    """Coerce and validate a receipt promotion state."""

    if isinstance(value, bool):
        raise ValueError("PanelUnwrapReceipt promotion must be one of -1, 0, or 1.")
    if isinstance(value, int):
        promotion = value
    elif isinstance(value, float) and value.is_integer():
        promotion = int(value)
    elif isinstance(value, str) and value in {"-1", "0", "1"}:
        promotion = int(value)
    else:
        raise ValueError("PanelUnwrapReceipt promotion must be one of -1, 0, or 1.")
    if promotion not in {-1, 0, 1}:
        raise ValueError("PanelUnwrapReceipt promotion must be one of -1, 0, or 1.")
    return cast(Promotion, promotion)


def _missing(key: str) -> KeyError:
    return KeyError(f"PanelUnwrapReceipt is missing required field '{key}'.")


def _coerce_required_str(payload: Mapping[str, Any], key: str) -> str:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_str_value(value, key)


def _coerce_str_value(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"PanelUnwrapReceipt field '{key}' must be a string.")
    if not value:
        raise ValueError(f"PanelUnwrapReceipt field '{key}' must be non-empty.")
    return value


def _coerce_bool(payload: Mapping[str, Any], key: str) -> bool:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    if not isinstance(value, bool):
        raise TypeError(f"PanelUnwrapReceipt field '{key}' must be a boolean.")
    return value


def _coerce_non_negative_int_value(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"PanelUnwrapReceipt field '{key}' must be an integer.")
    try:
        coerced = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"PanelUnwrapReceipt field '{key}' must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"PanelUnwrapReceipt field '{key}' must be an integer.")
    if coerced < 0:
        raise ValueError(f"PanelUnwrapReceipt field '{key}' must be non-negative.")
    return coerced


def _coerce_non_negative_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_non_negative_int_value(value, key)


def _coerce_finite_float_value(value: object, key: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"PanelUnwrapReceipt field '{key}' must be numeric.")
    try:
        coerced = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"PanelUnwrapReceipt field '{key}' must be numeric.") from exc
    if not math.isfinite(coerced):
        raise ValueError(f"PanelUnwrapReceipt field '{key}' must be finite.")
    return coerced


def _coerce_non_negative_float_value(value: object, key: str) -> float:
    coerced = _coerce_finite_float_value(value, key)
    if coerced < 0.0:
        raise ValueError(f"PanelUnwrapReceipt field '{key}' must be non-negative.")
    return coerced


def _coerce_non_negative_float(payload: Mapping[str, Any], key: str) -> float:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_non_negative_float_value(value, key)


def _coerce_float_list(payload: Mapping[str, Any], key: str) -> list[float]:
    try:
        values = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    if not isinstance(values, list):
        raise TypeError(f"PanelUnwrapReceipt field '{key}' must be a list.")
    return [
        _coerce_non_negative_float_value(value, f"{key}[{idx}]")
        for idx, value in enumerate(values)
    ]


def _coerce_grain_directions(payload: Mapping[str, Any]) -> list[str]:
    try:
        values = payload["grain_directions"]
    except KeyError as exc:
        raise _missing("grain_directions") from exc
    return _coerce_grain_direction_values(values)


def _coerce_grain_direction_values(values: object) -> list[str]:
    if not isinstance(values, list):
        raise TypeError("PanelUnwrapReceipt field 'grain_directions' must be a list.")
    directions = [str(value) for value in values]
    invalid = [direction for direction in directions if direction not in GRAIN_DIRECTIONS]
    if invalid:
        raise ValueError(
            "PanelUnwrapReceipt field 'grain_directions' entries must be one of "
            f"{', '.join(GRAIN_DIRECTIONS)}."
        )
    return directions


def _coerce_blocked_consumers(payload: Mapping[str, Any]) -> list[str]:
    blocked_consumers = payload.get("blocked_consumers", [])
    return _coerce_blocked_consumer_values(blocked_consumers)


def _coerce_blocked_consumer_values(blocked_consumers: object) -> list[str]:
    if not isinstance(blocked_consumers, list):
        raise TypeError("PanelUnwrapReceipt field 'blocked_consumers' must be a list.")
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
        return list(DEFAULT_PANEL_UNWRAP_BLOCKED_CONSUMERS)
    return blocked_consumers


@dataclass(frozen=True, slots=True)
class PanelUnwrapReceipt:
    """Hash-linked receipt for panel flattening and UV promotion decisions."""

    solver_receipt_hash: str
    panel_count: int
    panels_all_disks: bool
    per_panel_distortion: list[float]
    worst_panel_distortion: float
    mean_panel_distortion: float
    distortion_threshold: float
    subdivision_iterations: int
    grain_directions: list[str]
    uv_hash: str
    seam_topology_hash: str
    promotion: Promotion
    blocked_consumers: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "solver_receipt_hash",
            _coerce_str_value(self.solver_receipt_hash, "solver_receipt_hash"),
        )
        panel_count = _coerce_non_negative_int_value(self.panel_count, "panel_count")
        object.__setattr__(self, "panel_count", panel_count)
        if not isinstance(self.panels_all_disks, bool):
            raise TypeError("PanelUnwrapReceipt field 'panels_all_disks' must be a boolean.")
        distortions = [
            _coerce_non_negative_float_value(value, f"per_panel_distortion[{idx}]")
            for idx, value in enumerate(self.per_panel_distortion)
        ]
        if len(distortions) != panel_count:
            raise ValueError(
                "PanelUnwrapReceipt field 'per_panel_distortion' length must match "
                "panel_count."
            )
        object.__setattr__(self, "per_panel_distortion", distortions)
        object.__setattr__(
            self,
            "worst_panel_distortion",
            _coerce_non_negative_float_value(
                self.worst_panel_distortion,
                "worst_panel_distortion",
            ),
        )
        object.__setattr__(
            self,
            "mean_panel_distortion",
            _coerce_non_negative_float_value(
                self.mean_panel_distortion,
                "mean_panel_distortion",
            ),
        )
        object.__setattr__(
            self,
            "distortion_threshold",
            _coerce_non_negative_float_value(
                self.distortion_threshold,
                "distortion_threshold",
            ),
        )
        object.__setattr__(
            self,
            "subdivision_iterations",
            _coerce_non_negative_int_value(
                self.subdivision_iterations,
                "subdivision_iterations",
            ),
        )
        grain_directions = _coerce_grain_direction_values(self.grain_directions)
        if len(grain_directions) != panel_count:
            raise ValueError(
                "PanelUnwrapReceipt field 'grain_directions' length must match panel_count."
            )
        object.__setattr__(self, "grain_directions", grain_directions)
        object.__setattr__(self, "uv_hash", _coerce_str_value(self.uv_hash, "uv_hash"))
        object.__setattr__(
            self,
            "seam_topology_hash",
            _coerce_str_value(self.seam_topology_hash, "seam_topology_hash"),
        )
        object.__setattr__(self, "promotion", normalize_promotion(self.promotion))
        blocked_consumers = _coerce_blocked_consumer_values(self.blocked_consumers)
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked_consumers_for_promotion(self.promotion, blocked_consumers),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PanelUnwrapReceipt":
        """Build a validated receipt from a JSON-like mapping."""

        return cls(
            solver_receipt_hash=_coerce_required_str(payload, "solver_receipt_hash"),
            panel_count=_coerce_non_negative_int(payload, "panel_count"),
            panels_all_disks=_coerce_bool(payload, "panels_all_disks"),
            per_panel_distortion=_coerce_float_list(payload, "per_panel_distortion"),
            worst_panel_distortion=_coerce_non_negative_float(
                payload,
                "worst_panel_distortion",
            ),
            mean_panel_distortion=_coerce_non_negative_float(
                payload,
                "mean_panel_distortion",
            ),
            distortion_threshold=_coerce_non_negative_float(
                payload,
                "distortion_threshold",
            ),
            subdivision_iterations=_coerce_non_negative_int(
                payload,
                "subdivision_iterations",
            ),
            grain_directions=_coerce_grain_directions(payload),
            uv_hash=_coerce_required_str(payload, "uv_hash"),
            seam_topology_hash=_coerce_required_str(payload, "seam_topology_hash"),
            promotion=_coerce_promotion(payload),
            blocked_consumers=_coerce_blocked_consumers(payload),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PanelUnwrapReceipt":
        """Load a receipt from a JSON document."""

        receipt_path = Path(path)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("PanelUnwrapReceipt JSON must contain an object.")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable receipt payload."""

        return {
            "solver_receipt_hash": self.solver_receipt_hash,
            "panel_count": int(self.panel_count),
            "panels_all_disks": bool(self.panels_all_disks),
            "per_panel_distortion": [float(value) for value in self.per_panel_distortion],
            "worst_panel_distortion": float(self.worst_panel_distortion),
            "mean_panel_distortion": float(self.mean_panel_distortion),
            "distortion_threshold": float(self.distortion_threshold),
            "subdivision_iterations": int(self.subdivision_iterations),
            "grain_directions": list(self.grain_directions),
            "uv_hash": self.uv_hash,
            "seam_topology_hash": self.seam_topology_hash,
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


def load_panel_unwrap_receipt(path: str | Path) -> PanelUnwrapReceipt:
    """Load a panel unwrap receipt from JSON."""

    return PanelUnwrapReceipt.from_json(path)


def with_promotion(receipt: PanelUnwrapReceipt, promotion: object) -> PanelUnwrapReceipt:
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


def can_consume_panel_unwrap_receipt(
    receipt: PanelUnwrapReceipt, consumer: str | None = None
) -> bool:
    """Return whether a panel unwrap receipt is promoted for downstream consumers."""

    if receipt.promotion != 1:
        return False
    if consumer is None:
        return not receipt.blocked_consumers
    return consumer not in receipt.blocked_consumers
