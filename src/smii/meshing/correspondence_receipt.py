"""Typed receipts for mesh correspondence promotion decisions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

Promotion = Literal[-1, 0, 1]

DEFAULT_MIN_UNIQUE_TARGET_RATIO = 0.05
DEFAULT_CORRESPONDENCE_BLOCKED_CONSUMERS = (
    "seam_transfer",
    "seam_cost_field",
    "solver_promotion",
    "panel_unwrap",
)

__all__ = [
    "CorrespondenceReceipt",
    "DEFAULT_CORRESPONDENCE_BLOCKED_CONSUMERS",
    "Promotion",
    "TransformReceipt",
    "can_consume_correspondence_receipt",
    "is_diagnostic_nn_collapse",
    "load_correspondence_receipt",
    "normalize_promotion",
    "with_promotion",
]


def normalize_promotion(value: object) -> Promotion:
    """Coerce and validate a receipt promotion state."""

    if isinstance(value, bool):
        raise ValueError("CorrespondenceReceipt promotion must be one of -1, 0, or 1.")
    if isinstance(value, int):
        promotion = value
    elif isinstance(value, float) and value.is_integer():
        promotion = int(value)
    elif isinstance(value, str) and value in {"-1", "0", "1"}:
        promotion = int(value)
    else:
        raise ValueError("CorrespondenceReceipt promotion must be one of -1, 0, or 1.")
    if promotion not in {-1, 0, 1}:
        raise ValueError("CorrespondenceReceipt promotion must be one of -1, 0, or 1.")
    return cast(Promotion, promotion)


def _missing(key: str) -> KeyError:
    return KeyError(f"CorrespondenceReceipt is missing required field '{key}'.")


def _coerce_required_str(payload: Mapping[str, Any], key: str) -> str:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_str_value(value, key)


def _coerce_str_value(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"CorrespondenceReceipt field '{key}' must be a string.")
    return value


def _coerce_finite_float(payload: Mapping[str, Any], key: str) -> float:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_finite_float_value(value, key)


def _coerce_finite_float_value(value: object, key: str) -> float:
    try:
        coerced = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        message = f"CorrespondenceReceipt field '{key}' must be numeric."
        raise TypeError(message) from exc
    if not math.isfinite(coerced):
        raise ValueError(f"CorrespondenceReceipt field '{key}' must be finite.")
    return coerced


def _coerce_non_negative_float(payload: Mapping[str, Any], key: str) -> float:
    value = _coerce_finite_float(payload, key)
    if value < 0.0:
        raise ValueError(f"CorrespondenceReceipt field '{key}' must be non-negative.")
    return value


def _coerce_non_negative_float_value(value: object, key: str) -> float:
    coerced = _coerce_finite_float_value(value, key)
    if coerced < 0.0:
        raise ValueError(f"CorrespondenceReceipt field '{key}' must be non-negative.")
    return coerced


def _coerce_ratio(payload: Mapping[str, Any], key: str) -> float:
    value = _coerce_finite_float(payload, key)
    if not 0.0 <= value <= 1.0:
        message = f"CorrespondenceReceipt field '{key}' must be between 0 and 1."
        raise ValueError(message)
    return value


def _coerce_ratio_value(value: object, key: str) -> float:
    coerced = _coerce_finite_float_value(value, key)
    if not 0.0 <= coerced <= 1.0:
        message = f"CorrespondenceReceipt field '{key}' must be between 0 and 1."
        raise ValueError(message)
    return coerced


def _coerce_non_negative_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_non_negative_int_value(value, key)


def _coerce_non_negative_int_value(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"CorrespondenceReceipt field '{key}' must be an integer.")
    try:
        coerced = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        message = f"CorrespondenceReceipt field '{key}' must be an integer."
        raise TypeError(message) from exc
    if coerced < 0:
        raise ValueError(f"CorrespondenceReceipt field '{key}' must be non-negative.")
    return coerced


def _coerce_notes(payload: Mapping[str, Any]) -> list[str]:
    try:
        notes = payload["notes"]
    except KeyError as exc:
        raise _missing("notes") from exc
    return _coerce_note_values(notes)


def _coerce_note_values(notes: object) -> list[str]:
    if isinstance(notes, str):
        return [notes]
    if not isinstance(notes, list):
        raise TypeError("CorrespondenceReceipt field 'notes' must be a list.")
    return [str(note) for note in notes]


def _coerce_blocked_consumers(payload: Mapping[str, Any]) -> list[str]:
    blocked_consumers = payload.get("blocked_consumers", [])
    return _coerce_blocked_consumer_values(blocked_consumers)


def _coerce_blocked_consumer_values(blocked_consumers: object) -> list[str]:
    if not isinstance(blocked_consumers, list):
        raise TypeError("CorrespondenceReceipt field 'blocked_consumers' must be a list.")
    return [str(consumer) for consumer in blocked_consumers]


def _blocked_consumers_for_promotion(
    promotion: Promotion, blocked_consumers: list[str]
) -> list[str]:
    if promotion != 1 and not blocked_consumers:
        return list(DEFAULT_CORRESPONDENCE_BLOCKED_CONSUMERS)
    return blocked_consumers


def _coerce_promotion(payload: Mapping[str, Any]) -> Promotion:
    try:
        value = payload["promotion"]
    except KeyError as exc:
        raise _missing("promotion") from exc
    return normalize_promotion(value)


@dataclass(frozen=True, slots=True)
class CorrespondenceReceipt:
    """Hash-linked receipt for transfer/correspondence promotion gates."""

    source_mesh_hash: str
    target_mesh_hash: str
    transform_type: str
    mean_distance: float
    max_distance: float
    collision_ratio: float
    retention_ratio: float
    unique_targets_used: int
    total_target_vertices: int
    edge_retention_ratio: float
    promotion: Promotion
    notes: list[str]
    blocked_consumers: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_mesh_hash",
            _coerce_str_value(self.source_mesh_hash, "source_mesh_hash"),
        )
        object.__setattr__(
            self,
            "target_mesh_hash",
            _coerce_str_value(self.target_mesh_hash, "target_mesh_hash"),
        )
        object.__setattr__(
            self,
            "transform_type",
            _coerce_str_value(self.transform_type, "transform_type"),
        )
        object.__setattr__(
            self,
            "mean_distance",
            _coerce_non_negative_float_value(self.mean_distance, "mean_distance"),
        )
        object.__setattr__(
            self,
            "max_distance",
            _coerce_non_negative_float_value(self.max_distance, "max_distance"),
        )
        object.__setattr__(
            self,
            "collision_ratio",
            _coerce_ratio_value(self.collision_ratio, "collision_ratio"),
        )
        object.__setattr__(
            self,
            "retention_ratio",
            _coerce_ratio_value(self.retention_ratio, "retention_ratio"),
        )
        object.__setattr__(
            self,
            "unique_targets_used",
            _coerce_non_negative_int_value(
                self.unique_targets_used,
                "unique_targets_used",
            ),
        )
        object.__setattr__(
            self,
            "total_target_vertices",
            _coerce_non_negative_int_value(
                self.total_target_vertices, "total_target_vertices"
            ),
        )
        if self.unique_targets_used > self.total_target_vertices:
            raise ValueError(
                "CorrespondenceReceipt field 'unique_targets_used' must not exceed "
                "'total_target_vertices'."
            )
        object.__setattr__(
            self,
            "edge_retention_ratio",
            _coerce_ratio_value(self.edge_retention_ratio, "edge_retention_ratio"),
        )
        object.__setattr__(self, "promotion", normalize_promotion(self.promotion))
        object.__setattr__(self, "notes", _coerce_note_values(self.notes))
        blocked_consumers = _coerce_blocked_consumer_values(self.blocked_consumers)
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked_consumers_for_promotion(self.promotion, blocked_consumers),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CorrespondenceReceipt":
        """Build a validated receipt from a JSON-like mapping."""

        return cls(
            source_mesh_hash=_coerce_required_str(payload, "source_mesh_hash"),
            target_mesh_hash=_coerce_required_str(payload, "target_mesh_hash"),
            transform_type=_coerce_required_str(payload, "transform_type"),
            mean_distance=_coerce_non_negative_float(payload, "mean_distance"),
            max_distance=_coerce_non_negative_float(payload, "max_distance"),
            collision_ratio=_coerce_ratio(payload, "collision_ratio"),
            retention_ratio=_coerce_ratio(payload, "retention_ratio"),
            unique_targets_used=_coerce_non_negative_int(
                payload,
                "unique_targets_used",
            ),
            total_target_vertices=_coerce_non_negative_int(
                payload,
                "total_target_vertices",
            ),
            edge_retention_ratio=_coerce_ratio(payload, "edge_retention_ratio"),
            promotion=_coerce_promotion(payload),
            notes=_coerce_notes(payload),
            blocked_consumers=_coerce_blocked_consumers(payload),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CorrespondenceReceipt":
        """Load a receipt from a JSON document."""

        receipt_path = Path(path)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("CorrespondenceReceipt JSON must contain an object.")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable receipt payload."""

        return {
            "source_mesh_hash": self.source_mesh_hash,
            "target_mesh_hash": self.target_mesh_hash,
            "transform_type": self.transform_type,
            "mean_distance": float(self.mean_distance),
            "max_distance": float(self.max_distance),
            "collision_ratio": float(self.collision_ratio),
            "retention_ratio": float(self.retention_ratio),
            "unique_targets_used": int(self.unique_targets_used),
            "total_target_vertices": int(self.total_target_vertices),
            "edge_retention_ratio": float(self.edge_retention_ratio),
            "promotion": int(self.promotion),
            "notes": list(self.notes),
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


TransformReceipt = CorrespondenceReceipt


def load_correspondence_receipt(path: str | Path) -> CorrespondenceReceipt:
    """Load a correspondence receipt from JSON."""

    return CorrespondenceReceipt.from_json(path)


def with_promotion(
    receipt: CorrespondenceReceipt,
    promotion: object,
) -> CorrespondenceReceipt:
    """Return a copy of a receipt with a validated promotion value."""

    next_promotion = normalize_promotion(promotion)
    return replace(
        receipt,
        promotion=next_promotion,
        blocked_consumers=_blocked_consumers_for_promotion(
            next_promotion, receipt.blocked_consumers
        ),
    )


def is_diagnostic_nn_collapse(
    receipt: CorrespondenceReceipt,
    *,
    min_unique_target_ratio: float = DEFAULT_MIN_UNIQUE_TARGET_RATIO,
) -> bool:
    """Return whether a nearest-neighbor correspondence collapsed diagnostically.

    This only inspects already-recorded correspondence counts. It does not infer
    geometry quality beyond the receipt's explicit coverage signal.
    """

    threshold = _coerce_ratio_value(min_unique_target_ratio, "min_unique_target_ratio")
    if receipt.total_target_vertices <= 0:
        return True
    return (receipt.unique_targets_used / receipt.total_target_vertices) < threshold


def can_consume_correspondence_receipt(
    receipt: CorrespondenceReceipt,
    consumer: str | None = None,
    *,
    min_unique_target_ratio: float = DEFAULT_MIN_UNIQUE_TARGET_RATIO,
) -> bool:
    """Return whether a receipt is promoted and not a diagnostic NN collapse."""

    if receipt.promotion != 1:
        return False
    if consumer is None and receipt.blocked_consumers:
        return False
    if consumer is not None and consumer in receipt.blocked_consumers:
        return False
    return not is_diagnostic_nn_collapse(
        receipt,
        min_unique_target_ratio=min_unique_target_ratio,
    )
