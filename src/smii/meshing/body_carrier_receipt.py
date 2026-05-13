"""Typed receipts for body-carrier mesh promotion decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

Promotion = Literal[-1, 0, 1]
DEFAULT_BODY_BLOCKED_CONSUMERS = (
    "generate_undersuit",
    "seam_cost_field",
    "panel_unwrap",
)

__all__ = [
    "BodyCarrierReceipt",
    "DEFAULT_BODY_BLOCKED_CONSUMERS",
    "Promotion",
    "can_consume_receipt",
    "load_body_carrier_receipt",
    "normalize_promotion",
    "with_blocked_consumers",
    "with_promotion",
]


def normalize_promotion(value: object) -> Promotion:
    """Coerce and validate a receipt promotion state."""

    if isinstance(value, bool):
        raise ValueError("BodyCarrierReceipt promotion must be one of -1, 0, or 1.")
    if isinstance(value, int):
        promotion = value
    elif isinstance(value, float) and value.is_integer():
        promotion = int(value)
    elif isinstance(value, str) and value in {"-1", "0", "1"}:
        promotion = int(value)
    else:
        raise ValueError("BodyCarrierReceipt promotion must be one of -1, 0, or 1.")
    if promotion not in {-1, 0, 1}:
        raise ValueError("BodyCarrierReceipt promotion must be one of -1, 0, or 1.")
    return cast(Promotion, promotion)


def _coerce_required_str(payload: Mapping[str, Any], key: str) -> str:
    try:
        value = payload[key]
    except KeyError as exc:
        raise KeyError(f"BodyCarrierReceipt is missing required field '{key}'.") from exc
    if not isinstance(value, str):
        raise TypeError(f"BodyCarrierReceipt field '{key}' must be a string.")
    return value


def _coerce_str_value(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"BodyCarrierReceipt field '{key}' must be a string.")
    return value


def _coerce_non_negative_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        value = int(payload[key])
    except KeyError as exc:
        raise KeyError(f"BodyCarrierReceipt is missing required field '{key}'.") from exc
    except (TypeError, ValueError) as exc:
        raise TypeError(f"BodyCarrierReceipt field '{key}' must be an integer.") from exc
    if value < 0:
        raise ValueError(f"BodyCarrierReceipt field '{key}' must be non-negative.")
    return value


def _coerce_non_negative_int_value(value: object, key: str) -> int:
    try:
        coerced = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"BodyCarrierReceipt field '{key}' must be an integer.") from exc
    if coerced < 0:
        raise ValueError(f"BodyCarrierReceipt field '{key}' must be non-negative.")
    return coerced


def _coerce_float(payload: Mapping[str, Any], key: str) -> float:
    try:
        return float(payload[key])
    except KeyError as exc:
        raise KeyError(f"BodyCarrierReceipt is missing required field '{key}'.") from exc
    except (TypeError, ValueError) as exc:
        raise TypeError(f"BodyCarrierReceipt field '{key}' must be numeric.") from exc


def _coerce_float_value(value: object, key: str) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"BodyCarrierReceipt field '{key}' must be numeric.") from exc


def _coerce_landmark_residuals(payload: Mapping[str, Any]) -> dict[str, float]:
    try:
        residuals = payload["landmark_residuals"]
    except KeyError as exc:
        raise KeyError(
            "BodyCarrierReceipt is missing required field 'landmark_residuals'."
        ) from exc
    if not isinstance(residuals, Mapping):
        raise TypeError("BodyCarrierReceipt field 'landmark_residuals' must be an object.")
    return _coerce_landmark_residual_values(residuals)


def _coerce_landmark_residual_values(residuals: object) -> dict[str, float]:
    if not isinstance(residuals, Mapping):
        raise TypeError("BodyCarrierReceipt field 'landmark_residuals' must be an object.")
    return {str(name): float(value) for name, value in residuals.items()}


def _coerce_blocked_consumers(payload: Mapping[str, Any]) -> list[str]:
    try:
        blocked_consumers = payload["blocked_consumers"]
    except KeyError as exc:
        raise KeyError(
            "BodyCarrierReceipt is missing required field 'blocked_consumers'."
        ) from exc
    if not isinstance(blocked_consumers, list):
        raise TypeError("BodyCarrierReceipt field 'blocked_consumers' must be a list.")
    return _coerce_blocked_consumer_values(blocked_consumers)


def _coerce_promotion(payload: Mapping[str, Any]) -> Promotion:
    try:
        value = payload["promotion"]
    except KeyError as exc:
        raise KeyError("BodyCarrierReceipt is missing required field 'promotion'.") from exc
    return normalize_promotion(value)


def _coerce_blocked_consumer_values(blocked_consumers: object) -> list[str]:
    if not isinstance(blocked_consumers, list):
        raise TypeError("BodyCarrierReceipt field 'blocked_consumers' must be a list.")
    return [str(consumer) for consumer in blocked_consumers]


def _blocked_consumers_for_promotion(
    promotion: Promotion, blocked_consumers: list[str]
) -> list[str]:
    if promotion != 1 and not blocked_consumers:
        return list(DEFAULT_BODY_BLOCKED_CONSUMERS)
    return blocked_consumers


@dataclass(frozen=True, slots=True)
class BodyCarrierReceipt:
    """Hash-linked mesh receipt used to gate downstream body consumers."""

    source_hash: str
    raw_reprojection_hash: str
    refined_pre_repair_hash: str
    repaired_export_hash: str
    vertex_count: int
    face_count: int
    topology_label: str
    landmark_residuals: dict[str, float]
    skull_rigidity_residual: float
    body_fit_confidence: float
    promotion: Promotion
    blocked_consumers: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_hash",
            _coerce_str_value(self.source_hash, "source_hash"),
        )
        object.__setattr__(
            self,
            "raw_reprojection_hash",
            _coerce_str_value(self.raw_reprojection_hash, "raw_reprojection_hash"),
        )
        object.__setattr__(
            self,
            "refined_pre_repair_hash",
            _coerce_str_value(self.refined_pre_repair_hash, "refined_pre_repair_hash"),
        )
        object.__setattr__(
            self,
            "repaired_export_hash",
            _coerce_str_value(self.repaired_export_hash, "repaired_export_hash"),
        )
        object.__setattr__(
            self,
            "vertex_count",
            _coerce_non_negative_int_value(self.vertex_count, "vertex_count"),
        )
        object.__setattr__(
            self,
            "face_count",
            _coerce_non_negative_int_value(self.face_count, "face_count"),
        )
        object.__setattr__(
            self,
            "topology_label",
            _coerce_str_value(self.topology_label, "topology_label"),
        )
        object.__setattr__(
            self,
            "landmark_residuals",
            _coerce_landmark_residual_values(self.landmark_residuals),
        )
        object.__setattr__(
            self,
            "skull_rigidity_residual",
            _coerce_float_value(self.skull_rigidity_residual, "skull_rigidity_residual"),
        )
        object.__setattr__(
            self,
            "body_fit_confidence",
            _coerce_float_value(self.body_fit_confidence, "body_fit_confidence"),
        )
        object.__setattr__(self, "promotion", normalize_promotion(self.promotion))
        blocked_consumers = _coerce_blocked_consumer_values(self.blocked_consumers)
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked_consumers_for_promotion(self.promotion, blocked_consumers),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BodyCarrierReceipt":
        """Build a validated receipt from a JSON-like mapping."""

        return cls(
            source_hash=_coerce_required_str(payload, "source_hash"),
            raw_reprojection_hash=_coerce_required_str(payload, "raw_reprojection_hash"),
            refined_pre_repair_hash=_coerce_required_str(payload, "refined_pre_repair_hash"),
            repaired_export_hash=_coerce_required_str(payload, "repaired_export_hash"),
            vertex_count=_coerce_non_negative_int(payload, "vertex_count"),
            face_count=_coerce_non_negative_int(payload, "face_count"),
            topology_label=_coerce_required_str(payload, "topology_label"),
            landmark_residuals=_coerce_landmark_residuals(payload),
            skull_rigidity_residual=_coerce_float(payload, "skull_rigidity_residual"),
            body_fit_confidence=_coerce_float(payload, "body_fit_confidence"),
            promotion=_coerce_promotion(payload),
            blocked_consumers=_coerce_blocked_consumers(payload),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "BodyCarrierReceipt":
        """Load a receipt from a JSON document."""

        receipt_path = Path(path)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("BodyCarrierReceipt JSON must contain an object.")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable receipt payload."""

        return {
            "source_hash": self.source_hash,
            "raw_reprojection_hash": self.raw_reprojection_hash,
            "refined_pre_repair_hash": self.refined_pre_repair_hash,
            "repaired_export_hash": self.repaired_export_hash,
            "vertex_count": int(self.vertex_count),
            "face_count": int(self.face_count),
            "topology_label": self.topology_label,
            "landmark_residuals": {
                name: float(value) for name, value in self.landmark_residuals.items()
            },
            "skull_rigidity_residual": float(self.skull_rigidity_residual),
            "body_fit_confidence": float(self.body_fit_confidence),
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


def load_body_carrier_receipt(path: str | Path) -> BodyCarrierReceipt:
    """Load a body-carrier receipt from JSON."""

    return BodyCarrierReceipt.from_json(path)


def with_promotion(receipt: BodyCarrierReceipt, promotion: object) -> BodyCarrierReceipt:
    """Return a copy of a receipt with a validated promotion value."""

    next_promotion = normalize_promotion(promotion)
    return replace(
        receipt,
        promotion=next_promotion,
        blocked_consumers=_blocked_consumers_for_promotion(
            next_promotion, receipt.blocked_consumers
        ),
    )


def with_blocked_consumers(
    receipt: BodyCarrierReceipt,
    blocked_consumers: list[str],
    *,
    promotion: object | None = None,
) -> BodyCarrierReceipt:
    """Return a copy with an updated consumer block list and optional promotion."""

    next_promotion = receipt.promotion if promotion is None else normalize_promotion(promotion)
    next_blocked_consumers = _blocked_consumers_for_promotion(
        next_promotion, [str(consumer) for consumer in blocked_consumers]
    )
    return replace(
        receipt,
        blocked_consumers=next_blocked_consumers,
        promotion=next_promotion,
    )


def can_consume_receipt(receipt: BodyCarrierReceipt, consumer: str | None = None) -> bool:
    """Return whether a receipt is promoted and not blocked for the consumer."""

    if receipt.promotion != 1:
        return False
    if consumer is None:
        return not receipt.blocked_consumers
    return consumer not in receipt.blocked_consumers
