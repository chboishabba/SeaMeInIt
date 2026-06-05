"""Typed receipts for metric-correction operators between cuts and unwraps."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

Promotion = Literal[-1, 0, 1]
CORRECTION_TYPES = (
    "dart",
    "relief_cut",
    "gusset",
    "ease",
    "stretch_zone",
    "abstain",
)
CORRECTION_STATES = (
    "correctionOk",
    "correctionDegraded",
    "correctionRejected",
    "correctionAbstained",
)
DEFAULT_METRIC_CORRECTION_BLOCKED_CONSUMERS = ("panel_unwrap", "manufacturing")

__all__ = [
    "CORRECTION_STATES",
    "CORRECTION_TYPES",
    "DEFAULT_METRIC_CORRECTION_BLOCKED_CONSUMERS",
    "MetricCorrectionEntry",
    "MetricCorrectionReceipt",
    "Promotion",
    "can_consume_metric_correction_receipt",
    "load_metric_correction_receipt",
    "normalize_promotion",
    "with_promotion",
]


def normalize_promotion(value: object) -> Promotion:
    if isinstance(value, bool):
        raise ValueError("MetricCorrectionReceipt promotion must be one of -1, 0, or 1.")
    if isinstance(value, int):
        promotion = value
    elif isinstance(value, float) and value.is_integer():
        promotion = int(value)
    elif isinstance(value, str) and value in {"-1", "0", "1"}:
        promotion = int(value)
    else:
        raise ValueError("MetricCorrectionReceipt promotion must be one of -1, 0, or 1.")
    if promotion not in {-1, 0, 1}:
        raise ValueError("MetricCorrectionReceipt promotion must be one of -1, 0, or 1.")
    return cast(Promotion, promotion)


def _missing(key: str) -> KeyError:
    return KeyError(f"MetricCorrectionReceipt is missing required field '{key}'.")


def _str_value(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"MetricCorrectionReceipt field '{key}' must be a string.")
    if not value:
        raise ValueError(f"MetricCorrectionReceipt field '{key}' must be non-empty.")
    return value


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    try:
        return _str_value(payload[key], key)
    except KeyError as exc:
        raise _missing(key) from exc


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    return _str_value(value, key)


def _non_negative_int_value(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"MetricCorrectionReceipt field '{key}' must be an integer.")
    try:
        coerced = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"MetricCorrectionReceipt field '{key}' must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"MetricCorrectionReceipt field '{key}' must be an integer.")
    if coerced < 0:
        raise ValueError(f"MetricCorrectionReceipt field '{key}' must be non-negative.")
    return coerced


def _non_negative_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        return _non_negative_int_value(payload[key], key)
    except KeyError as exc:
        raise _missing(key) from exc


def _finite_float_value(value: object, key: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"MetricCorrectionReceipt field '{key}' must be numeric.")
    try:
        coerced = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"MetricCorrectionReceipt field '{key}' must be numeric.") from exc
    if not math.isfinite(coerced):
        raise ValueError(f"MetricCorrectionReceipt field '{key}' must be finite.")
    return coerced


def _non_negative_float_value(value: object, key: str) -> float:
    coerced = _finite_float_value(value, key)
    if coerced < 0.0:
        raise ValueError(f"MetricCorrectionReceipt field '{key}' must be non-negative.")
    return coerced


def _string_list_value(value: object, key: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"MetricCorrectionReceipt field '{key}' must be a list.")
    return [str(entry) for entry in value]


def _int_list_value(value: object, key: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"MetricCorrectionReceipt field '{key}' must be a list.")
    return [_non_negative_int_value(entry, f"{key}[{idx}]") for idx, entry in enumerate(value)]


def _float_mapping_value(value: object, key: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"MetricCorrectionReceipt field '{key}' must be an object.")
    return {
        str(term): _finite_float_value(amount, f"{key}.{term}") for term, amount in value.items()
    }


def _promotion(payload: Mapping[str, Any]) -> Promotion:
    try:
        return normalize_promotion(payload["promotion"])
    except KeyError as exc:
        raise _missing("promotion") from exc


def _blocked_consumers_for_promotion(
    promotion: Promotion, blocked_consumers: list[str]
) -> list[str]:
    if promotion != 1 and not blocked_consumers:
        return list(DEFAULT_METRIC_CORRECTION_BLOCKED_CONSUMERS)
    return blocked_consumers


@dataclass(frozen=True, slots=True)
class MetricCorrectionEntry:
    """One local metric correction decision for a panel or operator."""

    panel_label: int
    correction_type: str
    delta_metric_meaning: str
    raw_residual: float
    corrected_residual: float
    energy_terms: dict[str, float]
    result_state: str
    blockers: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "panel_label", _non_negative_int_value(self.panel_label, "panel_label")
        )
        correction_type = _str_value(self.correction_type, "correction_type")
        if correction_type not in CORRECTION_TYPES:
            raise ValueError(
                "MetricCorrectionReceipt field 'correction_type' must be one of "
                f"{', '.join(CORRECTION_TYPES)}."
            )
        object.__setattr__(self, "correction_type", correction_type)
        object.__setattr__(
            self,
            "delta_metric_meaning",
            _str_value(self.delta_metric_meaning, "delta_metric_meaning"),
        )
        object.__setattr__(
            self, "raw_residual", _non_negative_float_value(self.raw_residual, "raw_residual")
        )
        object.__setattr__(
            self,
            "corrected_residual",
            _non_negative_float_value(self.corrected_residual, "corrected_residual"),
        )
        object.__setattr__(
            self, "energy_terms", _float_mapping_value(self.energy_terms, "energy_terms")
        )
        result_state = _str_value(self.result_state, "result_state")
        if result_state not in CORRECTION_STATES:
            raise ValueError(
                "MetricCorrectionReceipt field 'result_state' must be one of "
                f"{', '.join(CORRECTION_STATES)}."
            )
        object.__setattr__(self, "result_state", result_state)
        object.__setattr__(self, "blockers", _string_list_value(self.blockers, "blockers"))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MetricCorrectionEntry":
        return cls(
            panel_label=_non_negative_int(payload, "panel_label"),
            correction_type=_required_str(payload, "correction_type"),
            delta_metric_meaning=_required_str(payload, "delta_metric_meaning"),
            raw_residual=_non_negative_float_value(payload["raw_residual"], "raw_residual"),
            corrected_residual=_non_negative_float_value(
                payload["corrected_residual"],
                "corrected_residual",
            ),
            energy_terms=_float_mapping_value(payload.get("energy_terms", {}), "energy_terms"),
            result_state=_required_str(payload, "result_state"),
            blockers=_string_list_value(payload.get("blockers", []), "blockers"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "panel_label": int(self.panel_label),
            "correction_type": self.correction_type,
            "delta_metric_meaning": self.delta_metric_meaning,
            "raw_residual": float(self.raw_residual),
            "corrected_residual": float(self.corrected_residual),
            "energy_terms": dict(self.energy_terms),
            "result_state": self.result_state,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True, slots=True)
class MetricCorrectionReceipt:
    """Hash-linked receipt for authorized metric correction operators."""

    solver_receipt_hash: str
    cut_topology_receipt_hash: str
    seam_edges_hash: str
    panels_requiring_correction: list[int]
    corrections: list[MetricCorrectionEntry]
    raw_residual_total: float
    corrected_residual_total: float
    residual_gate: float
    promotion: Promotion
    blocked_consumers: list[str]
    metric_correction_blockers: list[str]
    correction_payload_hash: str | None = None

    def __post_init__(self) -> None:
        for key in ("solver_receipt_hash", "cut_topology_receipt_hash", "seam_edges_hash"):
            object.__setattr__(self, key, _str_value(getattr(self, key), key))
        object.__setattr__(
            self,
            "panels_requiring_correction",
            _int_list_value(self.panels_requiring_correction, "panels_requiring_correction"),
        )
        corrections = [
            entry
            if isinstance(entry, MetricCorrectionEntry)
            else MetricCorrectionEntry.from_mapping(entry)
            for entry in self.corrections
        ]
        object.__setattr__(self, "corrections", corrections)
        for key in ("raw_residual_total", "corrected_residual_total", "residual_gate"):
            object.__setattr__(self, key, _non_negative_float_value(getattr(self, key), key))
        object.__setattr__(self, "promotion", normalize_promotion(self.promotion))
        blocked_consumers = _string_list_value(self.blocked_consumers, "blocked_consumers")
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked_consumers_for_promotion(self.promotion, blocked_consumers),
        )
        object.__setattr__(
            self,
            "metric_correction_blockers",
            _string_list_value(self.metric_correction_blockers, "metric_correction_blockers"),
        )
        if self.correction_payload_hash is not None:
            object.__setattr__(
                self,
                "correction_payload_hash",
                _str_value(self.correction_payload_hash, "correction_payload_hash"),
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MetricCorrectionReceipt":
        raw_corrections = payload.get("corrections", [])
        if not isinstance(raw_corrections, list):
            raise TypeError("MetricCorrectionReceipt field 'corrections' must be a list.")
        return cls(
            solver_receipt_hash=_required_str(payload, "solver_receipt_hash"),
            cut_topology_receipt_hash=_required_str(payload, "cut_topology_receipt_hash"),
            seam_edges_hash=_required_str(payload, "seam_edges_hash"),
            panels_requiring_correction=_int_list_value(
                payload.get("panels_requiring_correction", []),
                "panels_requiring_correction",
            ),
            corrections=[
                MetricCorrectionEntry.from_mapping(entry)
                for entry in raw_corrections
                if isinstance(entry, Mapping)
            ],
            raw_residual_total=_non_negative_float_value(
                payload["raw_residual_total"],
                "raw_residual_total",
            ),
            corrected_residual_total=_non_negative_float_value(
                payload["corrected_residual_total"],
                "corrected_residual_total",
            ),
            residual_gate=_non_negative_float_value(payload["residual_gate"], "residual_gate"),
            promotion=_promotion(payload),
            blocked_consumers=_string_list_value(
                payload.get("blocked_consumers", []), "blocked_consumers"
            ),
            metric_correction_blockers=_string_list_value(
                payload.get("metric_correction_blockers", []),
                "metric_correction_blockers",
            ),
            correction_payload_hash=_optional_str(payload, "correction_payload_hash"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "MetricCorrectionReceipt":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("MetricCorrectionReceipt JSON must contain an object.")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "solver_receipt_hash": self.solver_receipt_hash,
            "cut_topology_receipt_hash": self.cut_topology_receipt_hash,
            "seam_edges_hash": self.seam_edges_hash,
            "panels_requiring_correction": list(self.panels_requiring_correction),
            "corrections": [entry.to_dict() for entry in self.corrections],
            "raw_residual_total": float(self.raw_residual_total),
            "corrected_residual_total": float(self.corrected_residual_total),
            "residual_gate": float(self.residual_gate),
            "promotion": int(self.promotion),
            "blocked_consumers": list(self.blocked_consumers),
            "metric_correction_blockers": list(self.metric_correction_blockers),
        }
        if self.correction_payload_hash is not None:
            payload["correction_payload_hash"] = self.correction_payload_hash
        return payload

    def to_json(self, path: str | Path) -> Path:
        receipt_path = Path(path)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return receipt_path


def load_metric_correction_receipt(path: str | Path) -> MetricCorrectionReceipt:
    return MetricCorrectionReceipt.from_json(path)


def with_promotion(receipt: MetricCorrectionReceipt, promotion: object) -> MetricCorrectionReceipt:
    next_promotion = normalize_promotion(promotion)
    return replace(
        receipt,
        promotion=next_promotion,
        blocked_consumers=_blocked_consumers_for_promotion(
            next_promotion,
            receipt.blocked_consumers,
        ),
    )


def can_consume_metric_correction_receipt(
    receipt: MetricCorrectionReceipt, consumer: str | None = None
) -> bool:
    if receipt.promotion != 1:
        return False
    if consumer is None:
        return not receipt.blocked_consumers
    return consumer not in receipt.blocked_consumers
