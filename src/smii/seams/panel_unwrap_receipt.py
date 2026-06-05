"""Typed receipts for promoted panel-unwrap artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

Promotion = Literal[-1, 0, 1]
GRAIN_DIRECTIONS = ("warp", "weft", "bias")
UNWRAP_BACKENDS = ("bootstrap_projection", "lscm", "xatlas")
DEFAULT_PANEL_UNWRAP_BLOCKED_CONSUMERS = ("manufacturing",)

__all__ = [
    "DEFAULT_PANEL_UNWRAP_BLOCKED_CONSUMERS",
    "GRAIN_DIRECTIONS",
    "PanelUnwrapReceipt",
    "Promotion",
    "UNWRAP_BACKENDS",
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
        coerced = int(value)  # type: ignore[call-overload]
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


def _coerce_optional_finite_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    return _coerce_finite_float_value(value, key)


def _coerce_optional_non_negative_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    return _coerce_non_negative_float_value(value, key)


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
        _coerce_non_negative_float_value(value, f"{key}[{idx}]") for idx, value in enumerate(values)
    ]


def _coerce_optional_float_list(payload: Mapping[str, Any], key: str) -> list[float] | None:
    values = payload.get(key)
    if values is None:
        return None
    if not isinstance(values, list):
        raise TypeError(f"PanelUnwrapReceipt field '{key}' must be a list.")
    return [
        _coerce_non_negative_float_value(value, f"{key}[{idx}]") for idx, value in enumerate(values)
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


def _coerce_optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    return _coerce_str_value(value, key)


def _coerce_optional_bool(payload: Mapping[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"PanelUnwrapReceipt field '{key}' must be a boolean.")
    return value


def _coerce_string_list_value(value: object, key: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"PanelUnwrapReceipt field '{key}' must be a list.")
    return [str(entry) for entry in value]


def _coerce_optional_string_list(payload: Mapping[str, Any], key: str) -> list[str] | None:
    value = payload.get(key)
    if value is None:
        return None
    return _coerce_string_list_value(value, key)


def _json_object_value(value: object, key: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"PanelUnwrapReceipt field '{key}' must be an object.")
    return dict(value)


def _coerce_optional_json_object(payload: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    return _json_object_value(value, key)


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
    cut_topology_receipt_hash: str | None = None
    unwrap_backend: str | None = None
    backend_is_bootstrap: bool | None = None
    distortion_margin: float | None = None
    panel_unwrap_blockers: list[str] | None = None
    per_panel_corrected_residual: list[float] | None = None
    worst_corrected_residual: float | None = None
    mean_corrected_residual: float | None = None
    correction_payload_hash: str | None = None
    metric_correction_receipt_hash: str | None = None
    fabric_metric_receipt: dict[str, Any] | None = None
    correction_tree_receipt: dict[str, Any] | None = None
    correction_operator_scoring_receipt: dict[str, Any] | None = None
    realized_correction_operator_receipt: dict[str, Any] | None = None
    correction_tree_materialization_receipt: dict[str, Any] | None = None
    serialization_competition_receipt: dict[str, Any] | None = None
    selected_backend_per_panel: list[str] | None = None
    serialization_promoted: bool | None = None

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
                "PanelUnwrapReceipt field 'per_panel_distortion' length must match panel_count."
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
        if self.cut_topology_receipt_hash is not None:
            object.__setattr__(
                self,
                "cut_topology_receipt_hash",
                _coerce_str_value(
                    self.cut_topology_receipt_hash,
                    "cut_topology_receipt_hash",
                ),
            )
        object.__setattr__(self, "promotion", normalize_promotion(self.promotion))
        blocked_consumers = _coerce_blocked_consumer_values(self.blocked_consumers)
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked_consumers_for_promotion(self.promotion, blocked_consumers),
        )
        if self.unwrap_backend is not None:
            backend = _coerce_str_value(self.unwrap_backend, "unwrap_backend")
            if backend not in UNWRAP_BACKENDS:
                raise ValueError(
                    "PanelUnwrapReceipt field 'unwrap_backend' must be one of "
                    f"{', '.join(UNWRAP_BACKENDS)}."
                )
            object.__setattr__(self, "unwrap_backend", backend)
        if self.backend_is_bootstrap is not None and not isinstance(
            self.backend_is_bootstrap,
            bool,
        ):
            raise TypeError("PanelUnwrapReceipt field 'backend_is_bootstrap' must be a boolean.")
        if self.distortion_margin is not None:
            object.__setattr__(
                self,
                "distortion_margin",
                _coerce_finite_float_value(self.distortion_margin, "distortion_margin"),
            )
        if self.panel_unwrap_blockers is not None:
            object.__setattr__(
                self,
                "panel_unwrap_blockers",
                _coerce_string_list_value(
                    self.panel_unwrap_blockers,
                    "panel_unwrap_blockers",
                ),
            )
        if self.per_panel_corrected_residual is not None:
            corrected = [
                _coerce_non_negative_float_value(value, f"per_panel_corrected_residual[{idx}]")
                for idx, value in enumerate(self.per_panel_corrected_residual)
            ]
            if len(corrected) != panel_count:
                raise ValueError(
                    "PanelUnwrapReceipt field 'per_panel_corrected_residual' length must match "
                    "panel_count."
                )
            object.__setattr__(self, "per_panel_corrected_residual", corrected)
        for key in ("worst_corrected_residual", "mean_corrected_residual"):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, _coerce_non_negative_float_value(value, key))
        if self.correction_payload_hash is not None:
            object.__setattr__(
                self,
                "correction_payload_hash",
                _coerce_str_value(self.correction_payload_hash, "correction_payload_hash"),
            )
        if self.metric_correction_receipt_hash is not None:
            object.__setattr__(
                self,
                "metric_correction_receipt_hash",
                _coerce_str_value(
                    self.metric_correction_receipt_hash,
                    "metric_correction_receipt_hash",
                ),
            )
        if self.fabric_metric_receipt is not None:
            object.__setattr__(
                self,
                "fabric_metric_receipt",
                _json_object_value(self.fabric_metric_receipt, "fabric_metric_receipt"),
            )
        if self.correction_tree_receipt is not None:
            object.__setattr__(
                self,
                "correction_tree_receipt",
                _json_object_value(self.correction_tree_receipt, "correction_tree_receipt"),
            )
        if self.correction_operator_scoring_receipt is not None:
            object.__setattr__(
                self,
                "correction_operator_scoring_receipt",
                _json_object_value(
                    self.correction_operator_scoring_receipt,
                    "correction_operator_scoring_receipt",
                ),
            )
        if self.realized_correction_operator_receipt is not None:
            object.__setattr__(
                self,
                "realized_correction_operator_receipt",
                _json_object_value(
                    self.realized_correction_operator_receipt,
                    "realized_correction_operator_receipt",
                ),
            )
        if self.correction_tree_materialization_receipt is not None:
            object.__setattr__(
                self,
                "correction_tree_materialization_receipt",
                _json_object_value(
                    self.correction_tree_materialization_receipt,
                    "correction_tree_materialization_receipt",
                ),
            )
        if self.serialization_competition_receipt is not None:
            object.__setattr__(
                self,
                "serialization_competition_receipt",
                _json_object_value(
                    self.serialization_competition_receipt,
                    "serialization_competition_receipt",
                ),
            )
        if self.selected_backend_per_panel is not None:
            selected = _coerce_string_list_value(
                self.selected_backend_per_panel,
                "selected_backend_per_panel",
            )
            if len(selected) != panel_count:
                raise ValueError(
                    "PanelUnwrapReceipt field 'selected_backend_per_panel' length must match "
                    "panel_count."
                )
            invalid = [backend for backend in selected if backend not in UNWRAP_BACKENDS]
            if invalid:
                raise ValueError(
                    "PanelUnwrapReceipt field 'selected_backend_per_panel' entries must be one "
                    f"of {', '.join(UNWRAP_BACKENDS)}."
                )
            object.__setattr__(self, "selected_backend_per_panel", selected)
        if self.serialization_promoted is not None and not isinstance(
            self.serialization_promoted,
            bool,
        ):
            raise TypeError("PanelUnwrapReceipt field 'serialization_promoted' must be a boolean.")

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
            cut_topology_receipt_hash=_coerce_optional_str(
                payload,
                "cut_topology_receipt_hash",
            ),
            unwrap_backend=_coerce_optional_str(payload, "unwrap_backend"),
            backend_is_bootstrap=_coerce_optional_bool(payload, "backend_is_bootstrap"),
            distortion_margin=_coerce_optional_finite_float(payload, "distortion_margin"),
            panel_unwrap_blockers=_coerce_optional_string_list(
                payload,
                "panel_unwrap_blockers",
            ),
            per_panel_corrected_residual=_coerce_optional_float_list(
                payload,
                "per_panel_corrected_residual",
            ),
            worst_corrected_residual=_coerce_optional_non_negative_float(
                payload,
                "worst_corrected_residual",
            ),
            mean_corrected_residual=_coerce_optional_non_negative_float(
                payload,
                "mean_corrected_residual",
            ),
            correction_payload_hash=_coerce_optional_str(payload, "correction_payload_hash"),
            metric_correction_receipt_hash=_coerce_optional_str(
                payload,
                "metric_correction_receipt_hash",
            ),
            fabric_metric_receipt=_coerce_optional_json_object(
                payload,
                "fabric_metric_receipt",
            ),
            correction_tree_receipt=_coerce_optional_json_object(
                payload,
                "correction_tree_receipt",
            ),
            correction_operator_scoring_receipt=_coerce_optional_json_object(
                payload,
                "correction_operator_scoring_receipt",
            ),
            realized_correction_operator_receipt=_coerce_optional_json_object(
                payload,
                "realized_correction_operator_receipt",
            ),
            correction_tree_materialization_receipt=_coerce_optional_json_object(
                payload,
                "correction_tree_materialization_receipt",
            ),
            serialization_competition_receipt=_coerce_optional_json_object(
                payload,
                "serialization_competition_receipt",
            ),
            selected_backend_per_panel=_coerce_optional_string_list(
                payload,
                "selected_backend_per_panel",
            ),
            serialization_promoted=_coerce_optional_bool(payload, "serialization_promoted"),
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

        payload: dict[str, object] = {
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
        optional_fields = {
            "cut_topology_receipt_hash": self.cut_topology_receipt_hash,
            "unwrap_backend": self.unwrap_backend,
            "backend_is_bootstrap": self.backend_is_bootstrap,
            "distortion_margin": self.distortion_margin,
            "panel_unwrap_blockers": self.panel_unwrap_blockers,
            "per_panel_corrected_residual": self.per_panel_corrected_residual,
            "worst_corrected_residual": self.worst_corrected_residual,
            "mean_corrected_residual": self.mean_corrected_residual,
            "correction_payload_hash": self.correction_payload_hash,
            "metric_correction_receipt_hash": self.metric_correction_receipt_hash,
            "fabric_metric_receipt": self.fabric_metric_receipt,
            "correction_tree_receipt": self.correction_tree_receipt,
            "correction_operator_scoring_receipt": self.correction_operator_scoring_receipt,
            "realized_correction_operator_receipt": self.realized_correction_operator_receipt,
            "correction_tree_materialization_receipt": self.correction_tree_materialization_receipt,
            "serialization_competition_receipt": self.serialization_competition_receipt,
            "selected_backend_per_panel": self.selected_backend_per_panel,
            "serialization_promoted": self.serialization_promoted,
        }
        payload.update({key: value for key, value in optional_fields.items() if value is not None})
        return payload

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
