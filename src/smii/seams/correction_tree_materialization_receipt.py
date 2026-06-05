"""Typed receipts for materialized correction-tree operators."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

Promotion = Literal[-1, 0, 1]
CORRECTION_TREE_MATERIALIZATION_SCHEMA = "smii.correction_tree_materialization.v1"
DEFAULT_CORRECTION_TREE_MATERIALIZATION_BLOCKED_CONSUMERS = (
    "panel_unwrap",
    "manufacturing",
)

__all__ = [
    "CORRECTION_TREE_MATERIALIZATION_SCHEMA",
    "DEFAULT_CORRECTION_TREE_MATERIALIZATION_BLOCKED_CONSUMERS",
    "CorrectionTreeMaterializationEntry",
    "CorrectionTreeMaterializationReceipt",
    "Promotion",
    "can_consume_correction_tree_materialization_receipt",
    "load_correction_tree_materialization_receipt",
    "normalize_promotion",
    "with_promotion",
]


def normalize_promotion(value: object) -> Promotion:
    """Coerce and validate a receipt promotion state."""

    if isinstance(value, bool):
        raise ValueError(
            "CorrectionTreeMaterializationReceipt promotion must be one of -1, 0, or 1."
        )
    if isinstance(value, int):
        promotion = value
    elif isinstance(value, float) and value.is_integer():
        promotion = int(value)
    elif isinstance(value, str) and value in {"-1", "0", "1"}:
        promotion = int(value)
    else:
        raise ValueError(
            "CorrectionTreeMaterializationReceipt promotion must be one of -1, 0, or 1."
        )
    if promotion not in {-1, 0, 1}:
        raise ValueError(
            "CorrectionTreeMaterializationReceipt promotion must be one of -1, 0, or 1."
        )
    return cast(Promotion, promotion)


def _missing(key: str) -> KeyError:
    return KeyError(f"CorrectionTreeMaterializationReceipt is missing required field '{key}'.")


def _str_value(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"CorrectionTreeMaterializationReceipt field '{key}' must be a string.")
    if not value:
        raise ValueError(f"CorrectionTreeMaterializationReceipt field '{key}' must be non-empty.")
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


def _bool_value(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"CorrectionTreeMaterializationReceipt field '{key}' must be a boolean.")
    return value


def _required_bool(payload: Mapping[str, Any], key: str) -> bool:
    try:
        return _bool_value(payload[key], key)
    except KeyError as exc:
        raise _missing(key) from exc


def _non_negative_int_value(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"CorrectionTreeMaterializationReceipt field '{key}' must be an integer.")
    try:
        coerced = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"CorrectionTreeMaterializationReceipt field '{key}' must be an integer."
        ) from exc
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"CorrectionTreeMaterializationReceipt field '{key}' must be an integer.")
    if coerced < 0:
        raise ValueError(
            f"CorrectionTreeMaterializationReceipt field '{key}' must be non-negative."
        )
    return coerced


def _non_negative_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        return _non_negative_int_value(payload[key], key)
    except KeyError as exc:
        raise _missing(key) from exc


def _string_list_value(value: object, key: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"CorrectionTreeMaterializationReceipt field '{key}' must be a list.")
    return [str(entry) for entry in value]


def _json_object_value(value: object, key: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"CorrectionTreeMaterializationReceipt field '{key}' must be an object.")
    return {str(entry_key): entry_value for entry_key, entry_value in value.items()}


def _optional_json_object(payload: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    return _json_object_value(value, key)


def _int_list_value(value: object, key: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"CorrectionTreeMaterializationReceipt field '{key}' must be a list.")
    return [_non_negative_int_value(entry, f"{key}[{idx}]") for idx, entry in enumerate(value)]


def _promotion(payload: Mapping[str, Any]) -> Promotion:
    try:
        return normalize_promotion(payload["promotion"])
    except KeyError as exc:
        raise _missing("promotion") from exc


def _blocked_consumers_for_promotion(
    promotion: Promotion, blocked_consumers: list[str]
) -> list[str]:
    if promotion != 1 and not blocked_consumers:
        return list(DEFAULT_CORRECTION_TREE_MATERIALIZATION_BLOCKED_CONSUMERS)
    return blocked_consumers


@dataclass(frozen=True, slots=True)
class CorrectionTreeMaterializationEntry:
    """One realized correction-tree operator materialization."""

    node_id: str
    operator_family: str
    metric_realized: bool
    chart_materialized: bool
    materialization_kind: str
    affected_panels: list[int]
    backend_constraints_emitted: bool
    promotion: Promotion
    blockers: list[str]
    geometry: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _str_value(self.node_id, "node_id"))
        object.__setattr__(
            self,
            "operator_family",
            _str_value(self.operator_family, "operator_family"),
        )
        object.__setattr__(
            self,
            "metric_realized",
            _bool_value(self.metric_realized, "metric_realized"),
        )
        object.__setattr__(
            self,
            "chart_materialized",
            _bool_value(self.chart_materialized, "chart_materialized"),
        )
        object.__setattr__(
            self,
            "materialization_kind",
            _str_value(self.materialization_kind, "materialization_kind"),
        )
        object.__setattr__(
            self,
            "affected_panels",
            _int_list_value(self.affected_panels, "affected_panels"),
        )
        object.__setattr__(
            self,
            "backend_constraints_emitted",
            _bool_value(
                self.backend_constraints_emitted,
                "backend_constraints_emitted",
            ),
        )
        object.__setattr__(self, "promotion", normalize_promotion(self.promotion))
        object.__setattr__(
            self,
            "blockers",
            _string_list_value(self.blockers, "blockers"),
        )
        if self.geometry is not None:
            object.__setattr__(
                self,
                "geometry",
                _json_object_value(self.geometry, "geometry"),
            )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "CorrectionTreeMaterializationEntry":
        """Build a validated materialization entry from a JSON-like mapping."""

        return cls(
            node_id=_required_str(payload, "node_id"),
            operator_family=_required_str(payload, "operator_family"),
            metric_realized=_required_bool(payload, "metric_realized"),
            chart_materialized=_required_bool(payload, "chart_materialized"),
            materialization_kind=_required_str(payload, "materialization_kind"),
            affected_panels=_int_list_value(
                payload.get("affected_panels", []),
                "affected_panels",
            ),
            backend_constraints_emitted=_required_bool(
                payload,
                "backend_constraints_emitted",
            ),
            promotion=_promotion(payload),
            blockers=_string_list_value(payload.get("blockers", []), "blockers"),
            geometry=_optional_json_object(payload, "geometry"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable materialization entry."""

        payload: dict[str, object] = {
            "node_id": self.node_id,
            "operator_family": self.operator_family,
            "metric_realized": bool(self.metric_realized),
            "chart_materialized": bool(self.chart_materialized),
            "materialization_kind": self.materialization_kind,
            "affected_panels": list(self.affected_panels),
            "backend_constraints_emitted": bool(self.backend_constraints_emitted),
            "promotion": int(self.promotion),
            "blockers": list(self.blockers),
        }
        if self.geometry is not None:
            payload["geometry"] = dict(self.geometry)
        return payload


@dataclass(frozen=True, slots=True)
class CorrectionTreeMaterializationReceipt:
    """Hash-linked receipt for correction-tree materialization outcomes."""

    correction_tree_hash: str
    materializations: list[CorrectionTreeMaterializationEntry]
    materialized_operator_count: int
    promotion: Promotion
    blocked_consumers: list[str]
    blockers: list[str]
    correction_tree_receipt_hash: str | None = None
    correction_operator_scoring_receipt_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "correction_tree_hash",
            _str_value(self.correction_tree_hash, "correction_tree_hash"),
        )
        materializations = [
            entry
            if isinstance(entry, CorrectionTreeMaterializationEntry)
            else CorrectionTreeMaterializationEntry.from_mapping(entry)
            for entry in self.materializations
        ]
        object.__setattr__(self, "materializations", materializations)
        materialized_operator_count = _non_negative_int_value(
            self.materialized_operator_count,
            "materialized_operator_count",
        )
        if materialized_operator_count != len(materializations):
            raise ValueError(
                "CorrectionTreeMaterializationReceipt field "
                "'materialized_operator_count' length must match materializations."
            )
        object.__setattr__(
            self,
            "materialized_operator_count",
            materialized_operator_count,
        )
        object.__setattr__(self, "promotion", normalize_promotion(self.promotion))
        blocked_consumers = _string_list_value(self.blocked_consumers, "blocked_consumers")
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked_consumers_for_promotion(self.promotion, blocked_consumers),
        )
        object.__setattr__(
            self,
            "blockers",
            _string_list_value(self.blockers, "blockers"),
        )
        if self.correction_tree_receipt_hash is not None:
            object.__setattr__(
                self,
                "correction_tree_receipt_hash",
                _str_value(
                    self.correction_tree_receipt_hash,
                    "correction_tree_receipt_hash",
                ),
            )
        if self.correction_operator_scoring_receipt_hash is not None:
            object.__setattr__(
                self,
                "correction_operator_scoring_receipt_hash",
                _str_value(
                    self.correction_operator_scoring_receipt_hash,
                    "correction_operator_scoring_receipt_hash",
                ),
            )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "CorrectionTreeMaterializationReceipt":
        """Build a validated receipt from a JSON-like mapping."""

        raw_materializations = payload.get("materializations", [])
        if not isinstance(raw_materializations, list):
            raise TypeError(
                "CorrectionTreeMaterializationReceipt field 'materializations' must be a list."
            )
        materializations: list[CorrectionTreeMaterializationEntry] = []
        for idx, entry in enumerate(raw_materializations):
            if not isinstance(entry, Mapping):
                raise TypeError(
                    "CorrectionTreeMaterializationReceipt field "
                    f"'materializations[{idx}]' must be an object."
                )
            materializations.append(CorrectionTreeMaterializationEntry.from_mapping(entry))

        return cls(
            correction_tree_hash=_required_str(payload, "correction_tree_hash"),
            materializations=materializations,
            materialized_operator_count=_non_negative_int(
                payload,
                "materialized_operator_count",
            ),
            promotion=_promotion(payload),
            blocked_consumers=_string_list_value(
                payload.get("blocked_consumers", []),
                "blocked_consumers",
            ),
            blockers=_string_list_value(payload.get("blockers", []), "blockers"),
            correction_tree_receipt_hash=_optional_str(
                payload,
                "correction_tree_receipt_hash",
            ),
            correction_operator_scoring_receipt_hash=_optional_str(
                payload,
                "correction_operator_scoring_receipt_hash",
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CorrectionTreeMaterializationReceipt":
        """Load a receipt from a JSON document."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("CorrectionTreeMaterializationReceipt JSON must contain an object.")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable receipt payload."""

        payload: dict[str, object] = {
            "schema_version": CORRECTION_TREE_MATERIALIZATION_SCHEMA,
            "correction_tree_hash": self.correction_tree_hash,
            "materializations": [entry.to_dict() for entry in self.materializations],
            "materialized_operator_count": int(self.materialized_operator_count),
            "promotion": int(self.promotion),
            "blocked_consumers": list(self.blocked_consumers),
            "blockers": list(self.blockers),
        }
        if self.correction_tree_receipt_hash is not None:
            payload["correction_tree_receipt_hash"] = self.correction_tree_receipt_hash
        if self.correction_operator_scoring_receipt_hash is not None:
            payload["correction_operator_scoring_receipt_hash"] = (
                self.correction_operator_scoring_receipt_hash
            )
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


def load_correction_tree_materialization_receipt(
    path: str | Path,
) -> CorrectionTreeMaterializationReceipt:
    """Load a correction-tree materialization receipt from JSON."""

    return CorrectionTreeMaterializationReceipt.from_json(path)


def with_promotion(
    receipt: CorrectionTreeMaterializationReceipt,
    promotion: object,
) -> CorrectionTreeMaterializationReceipt:
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


def can_consume_correction_tree_materialization_receipt(
    receipt: CorrectionTreeMaterializationReceipt,
    consumer: str | None = None,
) -> bool:
    """Return whether a materialization receipt is promoted downstream."""

    if receipt.promotion != 1:
        return False
    if consumer is None:
        return not receipt.blocked_consumers
    return consumer not in receipt.blocked_consumers
