"""Typed receipts for cut-topology repair decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

Promotion = Literal[-1, 0, 1]
DEFAULT_CUT_TOPOLOGY_REPAIR_BLOCKED_CONSUMERS = ("panel_unwrap", "manufacturing")
CUT_TOPOLOGY_REPAIR_SCHEMA = "smii.cut_topology_repair.v1"

__all__ = [
    "CutTopologyRepairReceipt",
    "CUT_TOPOLOGY_REPAIR_SCHEMA",
    "DEFAULT_CUT_TOPOLOGY_REPAIR_BLOCKED_CONSUMERS",
    "Promotion",
    "can_consume_cut_topology_repair_receipt",
    "load_cut_topology_repair_receipt",
    "normalize_promotion",
    "with_promotion",
]


def normalize_promotion(value: object) -> Promotion:
    """Coerce and validate a receipt promotion state."""

    if isinstance(value, bool):
        raise ValueError("CutTopologyRepairReceipt promotion must be one of -1, 0, or 1.")
    if isinstance(value, int):
        promotion = value
    elif isinstance(value, float) and value.is_integer():
        promotion = int(value)
    elif isinstance(value, str) and value in {"-1", "0", "1"}:
        promotion = int(value)
    else:
        raise ValueError("CutTopologyRepairReceipt promotion must be one of -1, 0, or 1.")
    if promotion not in {-1, 0, 1}:
        raise ValueError("CutTopologyRepairReceipt promotion must be one of -1, 0, or 1.")
    return cast(Promotion, promotion)


def _missing(key: str) -> KeyError:
    return KeyError(f"CutTopologyRepairReceipt is missing required field '{key}'.")


def _str_value(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"CutTopologyRepairReceipt field '{key}' must be a string.")
    if not value:
        raise ValueError(f"CutTopologyRepairReceipt field '{key}' must be non-empty.")
    return value


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    try:
        return _str_value(payload[key], key)
    except KeyError as exc:
        raise _missing(key) from exc


def _non_negative_int_value(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"CutTopologyRepairReceipt field '{key}' must be an integer.")
    try:
        coerced = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"CutTopologyRepairReceipt field '{key}' must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"CutTopologyRepairReceipt field '{key}' must be an integer.")
    if coerced < 0:
        raise ValueError(f"CutTopologyRepairReceipt field '{key}' must be non-negative.")
    return coerced


def _non_negative_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        return _non_negative_int_value(payload[key], key)
    except KeyError as exc:
        raise _missing(key) from exc


def _string_list_value(value: object, key: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"CutTopologyRepairReceipt field '{key}' must be a list.")
    return [str(entry) for entry in value]


def _string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    try:
        return _string_list_value(payload[key], key)
    except KeyError as exc:
        raise _missing(key) from exc


def _panel_checks_value(value: object, key: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError(f"CutTopologyRepairReceipt field '{key}' must be a list.")
    checks: list[dict[str, object]] = []
    for idx, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise TypeError(
                f"CutTopologyRepairReceipt field '{key}[{idx}]' must be an object."
            )
        checks.append({str(check_key): check_value for check_key, check_value in entry.items()})
    return checks


def _panel_checks(payload: Mapping[str, Any], key: str) -> list[dict[str, object]]:
    try:
        return _panel_checks_value(payload[key], key)
    except KeyError as exc:
        raise _missing(key) from exc


def _promotion(payload: Mapping[str, Any]) -> Promotion:
    try:
        return normalize_promotion(payload["promotion"])
    except KeyError as exc:
        raise _missing("promotion") from exc


def _blocked_consumers_for_promotion(
    promotion: Promotion, blocked_consumers: list[str]
) -> list[str]:
    if promotion != 1 and not blocked_consumers:
        return list(DEFAULT_CUT_TOPOLOGY_REPAIR_BLOCKED_CONSUMERS)
    return blocked_consumers


@dataclass(frozen=True, slots=True)
class CutTopologyRepairReceipt:
    """Hash-linked receipt for cut-topology repair outcomes."""

    input_cut_topology_hash: str
    mesh_hash: str
    seam_edges_hash: str
    panel_count: int
    panel_checks: list[dict[str, object]]
    repairs: list[str]
    promotion: Promotion
    blocked_consumers: list[str]
    repair_blockers: list[str]

    def __post_init__(self) -> None:
        for key in ("input_cut_topology_hash", "mesh_hash", "seam_edges_hash"):
            object.__setattr__(self, key, _str_value(getattr(self, key), key))
        object.__setattr__(
            self,
            "panel_count",
            _non_negative_int_value(self.panel_count, "panel_count"),
        )
        panel_checks = _panel_checks_value(self.panel_checks, "panel_checks")
        if len(panel_checks) != self.panel_count:
            raise ValueError(
                "CutTopologyRepairReceipt field 'panel_checks' length must match panel_count."
            )
        object.__setattr__(self, "panel_checks", panel_checks)
        object.__setattr__(self, "repairs", _string_list_value(self.repairs, "repairs"))
        object.__setattr__(self, "promotion", normalize_promotion(self.promotion))
        blocked_consumers = _string_list_value(self.blocked_consumers, "blocked_consumers")
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked_consumers_for_promotion(self.promotion, blocked_consumers),
        )
        object.__setattr__(
            self,
            "repair_blockers",
            _string_list_value(self.repair_blockers, "repair_blockers"),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CutTopologyRepairReceipt":
        """Build a validated receipt from a JSON-like mapping."""

        return cls(
            input_cut_topology_hash=_required_str(payload, "input_cut_topology_hash"),
            mesh_hash=_required_str(payload, "mesh_hash"),
            seam_edges_hash=_required_str(payload, "seam_edges_hash"),
            panel_count=_non_negative_int(payload, "panel_count"),
            panel_checks=_panel_checks(payload, "panel_checks"),
            repairs=_string_list(payload, "repairs"),
            promotion=_promotion(payload),
            blocked_consumers=_string_list_value(
                payload.get("blocked_consumers", []),
                "blocked_consumers",
            ),
            repair_blockers=_string_list(payload, "repair_blockers"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CutTopologyRepairReceipt":
        """Load a receipt from a JSON document."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("CutTopologyRepairReceipt JSON must contain an object.")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable receipt payload."""

        return {
            "schema_version": CUT_TOPOLOGY_REPAIR_SCHEMA,
            "input_cut_topology_hash": self.input_cut_topology_hash,
            "mesh_hash": self.mesh_hash,
            "seam_edges_hash": self.seam_edges_hash,
            "panel_count": int(self.panel_count),
            "panel_checks": [dict(check) for check in self.panel_checks],
            "repairs": list(self.repairs),
            "promotion": int(self.promotion),
            "blocked_consumers": list(self.blocked_consumers),
            "repair_blockers": list(self.repair_blockers),
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


def load_cut_topology_repair_receipt(path: str | Path) -> CutTopologyRepairReceipt:
    """Load a cut topology repair receipt from JSON."""

    return CutTopologyRepairReceipt.from_json(path)


def with_promotion(
    receipt: CutTopologyRepairReceipt, promotion: object
) -> CutTopologyRepairReceipt:
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


def can_consume_cut_topology_repair_receipt(
    receipt: CutTopologyRepairReceipt, consumer: str | None = None
) -> bool:
    """Return whether a repair receipt is promoted for downstream consumers."""

    if receipt.promotion != 1:
        return False
    if consumer is None:
        return not receipt.blocked_consumers
    return consumer not in receipt.blocked_consumers
