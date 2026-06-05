"""Typed receipts for promoted cut-topology artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, cast

Promotion = Literal[-1, 0, 1]
DEFAULT_CUT_TOPOLOGY_BLOCKED_CONSUMERS = ("panel_unwrap", "manufacturing")

__all__ = [
    "DEFAULT_CUT_TOPOLOGY_BLOCKED_CONSUMERS",
    "CutTopologyReceipt",
    "Promotion",
    "can_consume_cut_topology_receipt",
    "load_cut_topology_receipt",
    "normalize_promotion",
    "with_promotion",
]


def normalize_promotion(value: object) -> Promotion:
    """Coerce and validate a receipt promotion state."""

    if isinstance(value, bool):
        raise ValueError("CutTopologyReceipt promotion must be one of -1, 0, or 1.")
    if isinstance(value, int):
        promotion = value
    elif isinstance(value, float) and value.is_integer():
        promotion = int(value)
    elif isinstance(value, str) and value in {"-1", "0", "1"}:
        promotion = int(value)
    else:
        raise ValueError("CutTopologyReceipt promotion must be one of -1, 0, or 1.")
    if promotion not in {-1, 0, 1}:
        raise ValueError("CutTopologyReceipt promotion must be one of -1, 0, or 1.")
    return cast(Promotion, promotion)


def _missing(key: str) -> KeyError:
    return KeyError(f"CutTopologyReceipt is missing required field '{key}'.")


def _coerce_str_value(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"CutTopologyReceipt field '{key}' must be a string.")
    if not value:
        raise ValueError(f"CutTopologyReceipt field '{key}' must be non-empty.")
    return value


def _coerce_required_str(payload: Mapping[str, Any], key: str) -> str:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_str_value(value, key)


def _coerce_bool_value(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"CutTopologyReceipt field '{key}' must be a boolean.")
    return value


def _coerce_bool(payload: Mapping[str, Any], key: str) -> bool:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_bool_value(value, key)


def _coerce_non_negative_int_value(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"CutTopologyReceipt field '{key}' must be an integer.")
    try:
        coerced = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"CutTopologyReceipt field '{key}' must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"CutTopologyReceipt field '{key}' must be an integer.")
    if coerced < 0:
        raise ValueError(f"CutTopologyReceipt field '{key}' must be non-negative.")
    return coerced


def _coerce_non_negative_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_non_negative_int_value(value, key)


def _coerce_int_list_value(value: object, key: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"CutTopologyReceipt field '{key}' must be a list.")
    return [
        _coerce_non_negative_int_value(entry, f"{key}[{idx}]") for idx, entry in enumerate(value)
    ]


def _coerce_int_list(payload: Mapping[str, Any], key: str) -> list[int]:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_int_list_value(value, key)


def _coerce_string_list_value(value: object, key: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"CutTopologyReceipt field '{key}' must be a list.")
    return [str(entry) for entry in value]


def _coerce_string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    try:
        value = payload[key]
    except KeyError as exc:
        raise _missing(key) from exc
    return _coerce_string_list_value(value, key)


def _coerce_promotion(payload: Mapping[str, Any]) -> Promotion:
    try:
        value = payload["promotion"]
    except KeyError as exc:
        raise _missing("promotion") from exc
    return normalize_promotion(value)


def _coerce_blocked_consumers(payload: Mapping[str, Any]) -> list[str]:
    blocked_consumers = payload.get("blocked_consumers", [])
    return _coerce_string_list_value(blocked_consumers, "blocked_consumers")


def _blocked_consumers_for_promotion(
    promotion: Promotion, blocked_consumers: list[str]
) -> list[str]:
    if promotion != 1 and not blocked_consumers:
        return list(DEFAULT_CUT_TOPOLOGY_BLOCKED_CONSUMERS)
    return blocked_consumers


@dataclass(frozen=True, slots=True)
class CutTopologyReceipt:
    """Hash-linked receipt for cut-graph admissibility decisions."""

    solver_receipt_hash: str
    mesh_hash: str
    seam_edges_hash: str
    seam_edge_segment_count: int
    seam_vertex_count: int
    seam_connected_component_count: int
    seam_endpoint_count: int
    seam_branch_vertex_count: int
    panel_count: int
    panel_face_counts: list[int]
    panel_boundary_edge_counts: list[int]
    panels_are_disks: bool
    typed_dart_count: int
    typed_gusset_count: int
    promotion: Promotion
    blocked_consumers: list[str]
    cut_topology_blockers: list[str]
    ordinary_boundary_component_count: int = 0
    typed_operator_count: int = 0
    typed_relief_cut_count: int = 0
    typed_ease_count: int = 0
    typed_stretch_zone_count: int = 0
    invalid_fragmentation_count: int = 0
    seam_graph_classifications: list[str] | None = None

    def __post_init__(self) -> None:
        for key in ("solver_receipt_hash", "mesh_hash", "seam_edges_hash"):
            object.__setattr__(self, key, _coerce_str_value(getattr(self, key), key))
        for key in (
            "seam_edge_segment_count",
            "seam_vertex_count",
            "seam_connected_component_count",
            "seam_endpoint_count",
            "seam_branch_vertex_count",
            "panel_count",
            "typed_dart_count",
            "typed_gusset_count",
            "ordinary_boundary_component_count",
            "typed_operator_count",
            "typed_relief_cut_count",
            "typed_ease_count",
            "typed_stretch_zone_count",
            "invalid_fragmentation_count",
        ):
            object.__setattr__(self, key, _coerce_non_negative_int_value(getattr(self, key), key))
        panel_face_counts = _coerce_int_list_value(self.panel_face_counts, "panel_face_counts")
        panel_boundary_edge_counts = _coerce_int_list_value(
            self.panel_boundary_edge_counts,
            "panel_boundary_edge_counts",
        )
        if len(panel_face_counts) != self.panel_count:
            raise ValueError(
                "CutTopologyReceipt field 'panel_face_counts' length must match panel_count."
            )
        if len(panel_boundary_edge_counts) != self.panel_count:
            raise ValueError(
                "CutTopologyReceipt field 'panel_boundary_edge_counts' length must match panel_count."
            )
        object.__setattr__(self, "panel_face_counts", panel_face_counts)
        object.__setattr__(self, "panel_boundary_edge_counts", panel_boundary_edge_counts)
        object.__setattr__(
            self,
            "panels_are_disks",
            _coerce_bool_value(self.panels_are_disks, "panels_are_disks"),
        )
        object.__setattr__(self, "promotion", normalize_promotion(self.promotion))
        blocked_consumers = _coerce_string_list_value(
            self.blocked_consumers,
            "blocked_consumers",
        )
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked_consumers_for_promotion(self.promotion, blocked_consumers),
        )
        object.__setattr__(
            self,
            "cut_topology_blockers",
            _coerce_string_list_value(self.cut_topology_blockers, "cut_topology_blockers"),
        )
        classifications = (
            [] if self.seam_graph_classifications is None else self.seam_graph_classifications
        )
        object.__setattr__(
            self,
            "seam_graph_classifications",
            _coerce_string_list_value(classifications, "seam_graph_classifications"),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CutTopologyReceipt":
        """Build a validated receipt from a JSON-like mapping."""

        return cls(
            solver_receipt_hash=_coerce_required_str(payload, "solver_receipt_hash"),
            mesh_hash=_coerce_required_str(payload, "mesh_hash"),
            seam_edges_hash=_coerce_required_str(payload, "seam_edges_hash"),
            seam_edge_segment_count=_coerce_non_negative_int(
                payload,
                "seam_edge_segment_count",
            ),
            seam_vertex_count=_coerce_non_negative_int(payload, "seam_vertex_count"),
            seam_connected_component_count=_coerce_non_negative_int(
                payload,
                "seam_connected_component_count",
            ),
            seam_endpoint_count=_coerce_non_negative_int(payload, "seam_endpoint_count"),
            seam_branch_vertex_count=_coerce_non_negative_int(
                payload,
                "seam_branch_vertex_count",
            ),
            panel_count=_coerce_non_negative_int(payload, "panel_count"),
            panel_face_counts=_coerce_int_list(payload, "panel_face_counts"),
            panel_boundary_edge_counts=_coerce_int_list(
                payload,
                "panel_boundary_edge_counts",
            ),
            panels_are_disks=_coerce_bool(payload, "panels_are_disks"),
            typed_dart_count=_coerce_non_negative_int(payload, "typed_dart_count"),
            typed_gusset_count=_coerce_non_negative_int(payload, "typed_gusset_count"),
            promotion=_coerce_promotion(payload),
            blocked_consumers=_coerce_blocked_consumers(payload),
            cut_topology_blockers=_coerce_string_list(payload, "cut_topology_blockers"),
            ordinary_boundary_component_count=_coerce_non_negative_int_value(
                payload.get("ordinary_boundary_component_count", 0),
                "ordinary_boundary_component_count",
            ),
            typed_operator_count=_coerce_non_negative_int_value(
                payload.get("typed_operator_count", 0),
                "typed_operator_count",
            ),
            typed_relief_cut_count=_coerce_non_negative_int_value(
                payload.get("typed_relief_cut_count", 0),
                "typed_relief_cut_count",
            ),
            typed_ease_count=_coerce_non_negative_int_value(
                payload.get("typed_ease_count", 0),
                "typed_ease_count",
            ),
            typed_stretch_zone_count=_coerce_non_negative_int_value(
                payload.get("typed_stretch_zone_count", 0),
                "typed_stretch_zone_count",
            ),
            invalid_fragmentation_count=_coerce_non_negative_int_value(
                payload.get("invalid_fragmentation_count", 0),
                "invalid_fragmentation_count",
            ),
            seam_graph_classifications=_coerce_string_list_value(
                payload.get("seam_graph_classifications", []),
                "seam_graph_classifications",
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CutTopologyReceipt":
        """Load a receipt from a JSON document."""

        receipt_path = Path(path)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("CutTopologyReceipt JSON must contain an object.")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable receipt payload."""

        return {
            "solver_receipt_hash": self.solver_receipt_hash,
            "mesh_hash": self.mesh_hash,
            "seam_edges_hash": self.seam_edges_hash,
            "seam_edge_segment_count": int(self.seam_edge_segment_count),
            "seam_vertex_count": int(self.seam_vertex_count),
            "seam_connected_component_count": int(self.seam_connected_component_count),
            "seam_endpoint_count": int(self.seam_endpoint_count),
            "seam_branch_vertex_count": int(self.seam_branch_vertex_count),
            "panel_count": int(self.panel_count),
            "panel_face_counts": list(self.panel_face_counts),
            "panel_boundary_edge_counts": list(self.panel_boundary_edge_counts),
            "panels_are_disks": bool(self.panels_are_disks),
            "typed_dart_count": int(self.typed_dart_count),
            "typed_gusset_count": int(self.typed_gusset_count),
            "ordinary_boundary_component_count": int(self.ordinary_boundary_component_count),
            "typed_operator_count": int(self.typed_operator_count),
            "typed_relief_cut_count": int(self.typed_relief_cut_count),
            "typed_ease_count": int(self.typed_ease_count),
            "typed_stretch_zone_count": int(self.typed_stretch_zone_count),
            "invalid_fragmentation_count": int(self.invalid_fragmentation_count),
            "seam_graph_classifications": list(self.seam_graph_classifications or []),
            "promotion": int(self.promotion),
            "blocked_consumers": list(self.blocked_consumers),
            "cut_topology_blockers": list(self.cut_topology_blockers),
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


def load_cut_topology_receipt(path: str | Path) -> CutTopologyReceipt:
    """Load a cut topology receipt from JSON."""

    return CutTopologyReceipt.from_json(path)


def with_promotion(receipt: CutTopologyReceipt, promotion: object) -> CutTopologyReceipt:
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


def can_consume_cut_topology_receipt(
    receipt: CutTopologyReceipt, consumer: str | None = None
) -> bool:
    """Return whether a cut topology receipt is promoted for downstream consumers."""

    if receipt.promotion != 1:
        return False
    if consumer is None:
        return not receipt.blocked_consumers
    return consumer not in receipt.blocked_consumers
