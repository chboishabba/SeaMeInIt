"""Placement utilities for hard-shell attachment interfaces."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import count
import math
from typing import Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np

from ..schemas.validators import validate_attachment_catalog

__all__ = [
    "Attachment",
    "AttachmentLayout",
    "AttachmentPlanner",
    "AttachmentRouting",
    "PanelSegment",
]


Vector = np.ndarray


def _as_array(values: Sequence[Sequence[float]] | Sequence[float]) -> Vector:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.astype(float)
    return array


def _normalize(vector: Vector) -> Vector:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize zero-length vector.")
    return vector / norm


@dataclass(frozen=True, slots=True)
class PanelSegment:
    """Description of a segmented shell panel used for attachment placement."""

    panel_id: str
    outline: tuple[tuple[float, float, float], ...]
    normal: tuple[float, float, float] | None = None
    tags: frozenset[str] = frozenset()

    def __init__(
        self,
        panel_id: str,
        outline: Sequence[Sequence[float]],
        *,
        normal: Sequence[float] | None = None,
        tags: Iterable[str] | None = None,
    ) -> None:
        if len(outline) < 3:
            raise ValueError("Panel outlines must contain at least three vertices.")
        processed_outline = tuple(tuple(float(coord) for coord in vertex) for vertex in outline)
        object.__setattr__(self, "panel_id", str(panel_id))
        object.__setattr__(self, "outline", processed_outline)
        object.__setattr__(self, "normal", tuple(float(x) for x in normal) if normal is not None else None)
        object.__setattr__(self, "tags", frozenset(tags or ()))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "PanelSegment":
        panel_identifier = payload.get("panel_id")
        if panel_identifier is None:
            raise KeyError("Panel mapping must define 'panel_id'.")
        outline = payload.get("outline")
        if not isinstance(outline, Sequence):
            raise TypeError("Panel mapping must include an 'outline' sequence.")
        tags_field = payload.get("tags")
        tags: Iterable[str] | None = None
        if isinstance(tags_field, Iterable) and not isinstance(tags_field, (str, bytes)):
            tags = [str(tag) for tag in tags_field]
        normal = payload.get("normal")
        normal_values: Sequence[float] | None
        if isinstance(normal, Sequence):
            normal_values = [float(value) for value in normal]
        else:
            normal_values = None
        return cls(str(panel_identifier), outline, normal=normal_values, tags=tags)

    def polygon(self) -> Vector:
        return _as_array(self.outline)

    def centroid(self) -> tuple[float, float, float]:
        poly = self.polygon()
        return tuple(np.mean(poly, axis=0).tolist())  # type: ignore[return-value]

    def normal_vector(self) -> Vector:
        if self.normal is not None:
            return _normalize(_as_array(self.normal))
        poly = self.polygon()
        ref = poly[0]
        for idx in range(1, len(poly) - 1):
            edge_a = poly[idx] - ref
            edge_b = poly[idx + 1] - ref
            candidate = np.cross(edge_a, edge_b)
            if np.linalg.norm(candidate) > 1e-9:
                return _normalize(candidate)
        raise ValueError(f"Unable to derive a normal for panel '{self.panel_id}'.")

    def basis(self) -> tuple[Vector, Vector, Vector]:
        normal = self.normal_vector()
        reference = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(reference, normal)) > 0.999:
            reference = np.array([1.0, 0.0, 0.0])
        u_axis = _normalize(np.cross(normal, reference))
        v_axis = _normalize(np.cross(normal, u_axis))
        return normal, u_axis, v_axis

    def extent_along(self, axis: Vector) -> float:
        axis = _normalize(axis)
        poly = self.polygon()
        projections = poly @ axis
        return float(projections.max() - projections.min())

    def to_metadata(self) -> Mapping[str, object]:
        return {
            "panel_id": self.panel_id,
            "centroid": list(self.centroid()),
            "normal": list(self.normal_vector()),
            "tags": sorted(self.tags),
        }


@dataclass(frozen=True, slots=True)
class AttachmentRouting:
    """Routing metadata for a placed attachment."""

    harness: str
    path: tuple[str, ...]
    bundle_tags: tuple[str, ...]
    estimated_length: float

    def to_dict(self) -> dict[str, object]:
        return {
            "harness": self.harness,
            "path": list(self.path),
            "bundle_tags": list(self.bundle_tags),
            "estimated_length": self.estimated_length,
        }


@dataclass(frozen=True, slots=True)
class Attachment:
    """Placed fastening primitive on a hard-shell panel."""

    identifier: str
    panel_id: str
    connector_id: str
    connector_type: str
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    compatibility_tags: tuple[str, ...]
    routing: AttachmentRouting
    fastener_pattern: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "panel_id": self.panel_id,
            "connector_id": self.connector_id,
            "connector_type": self.connector_type,
            "position": list(self.position),
            "normal": list(self.normal),
            "compatibility_tags": list(self.compatibility_tags),
            "routing": self.routing.to_dict(),
            "fastener_pattern": dict(self.fastener_pattern),
        }


@dataclass(slots=True)
class AttachmentLayout:
    """Aggregate result from attachment placement."""

    attachments: list[Attachment]
    panel_summary: Mapping[str, Mapping[str, object]]

    def visualization_payload(self) -> dict[str, list[Mapping[str, object]]]:
        markers: list[Mapping[str, object]] = []
        routing_paths: list[Mapping[str, object]] = []
        for attachment in self.attachments:
            markers.append(
                {
                    "id": attachment.identifier,
                    "panel_id": attachment.panel_id,
                    "connector_type": attachment.connector_type,
                    "position": list(attachment.position),
                    "normal": list(attachment.normal),
                    "label": attachment.connector_id,
                }
            )
            routing_paths.append(
                {
                    "attachment_id": attachment.identifier,
                    "harness": attachment.routing.harness,
                    "path": list(attachment.routing.path),
                    "estimated_length": attachment.routing.estimated_length,
                    "bundle_tags": list(attachment.routing.bundle_tags),
                }
            )
        return {
            "attachment_markers": markers,
            "routing_paths": routing_paths,
        }


class AttachmentPlanner:
    """Plan attachment placements on segmented panels."""

    def __init__(self, catalog: Mapping[str, object]) -> None:
        validate_attachment_catalog(catalog)
        self._catalog = catalog
        self._connectors = {
            connector["id"]: connector
            for connector in catalog.get("connectors", [])  # type: ignore[union-attr]
            if isinstance(connector, Mapping) and "id" in connector
        }
        self._placement_rules: list[Mapping[str, object]] = [
            rule for rule in catalog.get("placement_rules", []) if isinstance(rule, Mapping)
        ]

    @property
    def catalog(self) -> Mapping[str, object]:
        return self._catalog

    def place(self, panels: Sequence[PanelSegment]) -> AttachmentLayout:
        attachments: list[Attachment] = []
        panel_summary: MutableMapping[str, MutableMapping[str, object]] = {}
        identifier_counter = count(start=1)

        for panel in panels:
            panel_summary.setdefault(panel.panel_id, {"attachment_count": 0, "rules_applied": []})
            applicable_rules = [rule for rule in self._placement_rules if self._rule_matches(panel, rule)]
            for rule in applicable_rules:
                connectors = self._connectors_for_rule(rule)
                if not connectors:
                    continue
                connector = connectors[0]
                placements = self._placements_for_rule(panel, rule, connector, identifier_counter)
                if not placements:
                    continue
                panel_summary[panel.panel_id]["rules_applied"].append(rule.get("panel", panel.panel_id))
                panel_summary[panel.panel_id]["attachment_count"] = (
                    panel_summary[panel.panel_id]["attachment_count"] + len(placements)
                )
                attachments.extend(placements)

        for summary in panel_summary.values():
            summary["rules_applied"] = list(dict.fromkeys(summary["rules_applied"]))
        return AttachmentLayout(attachments=attachments, panel_summary=panel_summary)

    def _rule_matches(self, panel: PanelSegment, rule: Mapping[str, object]) -> bool:
        panel_name = rule.get("panel")
        if panel_name is not None and str(panel_name) != panel.panel_id:
            return False
        tags_any = rule.get("panel_tags_any")
        if tags_any:
            required_any = {str(tag) for tag in tags_any if isinstance(tag, (str, bytes))}
            if not (panel.tags & required_any):
                return False
        tags_all = rule.get("panel_tags_all")
        if tags_all:
            required_all = {str(tag) for tag in tags_all if isinstance(tag, (str, bytes))}
            if not required_all.issubset(panel.tags):
                return False
        return True

    def _connectors_for_rule(self, rule: Mapping[str, object]) -> list[Mapping[str, object]]:
        allowed_types = [str(entry) for entry in rule.get("allowed_connector_types", [])]
        required_compat = {str(tag) for tag in rule.get("required_compatibility", [])}
        connectors: list[Mapping[str, object]] = []
        for connector in self._connectors.values():
            connector_type = str(connector.get("connector_type"))
            if allowed_types and connector_type not in allowed_types:
                continue
            connector_tags = {str(tag) for tag in connector.get("compatibility_tags", [])}
            if required_compat and not required_compat.issubset(connector_tags):
                continue
            connectors.append(connector)
        connectors.sort(key=lambda item: str(item.get("id")))
        return connectors

    def _placements_for_rule(
        self,
        panel: PanelSegment,
        rule: Mapping[str, object],
        connector: Mapping[str, object],
        identifier_counter: Iterator[int],
    ) -> list[Attachment]:
        max_per_panel = max(1, int(rule.get("max_per_panel", 1)))
        spacing = float(rule.get("spacing", 0.0))
        edge_clearance = max(0.0, float(rule.get("edge_clearance", 0.0)))
        normal_offset = self._normal_offset(rule, connector)
        harness = str(rule.get("harness", "primary"))
        routing_tags = tuple(str(tag) for tag in rule.get("routing_tags", []))
        nominal_length = float(rule.get("nominal_cable_length", 0.0))

        centroid = np.asarray(panel.centroid())
        normal, u_axis, v_axis = panel.basis()
        u_extent = panel.extent_along(u_axis)
        v_extent = panel.extent_along(v_axis)

        orientation = str(rule.get("orientation", "auto"))
        if orientation not in {"auto", "longitudinal", "transverse", "u", "v"}:
            orientation = "auto"
        if orientation in {"longitudinal", "u"}:
            axis = u_axis
            axis_extent = u_extent
        elif orientation in {"transverse", "v"}:
            axis = v_axis
            axis_extent = v_extent
        else:
            if u_extent >= v_extent:
                axis, axis_extent = u_axis, u_extent
            else:
                axis, axis_extent = v_axis, v_extent

        available_span = max(0.0, axis_extent - 2 * edge_clearance)
        if spacing <= 0.0:
            spacing = axis_extent / max(max_per_panel, 1)
        spacing = max(spacing, 1e-6)
        if available_span <= 1e-9:
            count = 1
        elif max_per_panel == 1:
            count = 1
        else:
            count = min(max_per_panel, int(math.floor(available_span / spacing)) + 1)
            count = max(1, count)
        if max_per_panel == 1:
            offsets = [0.0]
        else:
            if available_span <= 1e-9 or count == 1:
                offsets = [0.0]
            else:
                effective_span = min(available_span, spacing * (count - 1))
                offsets = np.linspace(-effective_span / 2.0, effective_span / 2.0, count).tolist()

        attachments: list[Attachment] = []
        compatibility_tags = tuple(str(tag) for tag in connector.get("compatibility_tags", []))
        fastener_pattern = dict(connector.get("fastener_pattern", {}))
        fastener_pattern.setdefault("primitive", "bolt")

        for offset_value in offsets:
            position_vector = centroid + axis * float(offset_value) + normal * normal_offset
            identifier = f"{panel.panel_id}-{connector.get('id')}-{next(identifier_counter)}"
            routing_length = nominal_length if nominal_length > 0 else float(abs(normal_offset) + axis_extent / 2.0)
            routing = AttachmentRouting(
                harness=harness,
                path=(panel.panel_id,),
                bundle_tags=routing_tags,
                estimated_length=routing_length,
            )
            attachments.append(
                Attachment(
                    identifier=identifier,
                    panel_id=panel.panel_id,
                    connector_id=str(connector.get("id")),
                    connector_type=str(connector.get("connector_type")),
                    position=tuple(float(value) for value in position_vector.tolist()),
                    normal=tuple(float(value) for value in normal.tolist()),
                    compatibility_tags=compatibility_tags,
                    routing=routing,
                    fastener_pattern=fastener_pattern,
                )
            )

        return attachments[:max_per_panel]

    def _normal_offset(self, rule: Mapping[str, object], connector: Mapping[str, object]) -> float:
        if "normal_offset" in rule:
            return float(rule["normal_offset"])
        placement = connector.get("placement")
        if isinstance(placement, Mapping) and "normal_offset" in placement:
            return float(placement["normal_offset"])
        return 0.0
