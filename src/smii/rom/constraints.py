"""Constraint registry for seam-aware ROM work."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence


@dataclass(frozen=True, slots=True)
class ConstraintSet:
    """Raw constraint payloads loaded from disk."""

    forbidden_vertices: Mapping[str, tuple[int, ...]]
    anchors: Mapping[str, int | None]
    symmetry_vertex_pairs: tuple[tuple[int, int], ...]
    symmetry_edge_pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    panel_limits_default: Mapping[str, float | int | None]
    panel_overrides: Mapping[str, Mapping[str, float | int | None]]


class ConstraintRegistry:
    """Provides deterministic constraint lookups for vertices and edges."""

    def __init__(self, constraint_set: ConstraintSet) -> None:
        self.constraint_set = constraint_set
        self._forbidden_cache = {
            int(vertex)
            for vertices in constraint_set.forbidden_vertices.values()
            for vertex in vertices
        }
        self._symmetry_vertex_map = self._build_symmetry_map(constraint_set.symmetry_vertex_pairs)

    @staticmethod
    def _build_symmetry_map(pairs: Iterable[tuple[int, int]]) -> Mapping[int, int]:
        mapping: MutableMapping[int, int] = {}
        for left, right in pairs:
            mapping[int(left)] = int(right)
            mapping[int(right)] = int(left)
        return mapping

    def is_vertex_forbidden(self, vertex_id: int) -> bool:
        """Return True if the vertex is forbidden."""

        return int(vertex_id) in self._forbidden_cache

    def anchor_vertex(self, name: str) -> int | None:
        """Return the configured anchor vertex id, if present."""

        return self.constraint_set.anchors.get(name)

    def symmetry_partner(self, vertex_id: int) -> int | None:
        """Return the mirrored vertex id, if defined."""

        return self._symmetry_vertex_map.get(int(vertex_id))

    def panel_limits(self) -> Mapping[str, Mapping[str, float | int | None]]:
        """Return defaults and garment overrides for panel limits."""

        return {
            "default": self.constraint_set.panel_limits_default,
            "garment_overrides": self.constraint_set.panel_overrides,
        }

    def panel_limits_for(self, garment: str) -> Mapping[str, float | int | None]:
        """Return panel limits merged with any garment override if present."""

        overrides = self.constraint_set.panel_overrides.get(garment)
        if overrides:
            merged: MutableMapping[str, float | int | None] = {
                **self.constraint_set.panel_limits_default
            }
            merged.update({str(k): v for k, v in overrides.items()})
            return merged
        return self.constraint_set.panel_limits_default


def _load_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Constraint file '{path}' must contain a JSON object.")
    return payload


def _to_int_tuple(sequence: Iterable[object]) -> tuple[int, ...]:
    return tuple(int(value) for value in sequence)


def _parse_edge_pairs(raw_pairs: Sequence[object]) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    parsed: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for entry in raw_pairs:
        left: Sequence[object] | None = None
        right: Sequence[object] | None = None

        if isinstance(entry, Mapping):
            left = entry.get("left") or entry.get("source")
            right = entry.get("right") or entry.get("target")
        elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray)):
            if len(entry) == 2 and all(isinstance(item, Sequence) for item in entry):
                left, right = entry  # type: ignore[assignment]
            elif len(entry) == 4:
                left, right = entry[:2], entry[2:]

        if left is None or right is None:
            continue

        if len(left) != 2 or len(right) != 2:
            raise ValueError("Symmetry edge pairs must contain exactly two vertices per side.")

        parsed.append(((int(left[0]), int(left[1])), (int(right[0]), int(right[1]))))

    return tuple(parsed)


def load_constraints(root: str | Path) -> ConstraintRegistry:
    """Load constraint manifests from a directory."""

    root_path = Path(root)
    forbidden = _load_json(root_path / "forbidden_vertices.json")
    anchors = _load_json(root_path / "anchors.json")
    symmetry = _load_json(root_path / "symmetry_pairs.json")
    panel_limits = _load_json(root_path / "panel_limits.json")

    forbidden_vertices = {
        key: _to_int_tuple(value)
        for key, value in forbidden.get("regions", {}).items()  # type: ignore[arg-type]
    }
    anchor_vertices = {
        key: (int(value) if value is not None else None)
        for key, value in anchors.get("landmarks", {}).items()  # type: ignore[arg-type]
    }
    symmetry_vertex_pairs = tuple(
        (int(left), int(right))
        for left, right in symmetry.get("vertex_pairs", [])  # type: ignore[arg-type]
    )
    symmetry_edge_pairs = _parse_edge_pairs(symmetry.get("edge_pairs", []))  # type: ignore[arg-type]

    limits_payload = panel_limits.get("default", {}) if isinstance(panel_limits, Mapping) else {}
    overrides_payload = (
        panel_limits.get("garment_overrides", {}) if isinstance(panel_limits, Mapping) else {}
    )

    constraint_set = ConstraintSet(
        forbidden_vertices=forbidden_vertices,
        anchors=anchor_vertices,
        symmetry_vertex_pairs=symmetry_vertex_pairs,
        symmetry_edge_pairs=symmetry_edge_pairs,
        panel_limits_default={str(key): value for key, value in limits_payload.items()},
        panel_overrides={str(k): v for k, v in overrides_payload.items()},
    )
    return ConstraintRegistry(constraint_set)
