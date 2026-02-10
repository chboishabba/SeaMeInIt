"""Edge cost derivation helpers for seam optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import warnings

from smii.rom.seam_costs import SeamCostField

try:  # Optional import to avoid heavy deps at module import time
    from suit.seam_generator import SeamGraph
except Exception:  # pragma: no cover
    SeamGraph = None  # type: ignore[assignment]

EdgeAggregationMode = Literal["mean", "max", "integral"]


@dataclass(frozen=True, slots=True)
class EdgeCostResult:
    """Edge costs aligned to a seam graph."""

    edges: tuple[tuple[int, int], ...]
    costs: np.ndarray
    mode: EdgeAggregationMode

    def as_mapping(self) -> Mapping[tuple[int, int], float]:
        """Return an edge→cost mapping."""

        return {edge: float(self.costs[idx]) for idx, edge in enumerate(self.edges)}


def _edge_length(edge: tuple[int, int], vertices: np.ndarray | None) -> float:
    if vertices is None:
        return 1.0
    a, b = edge
    if a >= len(vertices) or b >= len(vertices):
        return 1.0
    return float(np.linalg.norm(vertices[a] - vertices[b]))


def _aggregate_cost(
    vertex_costs: np.ndarray,
    mode: EdgeAggregationMode,
    edge: tuple[int, int],
    *,
    vertices: np.ndarray | None,
) -> float:
    a, b = edge
    costs = np.asarray([vertex_costs[a], vertex_costs[b]], dtype=float)
    if mode == "max":
        base = float(np.nanmax(costs))
    else:
        base = float(np.nanmean(costs))
    if not np.isfinite(base):
        base = 0.0
    if mode == "integral":
        base *= _edge_length(edge, vertices)
    return base


def _seam_vertex_set(seam_graph: SeamGraph) -> set[int]:
    vertices: set[int] = set()
    for panel in seam_graph.panels:
        vertices.update(int(idx) for idx in panel.seam_vertices)
    return vertices


def _edge_filter(
    edges: Iterable[tuple[int, int]],
    seam_vertices: set[int],
) -> tuple[tuple[int, int], ...]:
    filtered: list[tuple[int, int]] = []
    for raw in edges:
        a, b = int(raw[0]), int(raw[1])
        if a in seam_vertices and b in seam_vertices:
            filtered.append((min(a, b), max(a, b)))
    filtered.sort()
    return tuple(filtered)


def build_edge_costs(
    cost_field: SeamCostField,
    seam_graph: "SeamGraph",
    *,
    vertices: np.ndarray | None = None,
    mode: EdgeAggregationMode = "mean",
) -> EdgeCostResult:
    """Derive seam-edge costs from vertex costs.

    Parameters
    ----------
    cost_field:
        SeamCostField carrying ROM-derived per-vertex costs (and optional per-edge weights).
    seam_graph:
        SeamGraph whose seam vertices define the allowable edge subset.
    vertices:
        Optional global vertex positions; required for length-weighted (integral) aggregation.
    mode:
        Aggregation mode: ``mean`` (default), ``max``, or ``integral`` (length-weighted mean).
    """

    if SeamGraph is None:
        raise ImportError("suit.seam_generator.SeamGraph is required to build edge costs.")
    if mode not in ("mean", "max", "integral"):
        raise ValueError(f"Unsupported edge aggregation mode: {mode}")

    seam_vertices = _seam_vertex_set(seam_graph)
    if not seam_vertices:
        warnings.warn("Seam graph has no seam vertices; edge costs are empty.", RuntimeWarning, stacklevel=2)
        return EdgeCostResult(edges=tuple(), costs=np.zeros(0, dtype=float), mode=mode)

    candidate_edges: Sequence[tuple[int, int]] = cost_field.edges or tuple()
    filtered_edges = _edge_filter(candidate_edges, seam_vertices)
    if not filtered_edges:
        warnings.warn(
            "No edges intersect seam vertices; edge costs default to empty mapping.",
            RuntimeWarning,
            stacklevel=2,
        )
        return EdgeCostResult(edges=tuple(), costs=np.zeros(0, dtype=float), mode=mode)

    costs: list[float] = []
    for edge in filtered_edges:
        costs.append(_aggregate_cost(cost_field.vertex_costs, mode, edge, vertices=vertices))

    return EdgeCostResult(edges=filtered_edges, costs=np.asarray(costs, dtype=float), mode=mode)
