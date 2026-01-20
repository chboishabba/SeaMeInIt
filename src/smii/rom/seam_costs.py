"""Interfaces for deriving seam cost fields from ROM aggregation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, MutableMapping, Sequence

import numpy as np

from .aggregation import FieldStats, RomAggregation

if TYPE_CHECKING:  # pragma: no cover
    from suit.seam_generator import SeamGraph

CostArray = np.ndarray


@dataclass(frozen=True, slots=True)
class SeamCostField:
    """Cost surfaces derived from ROM aggregation for seam solvers."""

    field: str
    vertex_costs: CostArray
    edge_costs: CostArray
    edges: tuple[tuple[int, int], ...]
    samples_used: int
    metadata: Mapping[str, float]

    def to_edge_weights(self) -> Mapping[tuple[int, int], float]:
        """Return a mapping compatible with seam graphs or solvers."""

        return {edge: float(self.edge_costs[idx]) for idx, edge in enumerate(self.edges)}


def _normalise(values: np.ndarray, *, minimum: float) -> np.ndarray:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.zeros_like(values)
    scale = max(float(finite_values.max()), minimum)
    if scale <= 0:
        return np.zeros_like(values)
    return np.clip(values / scale, minimum, None)


def _build_costs(stats: FieldStats, *, variance_weight: float, maximum_weight: float, minimum_cost: float) -> np.ndarray:
    if stats.sample_count == 0:
        return np.full_like(stats.mean, np.nan)

    variance_term = _normalise(np.sqrt(stats.variance), minimum=minimum_cost)
    maximum_term = _normalise(np.abs(stats.maximum), minimum=minimum_cost)
    costs = variance_weight * variance_term + maximum_weight * maximum_term
    return np.clip(costs, minimum_cost, None)


def build_seam_cost_field(
    aggregation: RomAggregation,
    *,
    field: str,
    variance_weight: float = 1.0,
    maximum_weight: float = 0.25,
    minimum_cost: float = 1e-6,
) -> SeamCostField:
    """Map ROM aggregation outputs to seam graph-ready cost weights."""

    stats = aggregation.per_field.get(field)
    if stats is None:
        raise KeyError(f"Field '{field}' is not present in the aggregation.")

    vertex_costs = _build_costs(stats, variance_weight=variance_weight, maximum_weight=maximum_weight, minimum_cost=minimum_cost)

    edges: Sequence[tuple[int, int]] = aggregation.edges or tuple()
    edge_costs: np.ndarray
    if edges and field in aggregation.per_edge_field:
        edge_stats = aggregation.per_edge_field[field]
        edge_costs = _build_costs(
            edge_stats, variance_weight=variance_weight, maximum_weight=maximum_weight, minimum_cost=minimum_cost
        )
    else:
        edge_costs = np.zeros(len(edges), dtype=float) if edges else np.zeros(0, dtype=float)

    metadata = {
        "variance_weight": float(variance_weight),
        "maximum_weight": float(maximum_weight),
        "minimum_cost": float(minimum_cost),
    }

    return SeamCostField(
        field=field,
        vertex_costs=vertex_costs,
        edge_costs=edge_costs,
        edges=tuple(edges),
        samples_used=stats.sample_count,
        metadata=metadata,
    )


def annotate_seam_graph_with_costs(cost_field: SeamCostField, seam_graph: "SeamGraph") -> "SeamGraph":
    """Attach seam-aware cost summaries to a generated seam graph."""

    from suit.seam_generator import SeamGraph  # Local import to avoid heavy dependencies at module import time.

    edge_lookup: Mapping[tuple[int, int], float] = cost_field.to_edge_weights()
    normalized_lookup: dict[tuple[int, int], float] = {
        tuple(sorted(edge)): value for edge, value in edge_lookup.items()
    }

    seam_costs: MutableMapping[str, Mapping[str, float]] = {}
    for panel in seam_graph.panels:
        seam_vertices = tuple(panel.seam_vertices)
        vertex_costs = (
            np.asarray(cost_field.vertex_costs, dtype=float)[np.asarray(seam_vertices, dtype=int)]
            if seam_vertices
            else np.asarray([], dtype=float)
        )
        vertex_mean = float(np.nanmean(vertex_costs)) if vertex_costs.size else float("nan")
        vertex_max = float(np.nanmax(vertex_costs)) if vertex_costs.size else float("nan")

        seam_vertex_set = {int(vertex) for vertex in seam_vertices}
        edge_costs = [
            value
            for edge, value in normalized_lookup.items()
            if edge[0] in seam_vertex_set and edge[1] in seam_vertex_set
        ]
        edge_mean = float(np.nanmean(edge_costs)) if edge_costs else float("nan")
        edge_max = float(np.nanmax(edge_costs)) if edge_costs else float("nan")

        seam_costs[panel.name] = MappingProxyType(
            {
                "vertex_cost_mean": vertex_mean,
                "vertex_cost_max": vertex_max,
                "edge_cost_mean": edge_mean,
                "edge_cost_max": edge_max,
                "samples_used": int(cost_field.samples_used),
            }
        )

    return SeamGraph(
        panels=seam_graph.panels,
        measurement_loops=seam_graph.measurement_loops,
        seam_metadata=seam_graph.seam_metadata,
        seam_costs=MappingProxyType(dict(seam_costs)),
    )


def save_seam_cost_field(cost_field: SeamCostField, path: str | Path) -> None:
    """Persist seam cost field arrays to a compressed NPZ."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        field=cost_field.field,
        vertex_costs=cost_field.vertex_costs,
        edge_costs=cost_field.edge_costs,
        edges=np.asarray(cost_field.edges, dtype=int),
        samples_used=int(cost_field.samples_used),
        metadata=dict(cost_field.metadata),
    )


def load_seam_cost_field(path: str | Path) -> SeamCostField:
    """Load a seam cost field from NPZ created by :func:`save_seam_cost_field`."""

    cost_path = Path(path)
    payload = np.load(cost_path, allow_pickle=True)
    field = str(payload["field"])
    vertex_costs = np.asarray(payload["vertex_costs"], dtype=float)
    edge_costs = np.asarray(payload["edge_costs"], dtype=float)
    edges_arr = np.asarray(payload["edges"], dtype=int)
    edges = tuple(tuple(int(val) for val in row) for row in edges_arr.tolist())
    samples_used = int(payload["samples_used"])
    metadata_raw = payload.get("metadata", {})
    metadata = dict(metadata_raw.item()) if hasattr(metadata_raw, "item") else dict(metadata_raw)
    return SeamCostField(
        field=field,
        vertex_costs=vertex_costs,
        edge_costs=edge_costs,
        edges=edges,
        samples_used=samples_used,
        metadata=metadata,
    )
