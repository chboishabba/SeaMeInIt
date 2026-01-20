"""Interfaces for deriving seam cost fields from ROM aggregation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .aggregation import FieldStats, RomAggregation

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
