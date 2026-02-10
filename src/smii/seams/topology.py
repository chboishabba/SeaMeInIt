"""Topology helpers for seam edge selections."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, MutableMapping, Sequence

Edge = tuple[int, int]


def normalize_edge(edge: Edge) -> Edge:
    """Return a canonical undirected edge tuple."""

    a, b = int(edge[0]), int(edge[1])
    return (a, b) if a <= b else (b, a)


def edge_degrees(edges: Sequence[Edge]) -> Mapping[int, int]:
    """Compute per-vertex degree for an undirected edge list."""

    degrees: MutableMapping[int, int] = defaultdict(int)
    for raw in edges:
        edge = normalize_edge(raw)
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
    return dict(degrees)


def branch_excess(edges: Sequence[Edge], max_degree: int | None) -> int:
    """Return total degree overage above ``max_degree`` across vertices."""

    if max_degree is None:
        return 0
    if max_degree < 1:
        return 0
    return int(sum(max(0, degree - max_degree) for degree in edge_degrees(edges).values()))


def branch_penalty(edges: Sequence[Edge], max_degree: int | None, penalty_weight: float) -> float:
    """Return weighted topology penalty for excessive branching."""

    if max_degree is None or penalty_weight <= 0.0:
        return 0.0
    return float(branch_excess(edges, max_degree)) * float(penalty_weight)


def incremental_branch_penalty(
    edge: Edge,
    *,
    degrees: Mapping[int, int],
    max_degree: int | None,
    penalty_weight: float,
) -> float:
    """Return penalty increase if ``edge`` were added to the current degree map."""

    if max_degree is None or max_degree < 1 or penalty_weight <= 0.0:
        return 0.0

    edge_norm = normalize_edge(edge)
    delta = 0.0
    for vertex in edge_norm:
        before = max(0, int(degrees.get(vertex, 0)) - max_degree)
        after = max(0, int(degrees.get(vertex, 0)) + 1 - max_degree)
        delta += float(after - before) * float(penalty_weight)
    return delta


def _edge_cost(edge: Edge, edge_costs: Mapping[Edge, float]) -> float:
    normalized = normalize_edge(edge)
    return float(edge_costs.get(normalized, edge_costs.get((normalized[1], normalized[0]), 0.0)))


def regularize_branching(
    edges: Sequence[Edge],
    edge_costs: Mapping[Edge, float],
    *,
    max_degree: int | None,
    preserve_nonempty: bool = True,
) -> tuple[tuple[Edge, ...], tuple[str, ...]]:
    """Drop high-cost incident edges until degree overage is resolved when possible."""

    selected = [normalize_edge(edge) for edge in edges]
    selected = sorted(set(selected))
    warnings: list[str] = []

    if max_degree is None or max_degree < 1 or not selected:
        return tuple(selected), tuple(warnings)

    while True:
        degrees = edge_degrees(selected)
        overloaded = [vertex for vertex, degree in degrees.items() if degree > max_degree]
        if not overloaded:
            break

        candidates: list[tuple[float, Edge, int]] = []
        for vertex in overloaded:
            incident = [edge for edge in selected if vertex in edge]
            if not incident:
                continue
            worst = max(incident, key=lambda edge: (_edge_cost(edge, edge_costs), edge))
            candidates.append((_edge_cost(worst, edge_costs), worst, vertex))

        if not candidates:
            warnings.append("branch regularization found no removable overloaded edge")
            break

        _, edge_to_drop, overloaded_vertex = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        if preserve_nonempty and len(selected) <= 1:
            warnings.append("branch regularization stopped to keep at least one seam edge")
            break
        selected.remove(edge_to_drop)
        warnings.append(
            f"dropped edge {edge_to_drop} to reduce branching at vertex {overloaded_vertex}"
        )

    if branch_excess(selected, max_degree) > 0:
        warnings.append(
            f"branching remains above max_degree={max_degree} after regularization"
        )
    return tuple(sorted(selected)), tuple(warnings)
