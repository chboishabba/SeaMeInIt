"""Min-cut seam solver sharing the kernel + MDL objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from smii.rom.constraints import ConstraintRegistry
from smii.rom.seam_costs import SeamCostField
from smii.seams.kernels import EdgeKernel, KernelWeights, edge_energy
from smii.seams.mdl import MDLPrior, mdl_cost
from smii.seams.solver import PanelSolution, SeamSolution

try:  # Optional heavy import
    from suit.seam_generator import SeamGraph
except Exception:  # pragma: no cover
    SeamGraph = None  # type: ignore[assignment]

__all__ = ["Partition", "solve_seams_mincut"]


@dataclass(frozen=True, slots=True)
class Partition:
    """Predefined vertex partitions for a panel."""

    sources: tuple[int, ...]
    sinks: tuple[int, ...]


def _edge_costs(kernels: Mapping[tuple[int, int], EdgeKernel], weights: KernelWeights) -> Mapping[tuple[int, int], float]:
    return {edge: edge_energy(kernel, weights) for edge, kernel in kernels.items()}


def _panel_edges(panel, kernels: Mapping[tuple[int, int], EdgeKernel]) -> list[tuple[int, int]]:
    seam_vertices = {int(idx) for idx in panel.seam_vertices}
    return [
        edge for edge in kernels.keys() if edge[0] in seam_vertices and edge[1] in seam_vertices
    ]


def _default_partition(panel, vertices: np.ndarray) -> Partition | None:
    if not panel.seam_vertices:
        return None
    seam_indices = np.asarray(panel.seam_vertices, dtype=int)
    coords = vertices[seam_indices]
    midpoint = float(np.mean(coords[:, 0]))
    sources = tuple(int(idx) for idx in seam_indices if vertices[idx, 0] <= midpoint)
    sinks = tuple(int(idx) for idx in seam_indices if vertices[idx, 0] > midpoint)
    if not sources or not sinks:
        half = len(seam_indices) // 2
        sources = tuple(int(idx) for idx in seam_indices[:max(1, half)])
        sinks = tuple(int(idx) for idx in seam_indices[max(1, half):])
    return Partition(sources=sources, sinks=sinks)


def _build_adjacency(edges: Iterable[tuple[int, int]], costs: Mapping[tuple[int, int], float]) -> MutableMapping[int, MutableMapping[int, float]]:
    adj: MutableMapping[int, MutableMapping[int, float]] = {}
    for a, b in edges:
        w = float(costs.get((a, b), costs.get((b, a), 0.0)))
        adj.setdefault(a, {})[b] = adj.get(a, {}).get(b, 0.0) + w
        adj.setdefault(b, {})[a] = adj.get(b, {}).get(a, 0.0) + w
    return adj


def _stoer_wagner(adj: MutableMapping[int, MutableMapping[int, float]]) -> tuple[float, set[int]]:
    vertices = list(adj.keys())
    best_cut = set()
    best_weight = float("inf")
    while len(vertices) > 1:
        used = {vertices[0]}
        weights = {v: 0.0 for v in vertices}
        prev = vertices[0]
        order = [prev]
        for _ in range(len(vertices) - 1):
            for v in vertices:
                if v not in used:
                    weights[v] += adj[prev].get(v, 0.0)
            next_vertex = max((v for v in vertices if v not in used), key=lambda v: weights[v])
            used.add(next_vertex)
            order.append(next_vertex)
            prev = next_vertex
        s, t = order[-2], order[-1]
        cut_weight = sum(adj[t].get(v, 0.0) for v in adj[t] if v in used)
        if cut_weight < best_weight:
            best_weight = cut_weight
            best_cut = set(order[:-1])
        for v in adj[t]:
            if v != s:
                adj[s][v] = adj[s].get(v, 0.0) + adj[t][v]
                adj[v][s] = adj[v].get(s, 0.0) + adj[t][v]
        vertices.remove(t)
        adj.pop(t, None)
        for v in adj.values():
            v.pop(t, None)
    return best_weight, best_cut


def solve_seams_mincut(
    seam_graph: "SeamGraph",
    kernels: Mapping[tuple[int, int], EdgeKernel],
    weights: KernelWeights,
    mdl_prior: MDLPrior,
    constraints: ConstraintRegistry | None,
    *,
    cost_field: SeamCostField,
    vertices: np.ndarray,
    partitions: Mapping[str, Partition] | None = None,
    vertex_weight: float = 0.0,
) -> SeamSolution:
    """Min-cut seam solver per panel using kernel+MDL objective."""

    if SeamGraph is None:
        raise ImportError("suit.seam_generator.SeamGraph is required to run min-cut solver.")

    edge_costs = _edge_costs(kernels, weights)
    panel_solutions: MutableMapping[str, PanelSolution] = {}
    warnings: list[str] = []

    for panel in seam_graph.panels:
        if not panel.seam_vertices:
            solution = PanelSolution(
                panel=panel.name,
                edges=tuple(),
                vertices=tuple(),
                cost=0.0,
                cost_breakdown={"edge_cost": 0.0, "vertex_cost": 0.0},
                warnings=("panel has no seam vertices",),
            )
            panel_solutions[panel.name] = solution
            warnings.extend(solution.warnings)
            continue

        partition = partitions.get(panel.name) if partitions else _default_partition(panel, vertices)
        if partition is None or not partition.sources or not partition.sinks:
            panel_solution = PanelSolution(
                panel=panel.name,
                edges=tuple(),
                vertices=tuple(panel.seam_vertices),
                cost=0.0,
                cost_breakdown={"edge_cost": 0.0, "vertex_cost": 0.0},
                warnings=("partition missing",),
            )
            panel_solutions[panel.name] = panel_solution
            warnings.extend(panel_solution.warnings)
            continue

        panel_edges = _panel_edges(panel, kernels)
        if constraints:
            panel_edges = [
                edge
                for edge in panel_edges
                if not (constraints.is_vertex_forbidden(edge[0]) or constraints.is_vertex_forbidden(edge[1]))
            ]

        adj = _build_adjacency(panel_edges, edge_costs)
        if not adj:
            panel_solution = PanelSolution(
                panel=panel.name,
                edges=tuple(),
                vertices=tuple(panel.seam_vertices),
                cost=0.0,
                cost_breakdown={"edge_cost": 0.0, "vertex_cost": 0.0},
                warnings=("no edges available",),
            )
            panel_solutions[panel.name] = panel_solution
            warnings.extend(panel_solution.warnings)
            continue

        cut_weight, cut_set = _stoer_wagner(adj)
        cut_edges = [
            (min(a, b), max(a, b))
            for (a, b) in panel_edges
            if (a in cut_set and b not in cut_set) or (b in cut_set and a not in cut_set)
        ]
        vertex_cost_term = float(np.nansum(np.asarray(cost_field.vertex_costs, dtype=float)[list(panel.seam_vertices)]))
        vertex_cost_term *= float(vertex_weight)

        panel_solution = PanelSolution(
            panel=panel.name,
            edges=tuple(sorted(cut_edges)),
            vertices=tuple(panel.seam_vertices),
            cost=float(cut_weight) + vertex_cost_term,
            cost_breakdown={"edge_cost": float(cut_weight), "vertex_cost": vertex_cost_term},
            warnings=tuple(),
        )
        panel_solutions[panel.name] = panel_solution

    seam_solution = SeamSolution(
        solver="mincut",
        panel_solutions=panel_solutions,
        total_cost=0.0,
        warnings=tuple(warnings),
        metadata={"vertex_weight": vertex_weight},
    )
    mdl_value, mdl_breakdown = mdl_cost(seam_solution, mdl_prior, vertices=vertices)
    data_cost = sum(panel.cost for panel in panel_solutions.values())
    return SeamSolution(
        solver="mincut",
        panel_solutions=panel_solutions,
        total_cost=data_cost + mdl_value,
        warnings=tuple(warnings),
        metadata={
            "data_cost": data_cost,
            "mdl_cost": mdl_value,
            **mdl_breakdown,
            "vertex_weight": vertex_weight,
        },
    )
