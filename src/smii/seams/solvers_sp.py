"""Shortest-path seam solver sharing the kernel + MDL objective."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

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

__all__ = ["solve_seams_shortest_path"]


@dataclass(frozen=True, slots=True)
class AnchorPair:
    """Anchor vertices for a panel."""

    source: int
    target: int


def _edge_costs(kernels: Mapping[tuple[int, int], EdgeKernel], weights: KernelWeights) -> Mapping[tuple[int, int], float]:
    return {edge: edge_energy(kernel, weights) for edge, kernel in kernels.items()}


def _panel_edges(panel, kernels: Mapping[tuple[int, int], EdgeKernel]) -> list[tuple[int, int]]:
    seam_vertices = {int(idx) for idx in panel.seam_vertices}
    return [
        edge for edge in kernels.keys() if edge[0] in seam_vertices and edge[1] in seam_vertices
    ]


def _default_anchor(panel) -> AnchorPair | None:
    if not panel.seam_vertices:
        return None
    seam_vertices = sorted(int(v) for v in panel.seam_vertices)
    return AnchorPair(source=seam_vertices[0], target=seam_vertices[-1])


def _dijkstra_path(
    edges: Sequence[tuple[int, int]],
    costs: Mapping[tuple[int, int], float],
    source: int,
    target: int,
    vertices: np.ndarray,
) -> tuple[list[tuple[int, int]], float]:
    graph: MutableMapping[int, list[tuple[int, float]]] = {}
    for a, b in edges:
        base = float(costs.get((a, b), costs.get((b, a), 0.0)))
        length = float(np.linalg.norm(vertices[a] - vertices[b]))
        length = max(length, 1e-6)
        w = base * length + length
        graph.setdefault(a, []).append((b, w))
        graph.setdefault(b, []).append((a, w))

    dist = {source: 0.0}
    prev: MutableMapping[int, int | None] = {source: None}
    heap: list[tuple[float, int]] = [(0.0, source)]

    while heap:
        current_dist, node = heapq.heappop(heap)
        if node == target:
            break
        if current_dist > dist.get(node, float("inf")):
            continue
        for neighbor, weight in graph.get(node, []):
            new_dist = current_dist + weight
            if new_dist <= dist.get(neighbor, float("inf")):
                dist[neighbor] = new_dist
                prev[neighbor] = node
                heapq.heappush(heap, (new_dist, neighbor))

    if target not in prev and target != source:
        return ([], float("inf"))

    path_vertices = [target]
    while path_vertices[-1] != source:
        parent = prev[path_vertices[-1]]
        if parent is None:
            break
        path_vertices.append(parent)
    path_vertices.reverse()

    path_edges: list[tuple[int, int]] = []
    for idx in range(len(path_vertices) - 1):
        a, b = path_vertices[idx], path_vertices[idx + 1]
        path_edges.append((min(a, b), max(a, b)))
    total_cost = 0.0
    for edge in path_edges:
        base = float(costs.get(edge, costs.get((edge[1], edge[0]), 0.0)))
        length = max(float(np.linalg.norm(vertices[edge[0]] - vertices[edge[1]])), 1e-6)
        total_cost += base * length + length
    return path_edges, total_cost


def solve_seams_shortest_path(
    seam_graph: "SeamGraph",
    kernels: Mapping[tuple[int, int], EdgeKernel],
    weights: KernelWeights,
    mdl_prior: MDLPrior,
    constraints: ConstraintRegistry | None,
    *,
    cost_field: SeamCostField,
    vertices: np.ndarray,
    anchors: Mapping[str, AnchorPair] | None = None,
    vertex_weight: float = 0.0,
) -> SeamSolution:
    """Shortest-path seam solver per panel using kernel+MDL objective."""

    if SeamGraph is None:
        raise ImportError("suit.seam_generator.SeamGraph is required to run shortest-path solver.")

    edge_costs = _edge_costs(kernels, weights)
    panel_solutions: MutableMapping[str, PanelSolution] = {}
    warnings: list[str] = []

    for panel in seam_graph.panels:
        panel_warnings: list[str] = []

        if not panel.seam_vertices:
            panel_warnings.append("panel has no seam vertices")
            solution = PanelSolution(
                panel=panel.name,
                edges=tuple(),
                vertices=tuple(),
                cost=0.0,
                cost_breakdown={"edge_cost": 0.0, "vertex_cost": 0.0},
                warnings=tuple(panel_warnings),
            )
            panel_solutions[panel.name] = solution
            warnings.extend(solution.warnings)
            continue

        anchor = anchors.get(panel.name) if anchors else _default_anchor(panel)
        if anchor is None:
            panel_warnings.append("no anchors provided")
            solution = PanelSolution(
                panel=panel.name,
                edges=tuple(),
                vertices=tuple(panel.seam_vertices),
                cost=0.0,
                cost_breakdown={"edge_cost": 0.0, "vertex_cost": 0.0},
                warnings=tuple(panel_warnings),
            )
            panel_solutions[panel.name] = solution
            warnings.extend(solution.warnings)
            continue

        panel_edges = _panel_edges(panel, kernels)
        if constraints:
            allowed_edges = []
            for edge in panel_edges:
                if constraints.is_vertex_forbidden(edge[0]) or constraints.is_vertex_forbidden(edge[1]):
                    continue
                allowed_edges.append(edge)
            panel_edges = allowed_edges

        path_edges, edge_cost = _dijkstra_path(panel_edges, edge_costs, anchor.source, anchor.target, vertices)
        if not path_edges or edge_cost == float("inf"):
            panel_warnings.append("no path between anchors")
        vertex_cost_term = float(np.nansum(np.asarray(cost_field.vertex_costs, dtype=float)[list(panel.seam_vertices)]))
        vertex_cost_term *= float(vertex_weight)

        panel_solution = PanelSolution(
            panel=panel.name,
            edges=tuple(path_edges),
            vertices=tuple(panel.seam_vertices),
            cost=edge_cost + vertex_cost_term,
            cost_breakdown={"edge_cost": edge_cost, "vertex_cost": vertex_cost_term},
            warnings=tuple(panel_warnings),
        )
        panel_solutions[panel.name] = panel_solution

    seam_solution = SeamSolution(
        solver="shortest_path",
        panel_solutions=panel_solutions,
        total_cost=0.0,
        warnings=tuple(warnings),
        metadata={"vertex_weight": vertex_weight},
    )
    mdl_value, mdl_breakdown = mdl_cost(seam_solution, mdl_prior, vertices=vertices)
    data_cost = sum(panel.cost for panel in panel_solutions.values())
    return SeamSolution(
        solver="shortest_path",
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
