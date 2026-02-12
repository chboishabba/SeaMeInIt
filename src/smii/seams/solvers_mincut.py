"""Min-cut seam solver sharing the kernel + MDL objective."""

from __future__ import annotations

import heapq
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


def _multi_source_dijkstra(
    adjacency: Mapping[int, Mapping[int, float]],
    sources: Sequence[int],
) -> Mapping[int, float]:
    dist: dict[int, float] = {}
    heap: list[tuple[float, int]] = []
    for source in sources:
        if source in adjacency:
            dist[int(source)] = 0.0
            heap.append((0.0, int(source)))
    heapq.heapify(heap)

    while heap:
        current_dist, node = heapq.heappop(heap)
        if current_dist > dist.get(node, float("inf")):
            continue
        for neighbor, weight in adjacency.get(node, {}).items():
            candidate = current_dist + max(float(weight), 1e-6)
            if candidate < dist.get(neighbor, float("inf")):
                dist[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return dist


def _fallback_split_by_geometry(
    vertices: np.ndarray,
    nodes: Sequence[int],
    sources: Sequence[int],
    sinks: Sequence[int],
) -> set[int]:
    if not nodes:
        return set()
    source_nodes = [int(v) for v in sources if int(v) < len(vertices)]
    sink_nodes = [int(v) for v in sinks if int(v) < len(vertices)]
    if source_nodes and sink_nodes:
        source_center = np.mean(vertices[np.asarray(source_nodes, dtype=int)], axis=0)
        sink_center = np.mean(vertices[np.asarray(sink_nodes, dtype=int)], axis=0)
        axis = np.asarray(sink_center - source_center, dtype=float)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm > 1e-8:
            axis /= axis_norm
            midpoint = (source_center + sink_center) * 0.5
            split = {
                int(node)
                for node in nodes
                if float(np.dot(vertices[int(node)] - midpoint, axis)) <= 0.0
            }
            if split and len(split) < len(nodes):
                return split

    ordered = sorted(int(node) for node in nodes)
    half = max(1, len(ordered) // 2)
    return set(ordered[:half])


def _partition_cut(
    panel_edges: Sequence[tuple[int, int]],
    edge_costs: Mapping[tuple[int, int], float],
    vertices: np.ndarray,
    partition: Partition,
) -> tuple[list[tuple[int, int]], float, str | None]:
    adjacency = _build_adjacency(panel_edges, edge_costs)
    if not adjacency:
        return [], 0.0, "no edges available"

    nodes = sorted(adjacency.keys())
    sources = tuple(int(v) for v in partition.sources if int(v) in adjacency)
    sinks = tuple(int(v) for v in partition.sinks if int(v) in adjacency)
    if not sources or not sinks:
        source_side = _fallback_split_by_geometry(vertices, nodes, partition.sources, partition.sinks)
        warning = "partition anchors missing from graph; used geometric split"
    else:
        dist_source = _multi_source_dijkstra(adjacency, sources)
        dist_sink = _multi_source_dijkstra(adjacency, sinks)
        source_side = {
            int(node)
            for node in nodes
            if dist_source.get(int(node), float("inf")) <= dist_sink.get(int(node), float("inf"))
        }
        warning = None
        if not source_side or len(source_side) == len(nodes):
            source_side = _fallback_split_by_geometry(vertices, nodes, sources, sinks)
            warning = "distance partition collapsed; used geometric split"

    cut_edges: list[tuple[int, int]] = []
    cut_weight = 0.0
    for a, b in panel_edges:
        a_side = int(a) in source_side
        b_side = int(b) in source_side
        if a_side == b_side:
            continue
        edge = (min(int(a), int(b)), max(int(a), int(b)))
        cut_edges.append(edge)
        cut_weight += float(edge_costs.get(edge, edge_costs.get((edge[1], edge[0]), 0.0)))

    if not cut_edges:
        warning = warning or "cut produced no crossing edges"
    return sorted(set(cut_edges)), cut_weight, warning


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
        panel_warnings: list[str] = []
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

        if not panel_edges:
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

        cut_edges, cut_weight, cut_warning = _partition_cut(panel_edges, edge_costs, vertices, partition)
        if cut_warning:
            panel_warnings.append(cut_warning)
        vertex_cost_term = float(np.nansum(np.asarray(cost_field.vertex_costs, dtype=float)[list(panel.seam_vertices)]))
        vertex_cost_term *= float(vertex_weight)

        panel_solution = PanelSolution(
            panel=panel.name,
            edges=tuple(sorted(cut_edges)),
            vertices=tuple(panel.seam_vertices),
            cost=float(cut_weight) + vertex_cost_term,
            cost_breakdown={"edge_cost": float(cut_weight), "vertex_cost": vertex_cost_term},
            warnings=tuple(panel_warnings),
        )
        panel_solutions[panel.name] = panel_solution
        warnings.extend(panel_solution.warnings)

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
