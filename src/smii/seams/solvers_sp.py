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


def _connected_components(
    vertices: Sequence[int],
    edges: Sequence[tuple[int, int]],
) -> list[set[int]]:
    adjacency: MutableMapping[int, set[int]] = {}
    for vertex in vertices:
        adjacency[int(vertex)] = set()
    for a, b in edges:
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))

    remaining = set(adjacency.keys())
    components: list[set[int]] = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in adjacency.get(node, ()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _component_lookup(components: Sequence[set[int]]) -> Mapping[int, int]:
    lookup: dict[int, int] = {}
    for idx, component in enumerate(components):
        for vertex in component:
            lookup[int(vertex)] = idx
    return lookup


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


def _edge_list_cost(
    edges: Sequence[tuple[int, int]],
    costs: Mapping[tuple[int, int], float],
    vertices: np.ndarray,
) -> float:
    total_cost = 0.0
    for edge in edges:
        a, b = edge
        edge_key = (min(int(a), int(b)), max(int(a), int(b)))
        base = float(costs.get(edge_key, costs.get((edge_key[1], edge_key[0]), 0.0)))
        length = max(float(np.linalg.norm(vertices[edge_key[0]] - vertices[edge_key[1]])), 1e-6)
        total_cost += base * length + length
    return total_cost


def _filter_long_edges(
    edges: Sequence[tuple[int, int]],
    vertices: np.ndarray,
    *,
    max_edge_length_factor: float,
) -> list[tuple[int, int]]:
    if not edges:
        return []
    lengths = np.asarray([np.linalg.norm(vertices[a] - vertices[b]) for a, b in edges], dtype=float)
    median = float(np.median(lengths))
    threshold = max(median * float(max_edge_length_factor), 1e-6)
    filtered = [edge for edge, length in zip(edges, lengths) if float(length) <= threshold]
    if not filtered:
        return list(edges)
    return filtered


def _loop_waypoint_pairs(
    panel,
    *,
    sample_count: int,
) -> list[AnchorPair]:
    if sample_count <= 0:
        return []
    if not hasattr(panel, "anchor_loops"):
        return []
    anchor_loops = tuple(panel.anchor_loops) if hasattr(panel, "anchor_loops") else ()
    if len(anchor_loops) != 2:
        return []
    if hasattr(panel, "loop_vertices"):
        loop_map = panel.loop_vertices()
    else:
        loop_map = {}
    if not isinstance(loop_map, Mapping):
        return []

    start = sorted({int(v) for v in loop_map.get(anchor_loops[0], ())})
    end = sorted({int(v) for v in loop_map.get(anchor_loops[1], ())})
    if not start or not end:
        return []

    count = min(int(sample_count), len(start), len(end))
    if count <= 0:
        return []
    if count == 1:
        return [AnchorPair(source=start[len(start) // 2], target=end[len(end) // 2])]

    start_idx = np.linspace(0, len(start) - 1, num=count, dtype=int)
    end_idx = np.linspace(0, len(end) - 1, num=count, dtype=int)
    pairs: list[AnchorPair] = []
    seen: set[tuple[int, int]] = set()
    for si, ei in zip(start_idx, end_idx):
        source = int(start[int(si)])
        target = int(end[int(ei)])
        key = (source, target)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(AnchorPair(source=source, target=target))
    return pairs


def _build_anchor_pairs(
    panel,
    *,
    primary: AnchorPair,
    waypoint_count: int,
) -> list[AnchorPair]:
    pairs = [primary]
    for candidate in _loop_waypoint_pairs(panel, sample_count=waypoint_count):
        if candidate not in pairs:
            pairs.append(candidate)
    return pairs


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
    loop_waypoint_count: int = 4,
    max_edge_length_factor: float = 2.25,
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

        if not panel_edges:
            panel_warnings.append("no edges available for shortest-path solve")
            vertex_cost_term = float(np.nansum(np.asarray(cost_field.vertex_costs, dtype=float)[list(panel.seam_vertices)]))
            vertex_cost_term *= float(vertex_weight)
            panel_solution = PanelSolution(
                panel=panel.name,
                edges=tuple(),
                vertices=tuple(panel.seam_vertices),
                cost=vertex_cost_term,
                cost_breakdown={"edge_cost": 0.0, "vertex_cost": vertex_cost_term},
                warnings=tuple(panel_warnings),
            )
            panel_solutions[panel.name] = panel_solution
            warnings.extend(panel_solution.warnings)
            continue

        local_edges = _filter_long_edges(
            panel_edges,
            vertices,
            max_edge_length_factor=max_edge_length_factor,
        )
        if len(local_edges) < len(panel_edges):
            panel_warnings.append(
                f"filtered long edges for pathing ({len(panel_edges)} -> {len(local_edges)})"
            )

        seam_vertices_tuple = tuple(int(v) for v in panel.seam_vertices)
        components_local = _connected_components(seam_vertices_tuple, local_edges)
        component_idx_local = _component_lookup(components_local)
        components_full = _connected_components(seam_vertices_tuple, panel_edges)
        component_idx_full = _component_lookup(components_full)
        if (
            anchor.source not in component_idx_local
            or anchor.target not in component_idx_local
            or component_idx_local[anchor.source] != component_idx_local[anchor.target]
        ):
            if (
                anchor.source in component_idx_full
                and anchor.target in component_idx_full
                and component_idx_full[anchor.source] == component_idx_full[anchor.target]
            ):
                panel_warnings.append("anchors disconnected on local graph; using unfiltered edge fallback")
            else:
                largest = max(components_local, key=len)
                if len(largest) >= 2:
                    ordered = sorted(largest)
                    anchor = AnchorPair(source=ordered[0], target=ordered[-1])
                    panel_warnings.append("anchors disconnected; using largest connected component anchors")
                else:
                    panel_warnings.append("panel graph is disconnected and has no pathable anchors")
                    vertex_cost_term = float(np.nansum(np.asarray(cost_field.vertex_costs, dtype=float)[list(panel.seam_vertices)]))
                    vertex_cost_term *= float(vertex_weight)
                    panel_solution = PanelSolution(
                        panel=panel.name,
                        edges=tuple(),
                        vertices=tuple(panel.seam_vertices),
                        cost=vertex_cost_term,
                        cost_breakdown={"edge_cost": 0.0, "vertex_cost": vertex_cost_term},
                        warnings=tuple(panel_warnings),
                    )
                    panel_solutions[panel.name] = panel_solution
                    warnings.extend(panel_solution.warnings)
                    continue

        all_path_edges: set[tuple[int, int]] = set()
        unresolved_pairs = 0
        fallback_pairs = 0
        for pair in _build_anchor_pairs(panel, primary=anchor, waypoint_count=loop_waypoint_count):
            edge_set_for_pair = local_edges
            if (
                pair.source in component_idx_local
                and pair.target in component_idx_local
                and component_idx_local[pair.source] == component_idx_local[pair.target]
            ):
                edge_set_for_pair = local_edges
            elif (
                pair.source in component_idx_full
                and pair.target in component_idx_full
                and component_idx_full[pair.source] == component_idx_full[pair.target]
            ):
                edge_set_for_pair = panel_edges
                fallback_pairs += 1
            else:
                unresolved_pairs += 1
                continue
            path_edges, edge_cost = _dijkstra_path(edge_set_for_pair, edge_costs, pair.source, pair.target, vertices)
            if not path_edges or edge_cost == float("inf"):
                unresolved_pairs += 1
                continue
            for edge in path_edges:
                all_path_edges.add((min(edge[0], edge[1]), max(edge[0], edge[1])))

        if not all_path_edges:
            panel_warnings.append("no path between anchors")
        if unresolved_pairs:
            panel_warnings.append(f"{unresolved_pairs} waypoint paths unresolved")
        if fallback_pairs:
            panel_warnings.append(f"{fallback_pairs} waypoint paths used unfiltered edges")

        selected_path_edges = tuple(sorted(all_path_edges))
        edge_cost = _edge_list_cost(selected_path_edges, edge_costs, vertices)
        vertex_cost_term = float(np.nansum(np.asarray(cost_field.vertex_costs, dtype=float)[list(panel.seam_vertices)]))
        vertex_cost_term *= float(vertex_weight)

        panel_solution = PanelSolution(
            panel=panel.name,
            edges=selected_path_edges,
            vertices=tuple(panel.seam_vertices),
            cost=edge_cost + vertex_cost_term,
            cost_breakdown={"edge_cost": edge_cost, "vertex_cost": vertex_cost_term},
            warnings=tuple(panel_warnings),
        )
        panel_solutions[panel.name] = panel_solution
        warnings.extend(panel_solution.warnings)

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
