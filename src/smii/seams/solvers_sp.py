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


def _edge_costs(
    kernels: Mapping[tuple[int, int], EdgeKernel], weights: KernelWeights
) -> Mapping[tuple[int, int], float]:
    return {edge: edge_energy(kernel, weights) for edge, kernel in kernels.items()}


def _normalize_edge(edge: tuple[int, int]) -> tuple[int, int]:
    return (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1])))


def _panel_edges(panel, kernels: Mapping[tuple[int, int], EdgeKernel]) -> list[tuple[int, int]]:
    seam_vertices = {int(idx) for idx in panel.seam_vertices}
    return [
        edge for edge in kernels.keys() if edge[0] in seam_vertices and edge[1] in seam_vertices
    ]


def _default_anchor(panel, vertices: np.ndarray | None = None) -> AnchorPair | None:
    if not panel.seam_vertices:
        return None
    if hasattr(panel, "anchor_loops") and hasattr(panel, "loop_vertices"):
        loops = tuple(panel.anchor_loops)
        loop_map = panel.loop_vertices()
        if len(loops) == 2 and isinstance(loop_map, Mapping):
            src_candidates = sorted({int(v) for v in loop_map.get(loops[0], ())})
            dst_candidates = sorted({int(v) for v in loop_map.get(loops[1], ())})
            if src_candidates and dst_candidates:
                if vertices is not None:
                    src_valid = [v for v in src_candidates if 0 <= v < len(vertices)]
                    dst_valid = [v for v in dst_candidates if 0 <= v < len(vertices)]
                    if src_valid and dst_valid:
                        src_pts = vertices[np.asarray(src_valid, dtype=int)]
                        dst_pts = vertices[np.asarray(dst_valid, dtype=int)]
                        src_centroid = np.mean(src_pts, axis=0)
                        dst_centroid = np.mean(dst_pts, axis=0)
                        source = min(
                            src_valid,
                            key=lambda idx: float(np.linalg.norm(vertices[idx] - src_centroid)),
                        )
                        target = min(
                            dst_valid,
                            key=lambda idx: float(np.linalg.norm(vertices[idx] - dst_centroid)),
                        )
                        return AnchorPair(source=int(source), target=int(target))
                return AnchorPair(
                    source=int(src_candidates[len(src_candidates) // 2]),
                    target=int(dst_candidates[len(dst_candidates) // 2]),
                )
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


def _edge_length_sum(edges: Sequence[tuple[int, int]], vertices: np.ndarray) -> float:
    total = 0.0
    for a, b in edges:
        total += float(np.linalg.norm(vertices[int(a)] - vertices[int(b)]))
    return total


def _is_simple_cycle(edges: Sequence[tuple[int, int]]) -> bool:
    if not edges:
        return False
    adjacency: MutableMapping[int, set[int]] = {}
    for a, b in edges:
        ai, bi = int(a), int(b)
        adjacency.setdefault(ai, set()).add(bi)
        adjacency.setdefault(bi, set()).add(ai)
    if not adjacency:
        return False
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return False

    start = next(iter(adjacency))
    stack = [start]
    visited: set[int] = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(neighbor for neighbor in adjacency[node] if neighbor not in visited)
    return len(visited) == len(adjacency)


def _is_union_of_simple_cycles(edges: Sequence[tuple[int, int]]) -> bool:
    """Return True when every connected component is a simple cycle."""

    if not edges:
        return False
    adjacency: MutableMapping[int, set[int]] = {}
    for a, b in edges:
        ai, bi = int(a), int(b)
        adjacency.setdefault(ai, set()).add(bi)
        adjacency.setdefault(bi, set()).add(ai)
    if not adjacency:
        return False
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return False

    visited: set[int] = set()
    for start in adjacency:
        if start in visited:
            continue
        stack = [start]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        if len(component) < 3:
            return False
        visited.update(component)
    return len(visited) == len(adjacency)


def _symmetry_penalty(
    edges: Sequence[tuple[int, int]],
    constraints: ConstraintRegistry | None,
    penalty_weight: float,
) -> float:
    if constraints is None or penalty_weight <= 0.0:
        return 0.0
    edge_set = {_normalize_edge(edge) for edge in edges}
    penalty = 0.0
    for edge in edge_set:
        partner = _normalize_edge(
            (
                constraints.symmetry_partner(edge[0]) or edge[0],
                constraints.symmetry_partner(edge[1]) or edge[1],
            )
        )
        if partner not in edge_set:
            penalty += float(penalty_weight)
    return penalty


def _build_loop_edges(
    edges: Sequence[tuple[int, int]],
    costs: Mapping[tuple[int, int], float],
    *,
    source: int,
    target: int,
    vertices: np.ndarray,
) -> tuple[tuple[tuple[int, int], ...], tuple[str, ...], bool]:
    """Build loop candidate from two distinct source-target paths."""

    first_path, _ = _dijkstra_path(edges, costs, source, target, vertices)
    if not first_path:
        return tuple(), ("no path between anchors",), False

    first_set = {_normalize_edge(edge) for edge in first_path}
    residual = [edge for edge in edges if _normalize_edge(edge) not in first_set]
    if not residual:
        return (
            tuple(sorted(first_set)),
            ("loop closure unavailable (single-route panel graph)",),
            False,
        )

    second_path, _ = _dijkstra_path(residual, costs, source, target, vertices)
    if not second_path:
        return (
            tuple(sorted(first_set)),
            ("loop closure unavailable (no alternate return path)",),
            False,
        )

    cycle_set = first_set | {_normalize_edge(edge) for edge in second_path}
    warnings: list[str] = []
    is_simple = _is_simple_cycle(tuple(cycle_set))
    if not is_simple:
        warnings.append("loop closure produced non-simple cycle")
    return tuple(sorted(cycle_set)), tuple(warnings), is_simple


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
    vertices: np.ndarray,
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

    start = sorted({int(v) for v in loop_map.get(anchor_loops[0], ()) if int(v) < len(vertices)})
    end = sorted({int(v) for v in loop_map.get(anchor_loops[1], ()) if int(v) < len(vertices)})
    if not start or not end:
        return []

    count = min(int(sample_count), len(start), len(end))
    if count <= 0:
        return []
    if count == 1:
        source = start[len(start) // 2]
        target = min(
            end, key=lambda idx: float(np.linalg.norm(vertices[int(idx)] - vertices[int(source)]))
        )
        return [AnchorPair(source=int(source), target=int(target))]

    # Farthest-point sample on source loop to spread waypoints geometrically.
    start_coords = vertices[np.asarray(start, dtype=int)]
    selected_positions: list[int] = [int(np.argmin(start_coords[:, 0]))]
    while len(selected_positions) < count:
        chosen = start_coords[np.asarray(selected_positions, dtype=int)]
        # Distance to nearest chosen point for each candidate.
        dists = np.linalg.norm(start_coords[:, None, :] - chosen[None, :, :], axis=2)
        score = np.min(dists, axis=1)
        next_idx = int(np.argmax(score))
        if next_idx in selected_positions:
            break
        selected_positions.append(next_idx)

    selected_positions = selected_positions[:count]
    selected_sources = [int(start[pos]) for pos in selected_positions]

    # Match each source to nearest unused target for local, non-crossing pairing.
    remaining_targets = set(int(v) for v in end)
    pairs: list[AnchorPair] = []
    for source in selected_sources:
        if not remaining_targets:
            break
        source_coord = vertices[int(source)]
        target = min(
            remaining_targets,
            key=lambda idx: float(np.linalg.norm(vertices[int(idx)] - source_coord)),
        )
        remaining_targets.remove(int(target))
        pairs.append(AnchorPair(source=int(source), target=int(target)))

    return pairs


def _build_anchor_pairs(
    panel,
    *,
    vertices: np.ndarray,
    primary: AnchorPair,
    waypoint_count: int,
) -> list[AnchorPair]:
    pairs = [primary]
    for candidate in _loop_waypoint_pairs(panel, vertices=vertices, sample_count=waypoint_count):
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
    loop_waypoint_count: int = 0,
    max_edge_length_factor: float = 1.8,
    require_loop: bool = False,
    strict_loop: bool = False,
    loop_count: int = 1,
    allow_unfiltered_fallback: bool = True,
    symmetry_penalty_weight: float = 0.0,
    reference_vertices: np.ndarray | None = None,
) -> SeamSolution:
    """Shortest-path seam solver per panel using kernel+MDL objective."""

    if SeamGraph is None:
        raise ImportError("suit.seam_generator.SeamGraph is required to run shortest-path solver.")

    edge_costs = _edge_costs(kernels, weights)
    panel_solutions: MutableMapping[str, PanelSolution] = {}
    warnings: list[str] = []

    reference_vertices_arr = None
    if reference_vertices is not None:
        ref = np.asarray(reference_vertices, dtype=float)
        if ref.shape == np.asarray(vertices, dtype=float).shape:
            reference_vertices_arr = ref
        else:
            warnings.append("reference_vertices shape mismatch; ignoring reference diagnostics")

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

        anchor = anchors.get(panel.name) if anchors else _default_anchor(panel, vertices=vertices)
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
                if constraints.is_vertex_forbidden(edge[0]) or constraints.is_vertex_forbidden(
                    edge[1]
                ):
                    continue
                allowed_edges.append(edge)
            panel_edges = allowed_edges

        if not panel_edges:
            panel_warnings.append("no edges available for shortest-path solve")
            vertex_cost_term = float(
                np.nansum(
                    np.asarray(cost_field.vertex_costs, dtype=float)[list(panel.seam_vertices)]
                )
            )
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
        local_anchor_connected = (
            anchor.source not in component_idx_local
            or anchor.target not in component_idx_local
            or component_idx_local[anchor.source] != component_idx_local[anchor.target]
        )
        if local_anchor_connected:
            if allow_unfiltered_fallback and (
                anchor.source in component_idx_full
                and anchor.target in component_idx_full
                and component_idx_full[anchor.source] == component_idx_full[anchor.target]
            ):
                panel_warnings.append(
                    "anchors disconnected on local graph; using unfiltered edge fallback"
                )
            else:
                largest = max(components_local, key=len)
                if len(largest) >= 2:
                    ordered = sorted(largest)
                    anchor = AnchorPair(source=ordered[0], target=ordered[-1])
                    panel_warnings.append(
                        "anchors disconnected; using largest connected component anchors"
                    )
                else:
                    panel_warnings.append("panel graph is disconnected and has no pathable anchors")
                    vertex_cost_term = float(
                        np.nansum(
                            np.asarray(cost_field.vertex_costs, dtype=float)[
                                list(panel.seam_vertices)
                            ]
                        )
                    )
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

        selected_path_edges: tuple[tuple[int, int], ...] = tuple()
        unresolved_pairs = 0
        fallback_pairs = 0
        if require_loop:
            requested_loop_count = max(1, int(loop_count))
            loop_waypoints = max(int(loop_waypoint_count), requested_loop_count - 1)
            pairs = _build_anchor_pairs(
                panel,
                vertices=vertices,
                primary=anchor,
                waypoint_count=loop_waypoints,
            )
            simple_loop_candidates: list[tuple[tuple[tuple[int, int], ...], float, set[int]]] = []
            fallback_loop: tuple[tuple[int, int], ...] = tuple()
            loop_attempt_warnings: list[str] = []
            for pair in pairs:
                edge_set_for_pair = local_edges
                if (
                    pair.source in component_idx_local
                    and pair.target in component_idx_local
                    and component_idx_local[pair.source] == component_idx_local[pair.target]
                ):
                    edge_set_for_pair = local_edges
                elif (
                    allow_unfiltered_fallback
                    and pair.source in component_idx_full
                    and pair.target in component_idx_full
                    and component_idx_full[pair.source] == component_idx_full[pair.target]
                ):
                    edge_set_for_pair = panel_edges
                    fallback_pairs += 1
                else:
                    unresolved_pairs += 1
                    continue

                loop_edges, loop_warnings, is_simple = _build_loop_edges(
                    edge_set_for_pair,
                    edge_costs,
                    source=pair.source,
                    target=pair.target,
                    vertices=vertices,
                )
                loop_attempt_warnings.extend(loop_warnings)
                normalized_loop = tuple(sorted({_normalize_edge(edge) for edge in loop_edges}))
                if not fallback_loop and normalized_loop:
                    fallback_loop = normalized_loop
                if not normalized_loop or not is_simple:
                    continue
                loop_vertices = {int(v) for edge in normalized_loop for v in edge}
                loop_cost = _edge_list_cost(normalized_loop, edge_costs, vertices)
                simple_loop_candidates.append((normalized_loop, loop_cost, loop_vertices))

            # Greedy choose lowest-cost disjoint simple cycles.
            selected_loop_cycles: list[tuple[tuple[int, int], ...]] = []
            used_vertices: set[int] = set()
            for cycle_edges, _, cycle_vertices in sorted(
                simple_loop_candidates, key=lambda item: item[1]
            ):
                if cycle_vertices & used_vertices:
                    continue
                selected_loop_cycles.append(cycle_edges)
                used_vertices.update(cycle_vertices)
                if len(selected_loop_cycles) >= requested_loop_count:
                    break

            if selected_loop_cycles:
                selected_edge_set: set[tuple[int, int]] = set()
                for cycle_edges in selected_loop_cycles:
                    selected_edge_set.update(cycle_edges)
                selected_path_edges = tuple(sorted(selected_edge_set))
            elif not strict_loop and fallback_loop:
                selected_path_edges = fallback_loop
                panel_warnings.extend(loop_attempt_warnings)
            else:
                selected_path_edges = tuple()
                panel_warnings.extend(loop_attempt_warnings)

            if selected_loop_cycles and len(selected_loop_cycles) < requested_loop_count:
                panel_warnings.append(
                    f"requested {requested_loop_count} loops but only {len(selected_loop_cycles)} disjoint simple loops found"
                )
        else:
            all_path_edges: set[tuple[int, int]] = set()
            for pair in _build_anchor_pairs(
                panel, vertices=vertices, primary=anchor, waypoint_count=loop_waypoint_count
            ):
                edge_set_for_pair = local_edges
                if (
                    pair.source in component_idx_local
                    and pair.target in component_idx_local
                    and component_idx_local[pair.source] == component_idx_local[pair.target]
                ):
                    edge_set_for_pair = local_edges
                elif (
                    allow_unfiltered_fallback
                    and pair.source in component_idx_full
                    and pair.target in component_idx_full
                    and component_idx_full[pair.source] == component_idx_full[pair.target]
                ):
                    edge_set_for_pair = panel_edges
                    fallback_pairs += 1
                else:
                    unresolved_pairs += 1
                    continue
                path_edges, edge_cost = _dijkstra_path(
                    edge_set_for_pair, edge_costs, pair.source, pair.target, vertices
                )
                if not path_edges or edge_cost == float("inf"):
                    unresolved_pairs += 1
                    continue
                for edge in path_edges:
                    all_path_edges.add(_normalize_edge(edge))
            selected_path_edges = tuple(sorted(all_path_edges))

        if not selected_path_edges:
            panel_warnings.append("no path between anchors")
        if unresolved_pairs:
            panel_warnings.append(f"{unresolved_pairs} waypoint paths unresolved")
        if fallback_pairs:
            panel_warnings.append(f"{fallback_pairs} waypoint paths used unfiltered edges")
        if require_loop and selected_path_edges:
            if loop_count > 1:
                if not _is_union_of_simple_cycles(selected_path_edges):
                    panel_warnings.append(
                        "require_loop enabled but selected seam is not a union of simple cycles"
                    )
                    if strict_loop:
                        selected_path_edges = tuple()
                        panel_warnings.append(
                            "strict_loop enabled; dropped non-cycle loop solution"
                        )
            elif not _is_simple_cycle(selected_path_edges):
                panel_warnings.append(
                    "require_loop enabled but selected seam is not a simple cycle"
                )
                if strict_loop:
                    selected_path_edges = tuple()
                    panel_warnings.append("strict_loop enabled; dropped non-simple loop solution")

        edge_cost = _edge_list_cost(selected_path_edges, edge_costs, vertices)
        vertex_cost_term = float(
            np.nansum(np.asarray(cost_field.vertex_costs, dtype=float)[list(panel.seam_vertices)])
        )
        vertex_cost_term *= float(vertex_weight)
        symmetry_penalty_term = _symmetry_penalty(
            selected_path_edges, constraints, symmetry_penalty_weight
        )

        cost_breakdown = {
            "edge_cost": edge_cost,
            "vertex_cost": vertex_cost_term,
            "symmetry_penalty": symmetry_penalty_term,
        }
        if reference_vertices_arr is not None:
            cost_breakdown["reference_edge_length"] = _edge_length_sum(
                selected_path_edges, reference_vertices_arr
            )

        panel_solution = PanelSolution(
            panel=panel.name,
            edges=selected_path_edges,
            vertices=tuple(panel.seam_vertices),
            cost=edge_cost + vertex_cost_term + symmetry_penalty_term,
            cost_breakdown=cost_breakdown,
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
            "require_loop": float(require_loop),
            "strict_loop": float(strict_loop),
            "loop_count": float(max(1, int(loop_count))),
            "allow_unfiltered_fallback": float(allow_unfiltered_fallback),
            "symmetry_penalty_weight": float(symmetry_penalty_weight),
            "reference_vertices": float(reference_vertices_arr is not None),
        },
    )
