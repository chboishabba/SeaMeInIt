#!/usr/bin/env python3
"""Solve promoted seam topology from a receipted seam-cost artifact."""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from smii.rom import load_seam_cost_field
from smii.seams import (
    CORRECTION_FAMILIES,
    MetricEnergyWeights,
    SolverPromotionReceipt,
    build_metric_panelization_payload,
    can_consume_seam_cost_receipt,
    load_seam_cost_receipt,
    normalize_families,
)


Edge = tuple[int, int]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if "vertices" not in payload or "faces" not in payload:
        raise KeyError("Mesh NPZ must contain 'vertices' and 'faces' arrays.")
    vertices = np.asarray(payload["vertices"], dtype=float)
    faces = np.asarray(payload["faces"], dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Mesh vertices must be shaped (N, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Mesh faces must be shaped (M, 3).")
    return vertices, faces


def _normalize_edge(a: int, b: int) -> Edge:
    aa = int(a)
    bb = int(b)
    return (aa, bb) if aa <= bb else (bb, aa)


def _mesh_edges(faces: np.ndarray) -> tuple[Edge, ...]:
    edges: set[Edge] = set()
    for a, b, c in np.asarray(faces, dtype=int):
        for u, v in ((a, b), (b, c), (c, a)):
            edge = _normalize_edge(int(u), int(v))
            if edge[0] != edge[1]:
                edges.add(edge)
    return tuple(sorted(edges))


def _face_edges(face: Sequence[int]) -> tuple[Edge, Edge, Edge]:
    a, b, c = (int(face[0]), int(face[1]), int(face[2]))
    return (_normalize_edge(a, b), _normalize_edge(b, c), _normalize_edge(c, a))


def _adjacency(edges: Sequence[Edge]) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        graph[int(a)].add(int(b))
        graph[int(b)].add(int(a))
    return graph


def _vertex_costs_from_edges(
    vertex_count: int,
    edges: Sequence[Edge],
    edge_costs: np.ndarray,
    vertex_costs: np.ndarray,
) -> np.ndarray:
    vertex_arr = np.asarray(vertex_costs, dtype=float)
    if vertex_arr.shape == (vertex_count,) and np.isfinite(vertex_arr).any():
        finite_max = np.nanmax(vertex_arr[np.isfinite(vertex_arr)])
        finite = np.where(np.isfinite(vertex_arr), vertex_arr, finite_max)
        return np.asarray(finite, dtype=float)

    totals = np.zeros(vertex_count, dtype=float)
    counts = np.zeros(vertex_count, dtype=float)
    for idx, (a, b) in enumerate(edges):
        cost = float(edge_costs[idx])
        if not np.isfinite(cost):
            continue
        totals[a] += cost
        totals[b] += cost
        counts[a] += 1.0
        counts[b] += 1.0
    return np.divide(totals, np.maximum(counts, 1.0))


def _select_field_minima_anchors(
    vertices: np.ndarray,
    vertex_costs: np.ndarray,
    *,
    anchor_count: int,
    min_separation_ratio: float,
) -> list[int]:
    if len(vertices) == 0:
        return []
    requested = max(1, min(int(anchor_count), len(vertices)))
    bbox_extent = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    min_separation = max(0.0, float(min_separation_ratio)) * bbox_extent
    ordered = sorted(
        range(len(vertices)),
        key=lambda idx: (float(vertex_costs[int(idx)]), int(idx)),
    )
    selected: list[int] = []
    for idx in ordered:
        if len(selected) >= requested:
            break
        if all(
            float(np.linalg.norm(vertices[idx] - vertices[prev])) >= min_separation
            for prev in selected
        ):
            selected.append(int(idx))
    for idx in ordered:
        if len(selected) >= requested:
            break
        if int(idx) not in selected:
            selected.append(int(idx))
    return selected


def _select_geometric_anchors(vertices: np.ndarray, *, anchor_count: int) -> list[int]:
    if len(vertices) == 0:
        return []
    candidates: list[int] = []
    for axis in range(3):
        candidates.append(int(np.argmin(vertices[:, axis])))
        candidates.append(int(np.argmax(vertices[:, axis])))
    unique = []
    for idx in candidates:
        if idx not in unique:
            unique.append(idx)
    return unique[: max(1, min(int(anchor_count), len(unique)))]


def _parse_manual_anchors(value: str | None, vertex_count: int) -> list[int]:
    if value is None or not value.strip():
        raise ValueError("--manual-anchors is required when --anchor-source=manual.")
    anchors = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not anchors:
        raise ValueError("--manual-anchors must contain at least one vertex index.")
    invalid = [idx for idx in anchors if idx < 0 or idx >= vertex_count]
    if invalid:
        raise ValueError(f"Manual anchors out of range: {invalid}")
    unique = []
    for idx in anchors:
        if idx not in unique:
            unique.append(idx)
    return unique


def _edge_cost_lookup(edges: Sequence[Edge], edge_costs: np.ndarray) -> dict[Edge, float]:
    return {edge: float(edge_costs[idx]) for idx, edge in enumerate(edges)}


def _face_topology(
    faces: np.ndarray,
) -> tuple[dict[Edge, list[int]], dict[int, list[tuple[int, Edge]]]]:
    edge_to_faces: dict[Edge, list[int]] = defaultdict(list)
    for face_idx, face in enumerate(np.asarray(faces, dtype=int)):
        for edge in _face_edges(face):
            edge_to_faces[edge].append(int(face_idx))

    face_graph: dict[int, list[tuple[int, Edge]]] = defaultdict(list)
    for edge, face_indices in edge_to_faces.items():
        if len(face_indices) != 2:
            continue
        a, b = int(face_indices[0]), int(face_indices[1])
        face_graph[a].append((b, edge))
        face_graph[b].append((a, edge))
    return edge_to_faces, face_graph


def _load_dart_candidate_edges(
    path: Path | None,
    *,
    max_candidates: int,
) -> tuple[set[Edge], str | None]:
    if path is None:
        return set(), None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_candidates = payload.get("candidates", []) if isinstance(payload, Mapping) else []
        if not isinstance(raw_candidates, list):
            return set(), "invalid_dart_relief_candidates"
        candidate_edges: set[Edge] = set()
        for candidate in raw_candidates[: max(0, int(max_candidates))]:
            if not isinstance(candidate, Mapping):
                continue
            raw_edges = candidate.get("path_edges", [])
            if not isinstance(raw_edges, list):
                continue
            for raw_edge in raw_edges:
                if (
                    isinstance(raw_edge, list)
                    and len(raw_edge) == 2
                    and not isinstance(raw_edge[0], bool)
                    and not isinstance(raw_edge[1], bool)
                ):
                    candidate_edges.add(_normalize_edge(int(raw_edge[0]), int(raw_edge[1])))
        return candidate_edges, None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return set(), "invalid_dart_relief_candidates"


def _face_costs(
    faces: np.ndarray,
    edge_cost_lookup: Mapping[Edge, float],
    vertex_costs: np.ndarray,
) -> np.ndarray:
    costs = np.zeros(int(faces.shape[0]), dtype=float)
    for face_idx, face in enumerate(np.asarray(faces, dtype=int)):
        edge_mean = float(np.mean([edge_cost_lookup.get(edge, 0.0) for edge in _face_edges(face)]))
        vertex_mean = float(np.mean([vertex_costs[int(vertex)] for vertex in face]))
        costs[int(face_idx)] = edge_mean + vertex_mean
    return costs


def _face_centroids(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.asarray([np.mean(vertices[np.asarray(face, dtype=int)], axis=0) for face in faces])


def _select_seed_faces(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    anchors: Sequence[int],
    face_costs: np.ndarray,
    target_panel_count: int,
) -> list[int]:
    if int(faces.shape[0]) == 0:
        return []
    vertex_to_faces: dict[int, list[int]] = defaultdict(list)
    for face_idx, face in enumerate(np.asarray(faces, dtype=int)):
        for vertex in face:
            vertex_to_faces[int(vertex)].append(int(face_idx))

    candidate_faces: list[int] = []
    for anchor in anchors:
        attached = vertex_to_faces.get(int(anchor), [])
        if attached:
            candidate_faces.append(
                min(attached, key=lambda idx: (float(face_costs[int(idx)]), int(idx)))
            )
    for idx in np.argsort(face_costs, kind="stable"):
        candidate_faces.append(int(idx))

    centroids = _face_centroids(vertices, faces)
    bbox_extent = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-9)
    seeds: list[int] = []
    for face_idx in candidate_faces:
        if face_idx in seeds:
            continue
        if seeds and len(seeds) < target_panel_count:
            min_dist = min(
                float(np.linalg.norm(centroids[face_idx] - centroids[seed])) for seed in seeds
            )
            if min_dist < bbox_extent / max(2.0, float(target_panel_count)):
                continue
        seeds.append(int(face_idx))
        if len(seeds) >= target_panel_count:
            break
    for idx in range(int(faces.shape[0])):
        if len(seeds) >= target_panel_count:
            break
        if idx not in seeds:
            seeds.append(idx)
    return seeds


def _assign_face_regions(
    *,
    face_count: int,
    seeds: Sequence[int],
    face_graph: Mapping[int, Sequence[tuple[int, Edge]]],
    edge_cost_lookup: Mapping[Edge, float],
    dart_edges: set[Edge],
) -> np.ndarray:
    labels = np.full(int(face_count), -1, dtype=int)
    heap: list[tuple[float, int, int]] = []
    for label, seed in enumerate(seeds):
        labels[int(seed)] = int(label)
        heapq.heappush(heap, (0.0, int(label), int(seed)))
    while heap:
        distance, label, face_idx = heapq.heappop(heap)
        if labels[int(face_idx)] != int(label):
            continue
        for neighbor, edge in sorted(face_graph.get(int(face_idx), ()), key=lambda item: item[0]):
            if labels[int(neighbor)] != -1:
                continue
            base_cost = max(float(edge_cost_lookup.get(edge, 1.0)), 1e-9)
            # Lower crossing cost makes dart-advisory edges attractive as region boundaries.
            next_distance = distance + (base_cost * (0.2 if edge in dart_edges else 1.0))
            labels[int(neighbor)] = int(label)
            heapq.heappush(heap, (next_distance, int(label), int(neighbor)))
    for face_idx in range(int(face_count)):
        if labels[face_idx] == -1:
            labels[face_idx] = int(len(seeds))
    return labels


def _connected_face_region_count(
    labels: np.ndarray,
    face_graph: Mapping[int, Sequence[tuple[int, Edge]]],
) -> int:
    remaining = set(range(int(labels.shape[0])))
    component_count = 0
    while remaining:
        component_count += 1
        start = min(remaining)
        remaining.remove(start)
        label = int(labels[start])
        queue: deque[int] = deque([start])
        while queue:
            face_idx = queue.popleft()
            for neighbor, _edge in face_graph.get(face_idx, ()):
                if neighbor in remaining and int(labels[int(neighbor)]) == label:
                    remaining.remove(int(neighbor))
                    queue.append(int(neighbor))
    return component_count


def _cut_graph_from_face_regions(
    *,
    faces: np.ndarray,
    edge_to_faces: Mapping[Edge, Sequence[int]],
    labels: np.ndarray,
    edge_cost_lookup: Mapping[Edge, float],
) -> tuple[tuple[Edge, ...], float, int, list[int], list[int]]:
    seam_edges: set[Edge] = set()
    for edge, face_indices in edge_to_faces.items():
        if len(face_indices) != 2:
            continue
        a, b = int(face_indices[0]), int(face_indices[1])
        if int(labels[a]) != int(labels[b]):
            seam_edges.add(edge)

    panel_labels = sorted({int(label) for label in labels})
    face_counts = [int(np.sum(labels == label)) for label in panel_labels]
    boundary_counts = []
    for label in panel_labels:
        boundary_count = 0
        region_faces = {idx for idx, value in enumerate(labels) if int(value) == label}
        for edge, face_indices in edge_to_faces.items():
            inside_count = sum(1 for idx in face_indices if int(idx) in region_faces)
            if inside_count == 1:
                boundary_count += 1
        boundary_counts.append(boundary_count)
    total_cost = float(sum(edge_cost_lookup.get(edge, 0.0) for edge in seam_edges))
    return tuple(sorted(seam_edges)), total_cost, len(panel_labels), face_counts, boundary_counts


def _anchor_components(
    anchors: Sequence[int],
    edges: Sequence[Edge],
    edge_costs: np.ndarray,
) -> list[set[int]]:
    if not anchors:
        return []
    finite = edge_costs[np.isfinite(edge_costs)]
    threshold = float(np.percentile(finite, 65.0)) if finite.size else float("inf")
    allowed_edges = [edge for idx, edge in enumerate(edges) if float(edge_costs[idx]) <= threshold]
    graph = _adjacency(allowed_edges)
    anchor_set = {int(anchor) for anchor in anchors}
    remaining = set(anchor_set)
    components: list[set[int]] = []
    while remaining:
        start = remaining.pop()
        seen_vertices = {start}
        seen_anchors = {start}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, ()):
                if neighbor in seen_vertices:
                    continue
                seen_vertices.add(neighbor)
                queue.append(neighbor)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    seen_anchors.add(neighbor)
        components.append(seen_anchors)
    return components


def _anchor_components_on_edges(
    anchors: Sequence[int],
    edges: Sequence[Edge],
) -> list[set[int]]:
    if not anchors:
        return []
    graph = _adjacency(edges)
    anchor_set = {int(anchor) for anchor in anchors}
    remaining = set(anchor_set)
    components: list[set[int]] = []
    while remaining:
        start = remaining.pop()
        seen_vertices = {start}
        seen_anchors = {start}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, ()):
                if neighbor in seen_vertices:
                    continue
                seen_vertices.add(neighbor)
                queue.append(neighbor)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    seen_anchors.add(neighbor)
        components.append(seen_anchors)
    return components


def _dijkstra_path(
    graph: Mapping[int, set[int]],
    edge_costs: Mapping[Edge, float],
    source: int,
    target: int,
) -> tuple[list[Edge], float]:
    if source == target:
        return [], 0.0
    dist = {int(source): 0.0}
    prev: dict[int, int | None] = {int(source): None}
    heap: list[tuple[float, int]] = [(0.0, int(source))]
    while heap:
        current_dist, node = heapq.heappop(heap)
        if node == int(target):
            break
        if current_dist > dist.get(node, float("inf")):
            continue
        for neighbor in sorted(graph.get(node, ())):
            edge = _normalize_edge(node, neighbor)
            weight = max(float(edge_costs.get(edge, 0.0)), 1e-9)
            next_dist = current_dist + weight
            if next_dist < dist.get(neighbor, float("inf")):
                dist[neighbor] = next_dist
                prev[neighbor] = node
                heapq.heappush(heap, (next_dist, neighbor))
    if int(target) not in prev:
        return [], float("inf")

    path_vertices = [int(target)]
    while path_vertices[-1] != int(source):
        parent = prev[path_vertices[-1]]
        if parent is None:
            break
        path_vertices.append(parent)
    path_vertices.reverse()
    path_edges = [
        _normalize_edge(path_vertices[idx], path_vertices[idx + 1])
        for idx in range(len(path_vertices) - 1)
    ]
    return path_edges, float(dist[int(target)])


def _solve_low_cost_paths(
    anchors: Sequence[int],
    edges: Sequence[Edge],
    edge_costs: np.ndarray,
) -> tuple[tuple[Edge, ...], float]:
    if len(anchors) < 2:
        return tuple(), 0.0
    graph = _adjacency(edges)
    cost_lookup = _edge_cost_lookup(edges, edge_costs)
    selected: set[Edge] = set()
    total_cost = 0.0
    ordered = list(dict.fromkeys(int(anchor) for anchor in anchors))
    for source, target in zip(ordered, ordered[1:]):
        path_edges, path_cost = _dijkstra_path(graph, cost_lookup, source, target)
        selected.update(path_edges)
        if np.isfinite(path_cost):
            total_cost += path_cost
    if not selected:
        cheapest_idx = int(np.nanargmin(edge_costs))
        selected.add(edges[cheapest_idx])
        total_cost = float(edge_costs[cheapest_idx])
    else:
        total_cost = float(sum(cost_lookup[edge] for edge in selected))
    return tuple(sorted(selected)), total_cost


def _connected_components(vertex_count: int, edges: Sequence[Edge]) -> list[set[int]]:
    graph = _adjacency(edges)
    remaining = set(range(vertex_count))
    components: list[set[int]] = []
    while remaining:
        start = remaining.pop()
        component = {start}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, ()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _panels_are_disk_proxy(
    *,
    vertex_count: int,
    faces: np.ndarray,
    mesh_edges: Sequence[Edge],
    seam_edges: Sequence[Edge],
) -> tuple[bool, int]:
    remaining_edges = sorted(set(mesh_edges) - set(seam_edges))
    components = _connected_components(vertex_count, remaining_edges)
    face_tuples = [tuple(int(v) for v in face) for face in np.asarray(faces, dtype=int)]
    if not components:
        return False, 0
    for component in components:
        if not component:
            return False, len(components)
        component_edges = [
            edge for edge in remaining_edges if edge[0] in component and edge[1] in component
        ]
        component_faces = [
            face for face in face_tuples if all(vertex in component for vertex in face)
        ]
        chi = len(component) - len(component_edges) + len(component_faces)
        if chi < 1:
            return False, len(components)
    return True, len(components)


def solve_seams(
    *,
    seam_cost_receipt_path: Path,
    costs_path: Path,
    mesh_path: Path,
    output_dir: Path,
    solver_mode: str = "shortest_path",
    anchor_source: str = "field_minima",
    anchor_count: int = 8,
    min_geodesic_separation: float = 0.1,
    manual_anchors: str | None = None,
    min_solver_anchors: int = 2,
    min_seam_edges: int = 1,
    min_seam_vertices: int = 2,
    target_panel_count: int = 4,
    min_panel_faces: int = 16,
    dart_relief_candidates_path: Path | None = None,
    max_dart_candidates: int = 6,
    max_corrections_per_panel: int = 3,
    correction_families: str | Sequence[str] | None = None,
    residual_weight: float = 1.0,
    seam_weight: float = 1.0,
    correction_weight: float = 1.0,
    complexity_weight: float = 1.0,
    manufacture_weight: float = 1.0,
    receipt_path: Path | None = None,
) -> SolverPromotionReceipt:
    """Solve seam topology and emit its promotion receipt."""

    cost_receipt = load_seam_cost_receipt(seam_cost_receipt_path)
    if not can_consume_seam_cost_receipt(cost_receipt, "solver_promotion"):
        raise ValueError(
            f"SeamCostReceipt not promoted ({cost_receipt.promotion}). "
            f"Blocked: {cost_receipt.blocked_consumers}"
        )
    if cost_receipt.costs_hash != _sha256_file(costs_path):
        raise ValueError("Seam costs hash does not match SeamCostReceipt.costs_hash.")

    vertices, faces = _load_mesh(mesh_path)
    if int(vertices.shape[0]) != int(cost_receipt.vertex_count):
        raise ValueError(
            "Mesh vertex count mismatch with SeamCostReceipt: "
            f"mesh={vertices.shape[0]}, receipt={cost_receipt.vertex_count}."
        )
    cost_field = load_seam_cost_field(costs_path)
    edge_costs = np.asarray(cost_field.edge_costs, dtype=float)
    cost_edges = tuple(_normalize_edge(a, b) for a, b in cost_field.edges)
    if len(cost_edges) != int(cost_receipt.edge_count):
        raise ValueError(
            "Seam cost edge count mismatch with SeamCostReceipt: "
            f"costs={len(cost_edges)}, receipt={cost_receipt.edge_count}."
        )
    if len(edge_costs) != len(cost_edges):
        raise ValueError("Seam cost field edge_costs and edges lengths differ.")
    if not np.isfinite(edge_costs).all():
        raise ValueError("Seam cost field contains non-finite edge costs.")

    mesh_edges = _mesh_edges(faces)
    if set(cost_edges) != set(mesh_edges):
        raise ValueError("Seam cost edges do not match mesh topology edges.")

    vertex_costs = _vertex_costs_from_edges(
        int(vertices.shape[0]),
        cost_edges,
        edge_costs,
        np.asarray(cost_field.vertex_costs, dtype=float),
    )
    requested_anchor_count = max(0, int(anchor_count))
    min_solver_anchors = max(0, int(min_solver_anchors))
    min_seam_edges = max(0, int(min_seam_edges))
    min_seam_vertices = max(0, int(min_seam_vertices))
    if anchor_source == "field_minima":
        anchors = _select_field_minima_anchors(
            vertices,
            vertex_costs,
            anchor_count=anchor_count,
            min_separation_ratio=min_geodesic_separation,
        )
    elif anchor_source == "geometric":
        anchors = _select_geometric_anchors(vertices, anchor_count=anchor_count)
    elif anchor_source == "manual":
        anchors = _parse_manual_anchors(manual_anchors, int(vertices.shape[0]))
    else:
        raise ValueError("anchor_source must be field_minima, geometric, or manual.")

    candidate_anchor_count = len(anchors)
    low_cost_anchor_components = _anchor_components(anchors, cost_edges, edge_costs)
    full_graph_anchor_components = _anchor_components_on_edges(anchors, cost_edges)
    connected_component_count = len(full_graph_anchor_components)
    anchor_fallback_used = connected_component_count > 1
    if anchor_fallback_used and full_graph_anchor_components:
        anchors = sorted(
            max(
                full_graph_anchor_components,
                key=lambda component: (len(component), -min(component)),
            )
        )

    dart_edges, dart_warning = _load_dart_candidate_edges(
        dart_relief_candidates_path,
        max_candidates=max_dart_candidates,
    )
    if dart_warning is not None:
        print(f"Warning: {dart_warning}; ignoring --dart-relief-candidates.", file=sys.stderr)

    cut_panel_face_counts: list[int] | None = None
    cut_panel_boundary_counts: list[int] | None = None
    correction_payload: dict[str, object] | None = None
    correction_payload_hash: str | None = None
    raw_residual_total: float | None = None
    corrected_residual_total: float | None = None
    selected_correction_count: int | None = None
    face_labels: np.ndarray | None = None
    if solver_mode in {"cut_graph", "metric_panelization"}:
        edge_to_faces, face_graph = _face_topology(faces)
        edge_lookup = _edge_cost_lookup(cost_edges, edge_costs)
        face_costs = _face_costs(faces, edge_lookup, vertex_costs)
        requested_panels = max(2, min(6, int(target_panel_count), int(faces.shape[0])))
        seeds = _select_seed_faces(
            vertices=vertices,
            faces=faces,
            anchors=anchors,
            face_costs=face_costs,
            target_panel_count=requested_panels,
        )
        labels = _assign_face_regions(
            face_count=int(faces.shape[0]),
            seeds=seeds,
            face_graph=face_graph,
            edge_cost_lookup=edge_lookup,
            dart_edges=dart_edges,
        )
        face_labels = np.asarray(labels, dtype=int)
        connected_component_count = _connected_face_region_count(labels, face_graph)
        (
            seam_edges,
            total_cost,
            panel_count,
            cut_panel_face_counts,
            cut_panel_boundary_counts,
        ) = _cut_graph_from_face_regions(
            faces=faces,
            edge_to_faces=edge_to_faces,
            labels=labels,
            edge_cost_lookup=edge_lookup,
        )
        if solver_mode == "metric_panelization":
            correction_payload = build_metric_panelization_payload(
                vertices=vertices,
                faces=faces,
                labels=labels,
                seam_edges=seam_edges,
                families=normalize_families(correction_families),
                max_corrections_per_panel=max_corrections_per_panel,
                weights=MetricEnergyWeights(
                    residual=float(residual_weight),
                    seam=float(seam_weight),
                    correction=float(correction_weight),
                    complexity=float(complexity_weight),
                    manufacture=float(manufacture_weight),
                ),
            )
            energy = correction_payload["energy"]
            if isinstance(energy, Mapping):
                raw_residual_total = float(energy.get("raw_residual_total", 0.0))
                corrected_residual_total = float(energy.get("corrected_residual_total", 0.0))
            selected_correction_count = int(correction_payload.get("selected_count", 0))
        panels_are_disks = True
    else:
        seam_edges, total_cost = _solve_low_cost_paths(anchors, cost_edges, edge_costs)
        panels_are_disks, panel_count = _panels_are_disk_proxy(
            vertex_count=int(vertices.shape[0]),
            faces=faces,
            mesh_edges=mesh_edges,
            seam_edges=seam_edges,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    seam_path = output_dir / "seam_edges.npz"
    seam_edge_array = np.asarray(seam_edges, dtype=int).reshape((-1, 2))
    if face_labels is not None:
        np.savez_compressed(seam_path, seam_edges=seam_edge_array, face_labels=face_labels)
    else:
        np.savez_compressed(seam_path, seam_edges=seam_edge_array)
    seam_hash = _sha256_file(seam_path)
    if correction_payload is not None:
        correction_path = output_dir / "corrections.json"
        _write_json(correction_path, correction_payload)
        correction_payload_hash = _sha256_file(correction_path)
    seam_vertices = sorted({vertex for edge in seam_edges for vertex in edge})
    solver_blockers: list[str] = []
    if len(anchors) < min_solver_anchors:
        solver_blockers.append("insufficient_solver_anchors")
    if len(seam_edges) < min_seam_edges:
        solver_blockers.append("insufficient_seam_edges")
    if len(seam_vertices) < min_seam_vertices:
        solver_blockers.append("insufficient_seam_vertices")
    if not np.isfinite(total_cost):
        solver_blockers.append("non_finite_total_seam_cost")
    if not panels_are_disks:
        solver_blockers.append("panels_not_disks")
    if solver_mode in {"cut_graph", "metric_panelization"}:
        if panel_count < 2:
            solver_blockers.append("insufficient_cut_panels")
        if cut_panel_boundary_counts is None or any(
            count <= 0 for count in cut_panel_boundary_counts
        ):
            solver_blockers.append("insufficient_cut_boundaries")
        if cut_panel_face_counts is None or any(
            count < max(1, int(min_panel_faces)) for count in cut_panel_face_counts
        ):
            solver_blockers.append("undersized_cut_panel")
    promotion = 1 if not solver_blockers else 0

    receipt = SolverPromotionReceipt(
        seam_cost_receipt_hash=_sha256_file(seam_cost_receipt_path),
        solver_mode=solver_mode,
        anchor_count=len(anchors),
        anchor_source=anchor_source,
        connected_component_count=connected_component_count,
        anchor_fallback_used=anchor_fallback_used,
        seam_edge_count=len(seam_edges),
        seam_vertex_count=len(seam_vertices),
        total_seam_cost=total_cost,
        panel_count=panel_count,
        panels_are_disks=panels_are_disks,
        seam_hash=seam_hash,
        promotion=promotion,
        blocked_consumers=[] if promotion == 1 else [],
        requested_anchor_count=requested_anchor_count,
        candidate_anchor_count=candidate_anchor_count,
        low_cost_anchor_component_count=len(low_cost_anchor_components),
        min_seam_edge_count=min_seam_edges,
        min_seam_vertex_count=min_seam_vertices,
        solver_blockers=solver_blockers,
        correction_payload_hash=correction_payload_hash,
        corrected_residual_total=corrected_residual_total,
        raw_residual_total=raw_residual_total,
        selected_correction_count=selected_correction_count,
    )
    target_receipt_path = receipt_path or (output_dir / "solver_promotion_receipt.json")
    receipt.to_json(target_receipt_path)
    print(f"Wrote seam edges to {seam_path}")
    if correction_payload_hash is not None:
        print(f"Wrote metric corrections to {output_dir / 'corrections.json'}")
    print(f"Wrote solver promotion receipt to {target_receipt_path}")
    return receipt


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seam-cost-receipt", type=Path, required=True)
    parser.add_argument(
        "--costs",
        type=Path,
        required=True,
        help="NPZ created by save_seam_cost_field.",
    )
    parser.add_argument("--mesh", type=Path, required=True, help="Mesh NPZ with vertices/faces.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--out-solver-receipt",
        type=Path,
        default=None,
        help="Output solver receipt path (default: <out-dir>/solver_promotion_receipt.json).",
    )
    parser.add_argument(
        "--solver-mode",
        choices=["shortest_path", "min_cut", "pda_mst", "cut_graph", "metric_panelization"],
        default="shortest_path",
    )
    parser.add_argument(
        "--anchor-source",
        choices=["field_minima", "geometric", "manual"],
        default="field_minima",
    )
    parser.add_argument("--anchor-count", type=int, default=8)
    parser.add_argument(
        "--min-geodesic-separation",
        type=float,
        default=0.1,
        help="Field-minima anchor separation as a fraction of mesh bbox diagonal.",
    )
    parser.add_argument(
        "--manual-anchors",
        type=str,
        default=None,
        help="Comma-separated vertex indices when --anchor-source=manual.",
    )
    parser.add_argument("--min-solver-anchors", type=int, default=2)
    parser.add_argument("--min-seam-edges", type=int, default=1)
    parser.add_argument("--min-seam-vertices", type=int, default=2)
    parser.add_argument("--target-panel-count", type=int, default=4)
    parser.add_argument("--min-panel-faces", type=int, default=16)
    parser.add_argument(
        "--dart-relief-candidates",
        type=Path,
        default=None,
        help="Optional JSON emitted by propose_dart_relief_cuts.py.",
    )
    parser.add_argument("--max-dart-candidates", type=int, default=6)
    parser.add_argument("--max-corrections-per-panel", type=int, default=3)
    parser.add_argument(
        "--correction-families",
        type=str,
        default=",".join(CORRECTION_FAMILIES),
        help="Comma-separated metric correction families.",
    )
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--seam-weight", type=float, default=1.0)
    parser.add_argument("--correction-weight", type=float, default=1.0)
    parser.add_argument("--complexity-weight", type=float, default=1.0)
    parser.add_argument("--manufacture-weight", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    solve_seams(
        seam_cost_receipt_path=args.seam_cost_receipt,
        costs_path=args.costs,
        mesh_path=args.mesh,
        output_dir=args.out_dir,
        solver_mode=args.solver_mode,
        anchor_source=args.anchor_source,
        anchor_count=args.anchor_count,
        min_geodesic_separation=args.min_geodesic_separation,
        manual_anchors=args.manual_anchors,
        min_solver_anchors=args.min_solver_anchors,
        min_seam_edges=args.min_seam_edges,
        min_seam_vertices=args.min_seam_vertices,
        target_panel_count=args.target_panel_count,
        min_panel_faces=args.min_panel_faces,
        dart_relief_candidates_path=args.dart_relief_candidates,
        max_dart_candidates=args.max_dart_candidates,
        max_corrections_per_panel=args.max_corrections_per_panel,
        correction_families=args.correction_families,
        residual_weight=args.residual_weight,
        seam_weight=args.seam_weight,
        correction_weight=args.correction_weight,
        complexity_weight=args.complexity_weight,
        manufacture_weight=args.manufacture_weight,
        receipt_path=args.out_solver_receipt,
    )


if __name__ == "__main__":
    main()
