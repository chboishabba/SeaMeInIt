#!/usr/bin/env python3
"""Solve promoted seam topology from a receipted seam-cost artifact."""

from __future__ import annotations

import argparse
import heapq
import hashlib
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from smii.rom import load_seam_cost_field
from smii.seams import (
    SolverPromotionReceipt,
    can_consume_seam_cost_receipt,
    load_seam_cost_receipt,
)


Edge = tuple[int, int]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    anchor_components = _anchor_components(anchors, cost_edges, edge_costs)
    connected_component_count = len(anchor_components)
    anchor_fallback_used = connected_component_count > 1
    if anchor_fallback_used and anchor_components:
        anchors = sorted(
            max(anchor_components, key=lambda component: (len(component), -min(component)))
        )

    seam_edges, total_cost = _solve_low_cost_paths(anchors, cost_edges, edge_costs)
    panels_are_disks, panel_count = _panels_are_disk_proxy(
        vertex_count=int(vertices.shape[0]),
        faces=faces,
        mesh_edges=mesh_edges,
        seam_edges=seam_edges,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    seam_path = output_dir / "seam_edges.npz"
    np.savez_compressed(seam_path, seam_edges=np.asarray(seam_edges, dtype=int))
    seam_hash = _sha256_file(seam_path)
    seam_vertices = sorted({vertex for edge in seam_edges for vertex in edge})
    promotion = 1 if panels_are_disks else 0

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
    )
    target_receipt_path = receipt_path or (output_dir / "solver_promotion_receipt.json")
    receipt.to_json(target_receipt_path)
    print(f"Wrote seam edges to {seam_path}")
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
        choices=["shortest_path", "min_cut", "pda_mst"],
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
        receipt_path=args.out_solver_receipt,
    )


if __name__ == "__main__":
    main()
