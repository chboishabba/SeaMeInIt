"""ROM-weighted edge routing and topologically exact chart opening.

A cut duplicates mesh vertex corners across cut edges; it never removes faces.
Costs are nonnegative edge penalties, not inverse strain rewards.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np


@dataclass(frozen=True)
class ReliefRoute:
    vertex_path: tuple[int, ...]
    cut_edges: tuple[tuple[int, int], ...]
    integrated_cost: float


def minimum_rom_relief(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_cost: np.ndarray,
    start: int,
    end: int,
    *,
    length_weight: float = 1.0,
    rom_weight: float = 1.0,
) -> ReliefRoute:
    """Shortest mesh-edge route between designated seam/opening endpoints.

    The caller must supply topologically admissible endpoints; the routing
    objective cannot by itself establish that a cut opens a closed tube.
    """
    xyz = np.asarray(vertices, dtype=float)
    tri = np.asarray(faces, dtype=np.int64)
    cost = np.asarray(vertex_cost, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("Expected (N,3) vertices and (F,3) triangles")
    if len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError("Vertices must be nonempty and finite")
    if np.any(tri < 0) or np.any(tri >= len(xyz)):
        raise ValueError("Invalid face indices")
    if cost.shape != (len(xyz),) or not np.isfinite(cost).all() or np.any(cost < 0):
        raise ValueError("ROM costs must be finite nonnegative per-vertex values")
    if length_weight <= 0 or rom_weight < 0:
        raise ValueError("Invalid routing weights")
    if not (0 <= start < len(xyz) and 0 <= end < len(xyz)) or start == end:
        raise ValueError("Distinct valid endpoints required")
    graph: list[dict[int, float]] = [{} for _ in xyz]
    for a, b, c in tri:
        for i, j in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            length = float(np.linalg.norm(xyz[i] - xyz[j]))
            if length <= 0:
                raise ValueError("Zero-length mesh edge")
            weight = length * (length_weight + rom_weight * (cost[i] + cost[j]) / 2)
            graph[i][j] = weight
            graph[j][i] = weight
    distance = {start: 0.0}
    previous: dict[int, int] = {}
    queue = [(0.0, start)]
    while queue:
        d, node = heapq.heappop(queue)
        if d > distance.get(node, float("inf")):
            continue
        if node == end:
            break
        for nxt, weight in sorted(graph[node].items()):
            nd = d + weight
            if nd < distance.get(nxt, float("inf")) - 1e-12:
                distance[nxt] = nd
                previous[nxt] = node
                heapq.heappush(queue, (nd, nxt))
    if end not in distance:
        raise ValueError("Endpoints disconnected")
    path = [end]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    edges = tuple(sorted((min(a, b), max(a, b)) for a, b in zip(path, path[1:])))
    return ReliefRoute(tuple(path), edges, distance[end])


def open_mesh_along_route(
    vertices: np.ndarray,
    faces: np.ndarray,
    route: ReliefRoute,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split face corners across cut edges, preserving exact source correspondence.

    Returns duplicated vertex coordinates, remapped faces and source vertex IDs.
    """
    xyz = np.asarray(vertices, dtype=float)
    tri = np.asarray(faces, dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("Expected triangle faces")
    cut = set(route.cut_edges)
    adjacency: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for fi, face in enumerate(tri):
        for k in range(3):
            a, b = int(face[k]), int(face[(k + 1) % 3])
            adjacency.setdefault((min(a, b), max(a, b)), []).append(
                (fi, k, (k + 1) % 3)
            )
    if not cut.issubset(adjacency):
        raise ValueError("Cut contains an edge absent from the mesh")
    if any(len(inc) > 2 for inc in adjacency.values()):
        raise ValueError("Nonmanifold source edge")
    parent = list(range(tri.size))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a: int, b: int) -> None:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for edge, inc in adjacency.items():
        if edge in cut or len(inc) != 2:
            continue
        (f0, _, _), (f1, _, _) = inc
        for v in edge:
            k0 = int(np.flatnonzero(tri[f0] == v)[0])
            k1 = int(np.flatnonzero(tri[f1] == v)[0])
            join(3 * f0 + k0, 3 * f1 + k1)
    index: dict[int, int] = {}
    source: list[int] = []
    remapped = np.empty_like(tri)
    for f in range(len(tri)):
        for k in range(3):
            r = root(3 * f + k)
            if r not in index:
                index[r] = len(source)
                source.append(int(tri[f, k]))
            remapped[f, k] = index[r]
    source_ids = np.asarray(source, dtype=np.int64)
    return xyz[source_ids].copy(), remapped, source_ids
