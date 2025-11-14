"""Helpers for computing geodesic measurement loops on watertight meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

LayerArray = np.ndarray

__all__ = [
    "MeasurementLoop",
    "build_vertex_adjacency",
    "solve_measurement_loops",
]


@dataclass(frozen=True, slots=True)
class MeasurementLoop:
    """Description of a closed geodesic loop extracted from a mesh."""

    name: str
    target: float
    vertices: tuple[int, ...]
    perimeter: float

    @property
    def relative_error(self) -> float:
        if self.target == 0.0:
            return 0.0
        return abs(self.perimeter - self.target) / self.target


def build_vertex_adjacency(faces: LayerArray, *, num_vertices: int | None = None) -> list[set[int]]:
    """Construct an undirected vertex adjacency list from triangular faces."""

    faces_arr = np.asarray(faces, dtype=int)
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
        raise ValueError("Faces must be shaped (M, 3).")

    if num_vertices is None:
        num_vertices = int(faces_arr.max(initial=-1)) + 1

    adjacency: list[set[int]] = [set() for _ in range(num_vertices)]
    for tri in faces_arr:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        adjacency[a].add(b)
        adjacency[a].add(c)
        adjacency[b].add(a)
        adjacency[b].add(c)
        adjacency[c].add(a)
        adjacency[c].add(b)
    return adjacency


def solve_measurement_loops(
    vertices: LayerArray,
    faces: LayerArray,
    measurements: Mapping[str, float] | None,
    *,
    max_relative_error: float = 0.12,
) -> dict[str, MeasurementLoop]:
    """Approximate closed geodesic loops for circumference measurements."""

    if not measurements:
        return {}

    vertices_arr = np.asarray(vertices, dtype=float)
    if vertices_arr.ndim != 2 or vertices_arr.shape[1] != 3:
        raise ValueError("Vertices must be shaped (N, 3).")

    adjacency = build_vertex_adjacency(faces, num_vertices=len(vertices_arr))
    edge_lengths = _edge_length_map(vertices_arr, adjacency)
    unique_edges = list(edge_lengths.keys())
    axis_projection = _principal_axis_projection(vertices_arr)
    candidate_loops = _enumerate_candidate_loops(adjacency, unique_edges, edge_lengths, axis_projection)

    loops: dict[str, MeasurementLoop] = {}
    for name, target in measurements.items():
        if not name.endswith("_circumference"):
            continue
        target_length = float(target)
        if target_length <= 0.0:
            continue
        loop = _solve_single_measurement(
            name,
            target_length,
            candidate_loops,
            max_relative_error=max_relative_error,
        )
        if loop is not None:
            loops[name] = loop
    return loops


def _edge_length_map(vertices: LayerArray, adjacency: Sequence[set[int]]) -> dict[tuple[int, int], float]:
    edge_lengths: dict[tuple[int, int], float] = {}
    for idx, neighbors in enumerate(adjacency):
        for nbr in neighbors:
            if idx < nbr:
                key = (idx, nbr)
                edge_lengths[key] = float(np.linalg.norm(vertices[idx] - vertices[nbr]))
    return edge_lengths


def _solve_single_measurement(
    name: str,
    target_length: float,
    candidate_loops: Sequence[tuple[tuple[int, ...], float]],
    *,
    max_relative_error: float,
) -> MeasurementLoop | None:
    best_cycle: tuple[int, ...] | None = None
    best_perimeter = float("inf")
    best_error = float("inf")

    for vertices, perimeter in candidate_loops:
        error = abs(perimeter - target_length)
        relative_error = error / target_length
        if relative_error < best_error:
            best_error = relative_error
            best_perimeter = perimeter
            best_cycle = vertices

    if best_cycle is None or best_error > max_relative_error:
        return None

    return MeasurementLoop(
        name=name,
        target=target_length,
        vertices=best_cycle,
        perimeter=best_perimeter,
    )


def _principal_axis_projection(vertices: LayerArray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis_vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    axis_vector = axis_vector / np.linalg.norm(axis_vector)
    return centered @ axis_vector


def _enumerate_candidate_loops(
    adjacency: Sequence[set[int]],
    unique_edges: Sequence[tuple[int, int]],
    edge_lengths: Mapping[tuple[int, int], float],
    axis_projection: np.ndarray,
) -> list[tuple[tuple[int, ...], float]]:
    axis_min = float(axis_projection.min(initial=0.0))
    axis_max = float(axis_projection.max(initial=0.0))
    axis_range = max(axis_max - axis_min, 1e-6)

    sample_count = max(16, min(200, len(adjacency)))
    band_half_width = axis_range / (sample_count * 2.0)
    if band_half_width <= 0.0:
        band_half_width = axis_range * 0.05

    centers = np.linspace(axis_min, axis_max, sample_count)

    loops: dict[frozenset[tuple[int, int]], tuple[tuple[int, ...], float]] = {}
    for center in centers:
        candidate_edges = [
            (u, v)
            for (u, v) in unique_edges
            if abs(axis_projection[u] - center) <= band_half_width
            and abs(axis_projection[v] - center) <= band_half_width
        ]
        if len(candidate_edges) < 3:
            continue
        for cycle_vertices in _extract_cycles(candidate_edges):
            if len(cycle_vertices) < 4:
                continue
            key = _cycle_key(cycle_vertices)
            if key in loops:
                continue
            loops[key] = (
                cycle_vertices,
                _cycle_perimeter(cycle_vertices, edge_lengths),
            )

    return list(loops.values())


def _extract_cycles(edges: Sequence[tuple[int, int]]) -> Iterable[tuple[int, ...]]:
    adjacency: dict[int, set[int]] = {}
    for u, v in edges:
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)

    visited: set[int] = set()
    for start in list(adjacency):
        if start in visited:
            continue
        component: list[int] = []
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(adjacency[node] - visited)

        if any(len(adjacency[vertex]) != 2 for vertex in component):
            continue

        ordered = _order_cycle(component[0], adjacency)
        if ordered:
            yield ordered


def _order_cycle(start: int, adjacency: Mapping[int, set[int]]) -> tuple[int, ...]:
    sequence: list[int] = [start]
    previous = None
    current = start
    while True:
        neighbors = adjacency[current]
        next_candidates = [neighbor for neighbor in neighbors if neighbor != previous]
        if not next_candidates:
            return ()
        next_vertex = next_candidates[0]
        sequence.append(next_vertex)
        previous, current = current, next_vertex
        if current == start:
            break
    return tuple(sequence)


def _cycle_key(vertices: Sequence[int]) -> frozenset[tuple[int, int]]:
    edges = []
    for idx in range(len(vertices) - 1):
        a = vertices[idx]
        b = vertices[idx + 1]
        edges.append((min(a, b), max(a, b)))
    return frozenset(edges)


def _cycle_perimeter(vertices: Sequence[int], edge_lengths: Mapping[tuple[int, int], float]) -> float:
    total = 0.0
    for idx in range(len(vertices) - 1):
        a = vertices[idx]
        b = vertices[idx + 1]
        total += edge_lengths[(min(a, b), max(a, b))]
    return total
