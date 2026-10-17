"""Deterministic surface-graph seam candidates for BT369 panel experiments.

Graph nodes are faces and arcs are shared mesh edges. Path tracing uses
Dijkstra with positive physical edge costs; labels are grown from spatially
separated anchors. This is a discrete flow-guided proxy, not a continuous
geodesic, garment stress solution, or manufacturing certification.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from smii.unwrap.bt369_manufacturing import ManufacturingLimits, _geometry, _partition, regularize_partition


@dataclass(frozen=True)
class SeamWeights:
    alpha: float = 1.0
    beta: float = 0.6
    gamma: float = 0.5
    delta: float = 0.3

    def __post_init__(self) -> None:
        if any(not math.isfinite(x) or x < 0 for x in (self.alpha, self.beta, self.gamma, self.delta)):
            raise ValueError("seam weights must be finite and nonnegative")
        if self.alpha <= 0:
            raise ValueError("length weight must be positive")


def surface_graph(*, vertices: np.ndarray, faces: Sequence[tuple[int, int, int]],
                  cells: Sequence[object], weights: SeamWeights = SeamWeights()):
    """Return face graph with reproducible nonnegative surface edge costs."""
    v, areas, edges = _geometry(vertices, faces)
    if len(cells) != len(faces):
        raise ValueError("cell count must equal face count")
    tri = v[np.asarray(faces, dtype=int)]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(norms[:, None], 1e-15)
    graph = [[] for _ in faces]
    for (a, b), incident in sorted(edges.items()):
        if len(incident) != 2:
            continue
        i, j = incident
        length = float(np.linalg.norm(v[a] - v[b]))
        orient = min(abs(int(cells[i].orientation_bin) - int(cells[j].orientation_bin)) % 6,
                     6 - abs(int(cells[i].orientation_bin) - int(cells[j].orientation_bin)) % 6) / 3
        residual = abs(float(cells[i].residual) - float(cells[j].residual))
        curvature = 1 - float(np.clip(np.dot(normals[i], normals[j]), -1, 1))
        cost = (weights.alpha * length + weights.beta * orient * length
                + weights.gamma * residual * length + weights.delta * curvature * length)
        cost = max(cost, 1e-12)
        graph[i].append((j, cost))
        graph[j].append((i, cost))
    for neighbors in graph:
        neighbors.sort()
    return graph


def _components(graph):
    seen = set()
    components = []
    for start in range(len(graph)):
        if start in seen:
            continue
        seen.add(start)
        component = []
        queue = deque([start])
        while queue:
            i = queue.popleft()
            component.append(i)
            for j, _ in graph[i]:
                if j not in seen:
                    seen.add(j)
                    queue.append(j)
        components.append(tuple(sorted(component)))
    return components


def _grow(graph, seeds):
    distances = [math.inf] * len(graph)
    labels = [-1] * len(graph)
    queue = []
    for label, seed in enumerate(seeds):
        distances[seed] = 0.0
        labels[seed] = label
        heapq.heappush(queue, (0.0, label, seed))
    while queue:
        d, label, node = heapq.heappop(queue)
        if (d, label) != (distances[node], labels[node]):
            continue
        for other, cost in graph[node]:
            new = d + cost
            if (new, label) < (distances[other], labels[other]):
                distances[other], labels[other] = new, label
                heapq.heappush(queue, (new, label, other))
    return distances, labels


def flow_trace_partition(*, vertices: np.ndarray, faces: Sequence[tuple[int, int, int]],
                         cells: Sequence[object], limits: ManufacturingLimits,
                         weights: SeamWeights = SeamWeights(),
                         target_charts: int = 24):
    """Multi-source minimal-cost paths produce connected macroscopic chart domains.

    Farthest-point anchors are selected using surface-graph distance. Each
    multi-source shortest-path tree is connected to its seed, and the
    inter-tree edges constitute a coherent seam network.
    """
    if target_charts < 1:
        raise ValueError("target_charts must be positive")
    graph = surface_graph(vertices=vertices, faces=faces, cells=cells, weights=weights)
    components = _components(graph)
    if target_charts < len(components):
        raise ValueError("target_charts smaller than disconnected surface components")
    seeds = [c[0] for c in components]
    distances, labels = _grow(graph, seeds)
    while len(seeds) < min(target_charts, len(faces)):
        eligible = [(distances[i], -i, i) for i in range(len(faces)) if i not in seeds]
        if not eligible:
            break
        _, _, seed = max(eligible)
        seeds.append(seed)
        distances, labels = _grow(graph, seeds)
    regions = defaultdict(list)
    for i, label in enumerate(labels):
        regions[label].append(i)
    partition = _partition(tuple(regions.values()), len(faces))
    return regularize_partition(vertices=vertices, faces=faces, partition=partition,
                                cells=cells, limits=limits)


def morphological_cleanup(*, vertices: np.ndarray, faces: Sequence[tuple[int, int, int]],
                          partition: Sequence[Sequence[int]], cells: Sequence[object],
                          limits: ManufacturingLimits, rounds: int = 2):
    """Simultaneous strict-neighbor-majority relabeling followed by merge-back.

    Changes only interior faces whose neighbors unanimously agree, avoiding
    arbitrary Laplacian motion of the mesh boundary.
    """
    if rounds < 0:
        raise ValueError("rounds must be nonnegative")
    charts = _partition(partition, len(faces))
    owner = {f: i for i, chart in enumerate(charts) for f in chart}
    _, _, edges = _geometry(vertices, faces)
    graph = [[] for _ in faces]
    for incident in edges.values():
        if len(incident) == 2:
            a, b = incident
            graph[a].append(b)
            graph[b].append(a)
    for _ in range(rounds):
        changes = {}
        for face, neighbors in enumerate(graph):
            if len(neighbors) >= 2 and all(owner[n] == owner[neighbors[0]] for n in neighbors):
                if owner[face] != owner[neighbors[0]]:
                    changes[face] = owner[neighbors[0]]
        if not changes:
            break
        owner.update(changes)
    regions = defaultdict(list)
    for face, label in sorted(owner.items()):
        regions[label].append(face)
    # Connected-component split ensures no disconnected chart is silently accepted.
    split = []
    for region in regions.values():
        remaining = set(region)
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            component = {seed}
            queue = deque([seed])
            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            split.append(tuple(sorted(component)))
    return regularize_partition(vertices=vertices, faces=faces, partition=split,
                                cells=cells, limits=limits)


def sensitivity(diagnostics: dict) -> list[dict[str, object]]:
    ratio = float(diagnostics["seam_length_over_sqrt_area"])
    count = int(diagnostics["chart_count"])
    area_ok = ("subminimum_chart_area" not in diagnostics["manufacturing_blockers"])
    return [{"max_seam_ratio": seam, "max_charts": charts,
             "passes": area_ok and ratio <= seam and count <= charts}
            for seam in (8, 12, 16, 24) for charts in (32, 64, 128)]
