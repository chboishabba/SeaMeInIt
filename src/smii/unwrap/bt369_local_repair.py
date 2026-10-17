"""Bounded chart-local topology edits for Afflec Tranche E experiments.

Edits operate on exact mesh faces. A cut is a chart split, not a sewn dart
with inferred physical take-up. All outputs require fresh backend evaluation.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

from smii.unwrap.bt369_manufacturing import _geometry, _partition


def _adjacency(vertices, faces):
    v, _, edges = _geometry(vertices, faces)
    neighbors = [[] for _ in faces]
    lengths = {}
    for (a, b), incident in edges.items():
        if len(incident) == 2:
            i, j = incident
            length = float(np.linalg.norm(v[a] - v[b]))
            neighbors[i].append(j)
            neighbors[j].append(i)
            lengths[tuple(sorted((i, j)))] = length
    for group in neighbors:
        group.sort()
    return neighbors, lengths


def _components(nodes, neighbors):
    remaining = set(nodes)
    result = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        found = {seed}
        queue = deque([seed])
        while queue:
            node = queue.popleft()
            for nxt in neighbors[node]:
                if nxt in remaining:
                    remaining.remove(nxt)
                    found.add(nxt)
                    queue.append(nxt)
        result.append(tuple(sorted(found)))
    return result


def boundary_shortening(
    *,
    vertices: np.ndarray,
    faces: Sequence[tuple[int, int, int]],
    partition: Sequence[Sequence[int]],
    min_faces: int = 10,
    max_moves: int = 100,
):
    """Greedily transfer boundary faces only when seam length strictly falls.

    The source region must stay connected and both resulting chart sizes
    must satisfy min_faces. Exact 3D seam cost is recomputed per move.
    """
    charts = _partition(partition, len(faces))
    neighbors, lengths = _adjacency(vertices, faces)
    regions = {i: set(c) for i, c in enumerate(charts)}
    owner = {f: i for i, c in regions.items() for f in c}
    moves = []
    for _ in range(max_moves):
        best = None
        for face in range(len(faces)):
            src = owner[face]
            if len(regions[src]) <= min_faces:
                continue
            targets = sorted({owner[n] for n in neighbors[face]} - {src})
            if not targets:
                continue
            # A transfer cannot disconnect its source chart.
            if len(_components(regions[src] - {face}, neighbors)) != 1:
                continue
            for target in targets:
                before = sum(
                    lengths[tuple(sorted((face, n)))] for n in neighbors[face] if owner[n] != src
                )
                after = sum(
                    lengths[tuple(sorted((face, n)))] for n in neighbors[face] if owner[n] != target
                )
                gain = before - after
                if gain > 1e-12:
                    key = (-gain, face, target)
                    if best is None or key < best[0]:
                        best = (key, face, src, target, gain)
        if best is None:
            break
        _, face, src, target, gain = best
        regions[src].remove(face)
        regions[target].add(face)
        owner[face] = target
        moves.append({"face": face, "from": src, "to": target, "seam_length_saved": gain})
    return _partition(tuple(regions.values()), len(faces)), moves


def local_dart_candidates(
    *,
    vertices: np.ndarray,
    faces: Sequence[tuple[int, int, int]],
    partition: Sequence[Sequence[int]],
    chart_indices: Sequence[int],
    min_faces: int = 10,
    max_candidates_per_chart: int = 3,
):
    """Generate bounded connected chart splits for local strain-relief replay.

    Each proposal bisects a selected chart by deterministic dual-graph BFS
    growth from a distant anchor. It is an exploratory cut/dart *candidate*,
    not a claim of distortion relief or material dart take-up.
    """
    charts = _partition(partition, len(faces))
    neighbors, _ = _adjacency(vertices, faces)
    proposals = []
    for chart_index in chart_indices:
        if not 0 <= chart_index < len(charts):
            raise IndexError(f"chart index {chart_index} out of range")
        chart = set(charts[chart_index])
        if len(chart) < 2 * min_faces:
            continue
        seeds = [min(chart), max(chart)]
        for seed in seeds[:max_candidates_per_chart]:
            selected = {seed}
            queue = deque([seed])
            while queue and len(selected) < len(chart) // 2:
                face = queue.popleft()
                for nxt in neighbors[face]:
                    if nxt in chart and nxt not in selected and len(selected) < len(chart) // 2:
                        selected.add(nxt)
                        queue.append(nxt)
            complement = chart - selected
            if len(selected) < min_faces or len(complement) < min_faces:
                continue
            for piece in _components(complement, neighbors):
                if len(piece) < min_faces:
                    break
            else:
                new_partition = _partition(
                    [c for i, c in enumerate(charts) if i != chart_index]
                    + [tuple(sorted(selected))]
                    + _components(complement, neighbors),
                    len(faces),
                )
                if new_partition != charts and not any(
                    p["partition"] == new_partition for p in proposals
                ):
                    proposals.append(
                        {
                            "source_chart_index": chart_index,
                            "seed_face": seed,
                            "partition": new_partition,
                            "claim_boundary": "topological_split_requires_fresh_uv_referee",
                        }
                    )
    return proposals
