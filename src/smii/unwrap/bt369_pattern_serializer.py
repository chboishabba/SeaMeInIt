"""BT369-native face partitioning for garment pattern serialization.

This module deliberately stops at chart partitioning. It reuses BT369's
certificate vocabulary (residual trit, Z/6 orientation, base-3 prefix and
bounded refinement depth) on an arbitrary face-backed garment panel, but does
not claim that the panel is spherical or that BT369 is a privileged physical
law.

The resulting partition is a candidate panelization. Numerical UV coordinates
and Gate-6 promotion remain owned by the existing serializer competition.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

BT369_PATTERN_PARTITION_SCHEMA = "smii.bt369_pattern_partition.v1"
BT369_PATTERN_CLAIM_BOUNDARY = (
    "bt369_partition_is_candidate_panelization_not_manufacturing_authority"
)

Face = tuple[int, int, int]
Edge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class BT369PatternPolicy:
    """Deterministic local compatibility policy for one partition candidate."""

    orientation_jump_limit: int = 1
    refinement_depth_jump_limit: int = 1
    shared_prefix_depth: int = 1
    isolate_foldovers: bool = True
    cut_residual_trit_boundaries: bool = True
    max_depth: int = 6

    def __post_init__(self) -> None:
        if not 0 <= int(self.orientation_jump_limit) <= 5:
            raise ValueError("orientation_jump_limit must lie in [0, 5]")
        if int(self.refinement_depth_jump_limit) < 0:
            raise ValueError("refinement_depth_jump_limit must be non-negative")
        if int(self.shared_prefix_depth) < 0:
            raise ValueError("shared_prefix_depth must be non-negative")
        if int(self.max_depth) < 0:
            raise ValueError("max_depth must be non-negative")

    @property
    def policy_key(self) -> str:
        return (
            f"o{int(self.orientation_jump_limit)}"
            f"-d{int(self.refinement_depth_jump_limit)}"
            f"-p{int(self.shared_prefix_depth)}"
            f"-f{int(bool(self.isolate_foldovers))}"
            f"-t{int(bool(self.cut_residual_trit_boundaries))}"
            f"-m{int(self.max_depth)}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "orientation_jump_limit": int(self.orientation_jump_limit),
            "refinement_depth_jump_limit": int(self.refinement_depth_jump_limit),
            "shared_prefix_depth": int(self.shared_prefix_depth),
            "isolate_foldovers": bool(self.isolate_foldovers),
            "cut_residual_trit_boundaries": bool(self.cut_residual_trit_boundaries),
            "max_depth": int(self.max_depth),
            "policy_key": self.policy_key,
        }


@dataclass(frozen=True, slots=True)
class BT369PatternCell:
    """BT369-style certificate state attached to one original panel face."""

    face_index: int
    residual: float
    residual_trit: int
    orientation_bin: int
    depth: int
    triadic_prefix: str
    area_weight: float
    foldover: bool
    high_distortion: bool
    centroid_uv: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "face_index": int(self.face_index),
            "residual": float(self.residual),
            "residual_trit": int(self.residual_trit),
            "orientation_bin": int(self.orientation_bin),
            "depth": int(self.depth),
            "triadic_prefix": self.triadic_prefix,
            "area_weight": float(self.area_weight),
            "foldover": bool(self.foldover),
            "high_distortion": bool(self.high_distortion),
            "centroid_uv": [float(self.centroid_uv[0]), float(self.centroid_uv[1])],
        }


@dataclass(frozen=True, slots=True)
class BT369PatternCut:
    """One removed face adjacency and the measured/state reasons for the cut."""

    face_a: int
    face_b: int
    mesh_edge: Edge
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "face_a": int(self.face_a),
            "face_b": int(self.face_b),
            "mesh_edge": [int(self.mesh_edge[0]), int(self.mesh_edge[1])],
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class BT369PatternPartitionReceipt:
    """Exact original-face partition plus its BT369 state certificate."""

    panel_id: str
    face_count: int
    policy: BT369PatternPolicy
    cells: tuple[BT369PatternCell, ...]
    partition: tuple[tuple[int, ...], ...]
    cut_edges: tuple[BT369PatternCut, ...]
    topology_hash: str
    face_coverage_complete: bool
    faces_pairwise_disjoint: bool
    cut_reason_histogram: dict[str, int]
    claim_boundary: str = BT369_PATTERN_CLAIM_BOUNDARY

    @property
    def chart_count(self) -> int:
        return len(self.partition)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": BT369_PATTERN_PARTITION_SCHEMA,
            "claim_boundary": self.claim_boundary,
            "panel_id": self.panel_id,
            "face_count": int(self.face_count),
            "chart_count": self.chart_count,
            "policy": self.policy.to_dict(),
            "cells": [cell.to_dict() for cell in self.cells],
            "partition": [list(chart) for chart in self.partition],
            "cut_edges": [cut.to_dict() for cut in self.cut_edges],
            "topology_hash": self.topology_hash,
            "face_coverage_complete": bool(self.face_coverage_complete),
            "faces_pairwise_disjoint": bool(self.faces_pairwise_disjoint),
            "cut_reason_histogram": dict(self.cut_reason_histogram),
            "promotion": 0,
            "blockers": ["backend_metrics_not_recomputed"],
        }


def default_bt369_pattern_policy_grid() -> tuple[BT369PatternPolicy, ...]:
    """Return the bounded first-pass policy sweep declared by the design."""

    return tuple(
        BT369PatternPolicy(
            orientation_jump_limit=orientation_limit,
            refinement_depth_jump_limit=depth_limit,
            shared_prefix_depth=prefix_depth,
            isolate_foldovers=True,
            cut_residual_trit_boundaries=True,
        )
        for orientation_limit in (0, 1, 2)
        for depth_limit in (0, 1, 2)
        for prefix_depth in (0, 1)
    )


def build_bt369_pattern_partition(
    *,
    vertices: np.ndarray,
    panel: object,
    failure_field: Mapping[str, Any],
    distortion_threshold: float,
    policy: BT369PatternPolicy,
    panel_id: str,
) -> BT369PatternPartitionReceipt:
    """Partition a panel by BT369-style local state compatibility.

    The partition preserves original face identity. No UV coordinates are
    generated here; callers must rerun a numerical serialization backend.
    """

    if not panel_id:
        raise ValueError("panel_id must be non-empty")
    threshold = float(distortion_threshold)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("distortion_threshold must be finite and positive")

    coords = np.asarray(vertices, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError("vertices must be a finite Nx3 array")
    faces = _panel_faces(panel)
    if not faces:
        raise ValueError("panel must contain at least one triangle face")
    if any(min(face) < 0 or max(face) >= len(coords) for face in faces):
        raise ValueError("panel face references a vertex outside vertices")

    residuals = _face_residuals(failure_field, face_count=len(faces))
    foldovers = _face_index_set(failure_field.get("foldover_faces", []), len(faces))
    high_distortion = _face_index_set(failure_field.get("high_distortion_faces", []), len(faces))
    frame = _panel_frame(coords, faces)
    centroids = np.asarray([coords[list(face)].mean(axis=0) for face in faces], dtype=float)
    centroid_uv = _normalized_centroid_uv(centroids, frame)
    face_areas = np.asarray([_triangle_area(coords, face) for face in faces], dtype=float)
    area_total = float(face_areas.sum())
    if area_total <= 1e-15:
        area_weights = np.full(len(faces), 1.0 / len(faces), dtype=float)
    else:
        area_weights = face_areas / area_total

    cells = tuple(
        _face_cell(
            face_index=face_index,
            face=face,
            vertices=coords,
            residual=float(residuals[face_index]),
            distortion_threshold=threshold,
            foldover=face_index in foldovers,
            high_distortion=face_index in high_distortion,
            centroid_uv=(float(centroid_uv[face_index, 0]), float(centroid_uv[face_index, 1])),
            area_weight=float(area_weights[face_index]),
            frame=frame,
            max_depth=int(policy.max_depth),
        )
        for face_index, face in enumerate(faces)
    )

    adjacency = _face_adjacency(faces)
    surviving_graph: dict[int, set[int]] = {idx: set() for idx in range(len(faces))}
    cuts: list[BT369PatternCut] = []
    for (left, right), edge in sorted(adjacency.items()):
        reasons = _cut_reasons(
            cells[left],
            cells[right],
            distortion_threshold=threshold,
            policy=policy,
        )
        if reasons:
            cuts.append(
                BT369PatternCut(
                    face_a=left,
                    face_b=right,
                    mesh_edge=edge,
                    reasons=tuple(reasons),
                )
            )
        else:
            surviving_graph[left].add(right)
            surviving_graph[right].add(left)

    partition = _graph_components(surviving_graph)
    flattened = [face_index for chart in partition for face_index in chart]
    face_coverage_complete = set(flattened) == set(range(len(faces))) and len(flattened) == len(
        faces
    )
    faces_pairwise_disjoint = len(flattened) == len(set(flattened))
    topology_hash = _topology_hash(partition)
    reason_histogram = Counter(reason for cut in cuts for reason in cut.reasons)

    return BT369PatternPartitionReceipt(
        panel_id=panel_id,
        face_count=len(faces),
        policy=policy,
        cells=cells,
        partition=partition,
        cut_edges=tuple(cuts),
        topology_hash=topology_hash,
        face_coverage_complete=face_coverage_complete,
        faces_pairwise_disjoint=faces_pairwise_disjoint,
        cut_reason_histogram=dict(sorted(reason_histogram.items())),
    )


def _panel_faces(panel: object) -> tuple[Face, ...]:
    faces: list[Face] = []
    for raw_face in getattr(panel, "faces"):
        face = tuple(int(vertex) for vertex in raw_face)
        if len(face) != 3:
            raise ValueError("BT369 pattern partition requires triangle faces")
        faces.append((face[0], face[1], face[2]))
    return tuple(faces)


def _face_residuals(failure_field: Mapping[str, Any], *, face_count: int) -> np.ndarray:
    raw = failure_field.get("face_distortions", [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise TypeError("failure_field.face_distortions must be a sequence")
    if len(raw) != face_count:
        raise ValueError("failure_field.face_distortions length must match panel face count")
    residuals = np.asarray([float(value) for value in raw], dtype=float)
    if not np.isfinite(residuals).all() or np.any(residuals < 0.0):
        raise ValueError("failure_field.face_distortions must be finite and non-negative")
    return residuals


def _face_index_set(value: object, face_count: int) -> set[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("failure-field face index collections must be sequences")
    result: set[int] = set()
    for item in value:
        if isinstance(item, bool):
            raise TypeError("failure-field face indices must be integers")
        index = int(item)
        if index < 0 or index >= face_count:
            raise ValueError(f"failure-field face index {index} is outside panel")
        result.add(index)
    return result


def _canonical_direction(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-15:
        raise ValueError("cannot canonicalize a zero direction")
    direction = np.asarray(vector, dtype=float) / norm
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        direction = -direction
    return direction


def _panel_frame(vertices: np.ndarray, faces: Sequence[Face]) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic geometry-attached 2D frame.

    PCA/SVD frames are sign-ambiguous and can rotate inside nearly degenerate
    eigenspaces. For a receipt-bearing orientation code we instead use the
    longest panel chord as ``u`` (deterministic lexicographic tie break) and a
    canonically signed aggregate oriented face normal. ``v = n × u``.
    """

    used = np.asarray(sorted({vertex for face in faces for vertex in face}), dtype=int)
    if len(used) < 2:
        return np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0])
    coords = vertices[used]
    best_pair: tuple[int, int] | None = None
    best_distance = -1.0
    chunk_size = 512
    n = len(used)
    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        c_i = coords[i0:i1]
        for j0 in range(i0, n, chunk_size):
            j1 = min(j0 + chunk_size, n)
            c_j = coords[j0:j1]
            diff = c_i[:, None, :] - c_j[None, :, :]
            d2 = np.sum(diff * diff, axis=-1)
            if i0 == j0:
                idx_i = np.arange(i1 - i0)[:, None]
                idx_j = np.arange(j1 - j0)[None, :]
                d2[idx_i >= idx_j] = -1.0
            max_val = float(np.max(d2))
            if max_val > best_distance - 1e-15:
                rows, cols = np.where(d2 >= max_val - 1e-15)
                for r, c in zip(rows, cols):
                    pair = (int(used[i0 + r]), int(used[j0 + c]))
                    val = float(d2[r, c])
                    if val > best_distance + 1e-15 or (
                        abs(val - best_distance) <= 1e-15
                        and (best_pair is None or pair < best_pair)
                    ):
                        best_distance = val
                        best_pair = pair
    if best_pair is None or best_distance <= 1e-15:
        return np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0])
    axis_u = _canonical_direction(vertices[best_pair[1]] - vertices[best_pair[0]])

    normal = np.zeros(3, dtype=float)
    for a, b, c in faces:
        normal += np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
    if float(np.linalg.norm(normal)) <= 1e-15:
        basis = min(
            (np.eye(3, dtype=float)[axis] for axis in range(3)),
            key=lambda candidate: abs(float(np.dot(candidate, axis_u))),
        )
        normal = np.cross(axis_u, basis)
    normal = _canonical_direction(normal)
    axis_v = np.cross(normal, axis_u)
    if float(np.linalg.norm(axis_v)) <= 1e-15:
        basis = min(
            (np.eye(3, dtype=float)[axis] for axis in range(3)),
            key=lambda candidate: abs(float(np.dot(candidate, axis_u))),
        )
        axis_v = np.cross(basis, axis_u)
    axis_v = _canonical_direction(axis_v)
    return axis_u, axis_v


def _normalized_centroid_uv(
    centroids: np.ndarray,
    frame: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    axis_u, axis_v = frame
    origin = centroids.mean(axis=0)
    projected = np.column_stack(((centroids - origin) @ axis_u, (centroids - origin) @ axis_v))
    mins = projected.min(axis=0)
    spans = projected.max(axis=0) - mins
    result = np.empty_like(projected)
    for column in range(2):
        if float(spans[column]) <= 1e-15:
            result[:, column] = 0.5
        else:
            result[:, column] = (projected[:, column] - mins[column]) / spans[column]
    return np.clip(result, 0.0, 1.0)


def _triangle_area(vertices: np.ndarray, face: Face) -> float:
    a, b, c = face
    return 0.5 * float(
        np.linalg.norm(np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a]))
    )


def _residual_trit(*, residual: float, threshold: float, foldover: bool) -> int:
    if foldover:
        return -1
    if residual <= threshold:
        return 0
    return 1


def _refinement_depth(*, residual: float, threshold: float, max_depth: int) -> int:
    if residual <= threshold or max_depth <= 0:
        return 0
    ratio = max(residual / threshold, 1.0)
    return min(max_depth, 1 + int(math.floor(math.log(ratio, 3.0))))


def _triadic_prefix(u: float, v: float, depth: int) -> str:
    if depth <= 0:
        return ""
    scale = 3**depth
    u_clamped = min(max(float(u), 0.0), 1.0)
    v_clamped = min(max(float(v), 0.0), 1.0)
    x = min(scale - 1, max(0, int(u_clamped * scale)))
    y = min(scale - 1, max(0, int(v_clamped * scale)))
    digits: list[str] = []
    for level in range(depth - 1, -1, -1):
        place = 3**level
        digits.append(str(((x // place) + (y // place)) % 3))
    return "".join(digits)


def _orientation_bin(
    vertices: np.ndarray,
    face: Face,
    frame: tuple[np.ndarray, np.ndarray],
) -> int:
    axis_u, axis_v = frame
    directed_edges = (
        vertices[face[1]] - vertices[face[0]],
        vertices[face[2]] - vertices[face[1]],
        vertices[face[0]] - vertices[face[2]],
    )
    edge = max(
        enumerate(directed_edges),
        key=lambda pair: (float(np.linalg.norm(pair[1])), -pair[0]),
    )[1]
    x = float(np.dot(edge, axis_u))
    y = float(np.dot(edge, axis_v))
    if abs(x) <= 1e-15 and abs(y) <= 1e-15:
        return 0
    angle = math.atan2(y, x) % (2.0 * math.pi)
    return min(5, int(math.floor(angle / (2.0 * math.pi / 6.0))))


def _face_cell(
    *,
    face_index: int,
    face: Face,
    vertices: np.ndarray,
    residual: float,
    distortion_threshold: float,
    foldover: bool,
    high_distortion: bool,
    centroid_uv: tuple[float, float],
    area_weight: float,
    frame: tuple[np.ndarray, np.ndarray],
    max_depth: int,
) -> BT369PatternCell:
    depth = _refinement_depth(
        residual=residual,
        threshold=distortion_threshold,
        max_depth=max_depth,
    )
    return BT369PatternCell(
        face_index=face_index,
        residual=residual,
        residual_trit=_residual_trit(
            residual=residual,
            threshold=distortion_threshold,
            foldover=foldover,
        ),
        orientation_bin=_orientation_bin(vertices, face, frame),
        depth=depth,
        triadic_prefix=_triadic_prefix(centroid_uv[0], centroid_uv[1], depth),
        area_weight=area_weight,
        foldover=foldover,
        high_distortion=high_distortion,
        centroid_uv=centroid_uv,
    )


def _normalize_edge(a: int, b: int) -> Edge:
    return (a, b) if a <= b else (b, a)


def _face_adjacency(faces: Sequence[Face]) -> dict[tuple[int, int], Edge]:
    edge_to_faces: dict[Edge, list[int]] = defaultdict(list)
    for face_index, (a, b, c) in enumerate(faces):
        for edge in (_normalize_edge(a, b), _normalize_edge(b, c), _normalize_edge(c, a)):
            edge_to_faces[edge].append(face_index)
    adjacency: dict[tuple[int, int], Edge] = {}
    for edge, incident in edge_to_faces.items():
        if len(incident) == 2:
            left, right = sorted(incident)
            adjacency[(left, right)] = edge
    return adjacency


def _circular_orientation_distance(left: int, right: int) -> int:
    raw = abs(int(left) - int(right)) % 6
    return min(raw, 6 - raw)


def _prefix_mismatch(
    left: BT369PatternCell,
    right: BT369PatternCell,
    required_depth: int,
) -> bool:
    if required_depth <= 0:
        return False
    if left.depth < required_depth or right.depth < required_depth:
        return False
    return left.triadic_prefix[:required_depth] != right.triadic_prefix[:required_depth]


def _cut_reasons(
    left: BT369PatternCell,
    right: BT369PatternCell,
    *,
    distortion_threshold: float,
    policy: BT369PatternPolicy,
) -> list[str]:
    reasons: list[str] = []
    either_hot = max(left.residual, right.residual) > distortion_threshold
    if policy.isolate_foldovers and left.foldover != right.foldover:
        reasons.append("foldover_boundary")
    if (
        policy.cut_residual_trit_boundaries
        and left.residual_trit != right.residual_trit
        and either_hot
    ):
        reasons.append("residual_trit_boundary")
    if either_hot and _circular_orientation_distance(
        left.orientation_bin, right.orientation_bin
    ) > int(policy.orientation_jump_limit):
        reasons.append("orientation_jump")
    if abs(left.depth - right.depth) > int(policy.refinement_depth_jump_limit):
        reasons.append("refinement_depth_jump")
    if _prefix_mismatch(left, right, int(policy.shared_prefix_depth)):
        reasons.append("triadic_prefix_mismatch")
    return reasons


def _graph_components(graph: Mapping[int, set[int]]) -> tuple[tuple[int, ...], ...]:
    remaining = set(graph)
    components: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(graph.get(node, ())):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(
        sorted(
            components,
            key=lambda component: (component[0], len(component), component),
        )
    )


def _topology_hash(partition: Sequence[Sequence[int]]) -> str:
    payload = json.dumps(
        [list(chart) for chart in partition],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
