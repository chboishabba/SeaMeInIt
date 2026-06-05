"""Gate 6c panel serialization backend competition receipts."""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .unwrap_backends import BOOTSTRAP_BACKEND, LSCM_BACKEND, unwrap_panel_vertices

XATLAS_BACKEND = "xatlas"
PANEL_SERIALIZATION_BACKENDS = (BOOTSTRAP_BACKEND, LSCM_BACKEND, XATLAS_BACKEND)
PANEL_SERIALIZATION_SCHEMA = "smii.panel_serialization_competition.v1"
DENSE_LSCM_VERTEX_LIMIT = 2000
DENSE_LSCM_FACE_LIMIT = 1800
DENSE_LSCM_SYSTEM_LIMIT = 1_500_000

Face = tuple[int, int, int]
Edge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class SerializationCandidateReceipt:
    """Measured result for one backend attempting to serialize one panel."""

    backend: str
    diagnostic_only: bool
    available: bool
    distortion: float | None
    foldovers: int | None
    boundary_deviation: float | None
    island_fragmentation: int
    exportable: bool
    score: float | None
    promoted: bool
    blockers: tuple[str, ...]
    chart_diagnostics: dict[str, object] | None = None
    materialization_constraints: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "diagnostic_only": self.diagnostic_only,
            "available": self.available,
            "distortion": self.distortion,
            "foldovers": self.foldovers,
            "boundary_deviation": self.boundary_deviation,
            "island_fragmentation": int(self.island_fragmentation),
            "exportable": self.exportable,
            "score": self.score,
            "promoted": self.promoted,
            "blockers": list(self.blockers),
            "chart_diagnostics": self.chart_diagnostics,
            "materialization_constraints": self.materialization_constraints,
        }


def serialize_panel(
    *,
    vertices: np.ndarray,
    panel: object,
    correction_tree: Mapping[str, object] | None,
    materialization_constraints: Mapping[str, object] | None = None,
    backend: str,
    distortion_threshold: float,
) -> tuple[SerializationCandidateReceipt, np.ndarray | None]:
    """Run one serialization backend and return its measured candidate receipt."""

    diagnostic_only = backend == BOOTSTRAP_BACKEND
    chart_diagnostics = panel_chart_diagnostics(vertices, panel)
    if not diagnostic_only and not bool(chart_diagnostics["backend_serializable"]):
        return _blocked_candidate(
            backend,
            diagnostic_only=False,
            blockers=("backend_skipped_invalid_chart_domain",),
            chart_diagnostics=chart_diagnostics,
            materialization_constraints=materialization_constraints,
        ), None
    panel_vertex_count = len(tuple(getattr(panel, "vertices")))
    panel_face_count = len(tuple(getattr(panel, "faces")))
    dense_lscm = (
        panel_vertex_count > DENSE_LSCM_VERTEX_LIMIT
        or panel_face_count > DENSE_LSCM_FACE_LIMIT
        or panel_vertex_count * max(1, panel_face_count) > DENSE_LSCM_SYSTEM_LIMIT
    )
    if backend == LSCM_BACKEND and dense_lscm:
        return _blocked_candidate(
            backend,
            diagnostic_only=False,
            blockers=("backend_resource_limit",),
            chart_diagnostics=chart_diagnostics,
            materialization_constraints=materialization_constraints,
        ), None
    try:
        uv = _unwrap(vertices=vertices, panel=panel, backend=backend)
    except ModuleNotFoundError:
        return _blocked_candidate(
            backend,
            diagnostic_only=diagnostic_only,
            blockers=("backend_unavailable",),
            materialization_constraints=materialization_constraints,
        ), None
    except Exception as exc:  # pragma: no cover - receipt should preserve backend failures.
        return _blocked_candidate(
            backend,
            diagnostic_only=diagnostic_only,
            blockers=(f"backend_failed:{exc.__class__.__name__}",),
            materialization_constraints=materialization_constraints,
        ), None

    blockers: list[str] = []
    exportable = _uv_is_exportable(uv, panel)
    if not exportable:
        blockers.append("boundary_not_exportable")
    distortion = _compute_distortion(vertices, panel, uv)
    if distortion > distortion_threshold:
        blockers.append("distortion_exceeds_threshold")
    foldovers = _count_foldovers(panel, uv)
    if foldovers > 0:
        blockers.append("foldovers_present")
    boundary_deviation = _boundary_deviation(vertices, panel, uv)
    island_fragmentation = _island_fragmentation(panel)
    if diagnostic_only:
        blockers.append("diagnostic_only_backend")
    if not _has_triangle_topology(panel):
        blockers.append("missing_triangle_topology")
    score = _score(
        distortion=distortion,
        foldovers=foldovers,
        boundary_deviation=boundary_deviation,
        island_fragmentation=island_fragmentation,
        exportable=exportable,
    )
    promoted = not blockers
    return (
        SerializationCandidateReceipt(
            backend=backend,
            diagnostic_only=diagnostic_only,
            available=True,
            distortion=distortion,
            foldovers=foldovers,
            boundary_deviation=boundary_deviation,
            island_fragmentation=island_fragmentation,
            exportable=exportable,
            score=score,
            promoted=promoted,
            blockers=tuple(dict.fromkeys(blockers)),
            chart_diagnostics=chart_diagnostics,
            materialization_constraints=dict(materialization_constraints)
            if materialization_constraints is not None
            else None,
        ),
        uv,
    )


def build_panel_serialization_competition_receipt(
    *,
    panel_id: str,
    correction_tree_hash: str | None,
    correction_tree: Mapping[str, object] | None,
    candidates: Sequence[SerializationCandidateReceipt],
    selected_backend: str | None,
) -> dict[str, object]:
    """Return the JSON receipt for one panel's backend competition."""

    selected = next(
        (candidate for candidate in candidates if candidate.backend == selected_backend),
        None,
    )
    return {
        "schema_version": PANEL_SERIALIZATION_SCHEMA,
        "claim_boundary": "serialization_is_not_morphology_authority",
        "panel_id": panel_id,
        "correction_tree_hash": correction_tree_hash,
        "correction_tree_promoted": None
        if correction_tree is None
        else str(correction_tree.get("promotion", 0)) == "1",
        "candidates": [candidate.to_dict() for candidate in candidates],
        "selected_backend": selected_backend,
        "promotion": 1 if selected is not None and selected.promoted else 0,
        "blockers": []
        if selected is not None and selected.promoted
        else _competition_blockers(candidates),
    }


def select_serialization_candidate(
    candidates: Sequence[SerializationCandidateReceipt],
) -> SerializationCandidateReceipt:
    """Select the best hard-gate-satisfying candidate, falling back to diagnostics."""

    promotable = [candidate for candidate in candidates if candidate.promoted]
    if promotable:
        return min(promotable, key=lambda candidate: float(candidate.score or math.inf))
    available = [candidate for candidate in candidates if candidate.available]
    if available:
        return min(
            available,
            key=lambda candidate: float(
                candidate.score if candidate.score is not None else math.inf
            ),
        )
    return candidates[0]


def _unwrap(*, vertices: np.ndarray, panel: object, backend: str) -> np.ndarray:
    if backend == XATLAS_BACKEND:
        return _xatlas_unwrap(vertices, panel)
    return unwrap_panel_vertices(
        vertices,
        panel_vertices=tuple(int(vertex) for vertex in getattr(panel, "vertices")),
        panel_faces=_panel_faces(panel),
        method=backend,
    )


def _xatlas_unwrap(vertices: np.ndarray, panel: object) -> np.ndarray:
    try:
        import xatlas  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ModuleNotFoundError("xatlas") from exc

    panel_vertices = tuple(int(vertex) for vertex in getattr(panel, "vertices"))
    panel_faces = _panel_faces(panel)
    local_index = {vertex: idx for idx, vertex in enumerate(panel_vertices)}
    local_faces = np.asarray(
        [
            (local_index[a], local_index[b], local_index[c])
            for a, b, c in panel_faces
            if a in local_index and b in local_index and c in local_index
        ],
        dtype=np.uint32,
    )
    local_vertices = np.asarray(vertices[list(panel_vertices)], dtype=np.float32)
    if local_faces.size == 0:
        raise ValueError("missing triangle topology")
    vmapping, indices, uvs = xatlas.parametrize(local_vertices, local_faces)
    uv_by_local = np.zeros((len(panel_vertices), 2), dtype=float)
    counts = np.zeros(len(panel_vertices), dtype=float)
    for atlas_vertex, source_vertex in enumerate(np.asarray(vmapping, dtype=int)):
        if 0 <= source_vertex < len(panel_vertices):
            uv_by_local[source_vertex] += np.asarray(uvs[atlas_vertex], dtype=float)
            counts[source_vertex] += 1.0
    missing = counts <= 0.0
    if np.any(missing):
        raise ValueError("xatlas uv mapping incomplete")
    uv_by_local /= counts[:, None]
    _ = indices
    return uv_by_local


def _blocked_candidate(
    backend: str,
    *,
    diagnostic_only: bool,
    blockers: Sequence[str],
    chart_diagnostics: dict[str, object] | None = None,
    materialization_constraints: Mapping[str, object] | None = None,
) -> SerializationCandidateReceipt:
    return SerializationCandidateReceipt(
        backend=backend,
        diagnostic_only=diagnostic_only,
        available=False,
        distortion=None,
        foldovers=None,
        boundary_deviation=None,
        island_fragmentation=0,
        exportable=False,
        score=None,
        promoted=False,
        blockers=tuple(dict.fromkeys(blockers)),
        chart_diagnostics=chart_diagnostics,
        materialization_constraints=dict(materialization_constraints)
        if materialization_constraints is not None
        else None,
    )


def panel_chart_diagnostics(vertices: np.ndarray, panel: object) -> dict[str, object]:
    """Measure whether a panel is a coherent chart domain for production backends."""

    panel_vertices = tuple(int(vertex) for vertex in getattr(panel, "vertices"))
    vertex_set = set(panel_vertices)
    faces = _panel_faces(panel)
    face_keys = [tuple(sorted(face)) for face in faces]
    duplicate_face_count = sum(count - 1 for count in Counter(face_keys).values() if count > 1)
    degenerate_triangle_count = sum(
        1 for face in faces if len(set(face)) < 3 or _triangle_area(vertices, face) <= 1e-12
    )
    face_components = _face_connected_components(faces)
    edge_to_faces: dict[Edge, list[int]] = defaultdict(list)
    for face_idx, face in enumerate(faces):
        for edge in _face_edges(face):
            edge_to_faces[edge].append(face_idx)
    nonmanifold_edges = sorted(edge for edge, indices in edge_to_faces.items() if len(indices) > 2)
    boundary_edges = tuple(edge for edge, indices in edge_to_faces.items() if len(indices) == 1)
    boundary_loop_count = _boundary_loop_count(boundary_edges)
    used_vertices = {vertex for face in faces for vertex in face}
    isolated_vertices = sorted(vertex for vertex in vertex_set if vertex not in used_vertices)
    unknown_face_vertices = sorted(used_vertices - vertex_set)
    oriented_faces = _faces_orientation_consistent(faces)
    blockers: list[str] = []
    if len(face_components) != 1:
        blockers.append("panel_fragmentation_invalid")
    if nonmanifold_edges:
        blockers.append("nonmanifold_panel_extract")
    if duplicate_face_count:
        blockers.append("duplicate_panel_faces")
    if degenerate_triangle_count:
        blockers.append("degenerate_panel_triangles")
    if isolated_vertices:
        blockers.append("isolated_panel_vertices")
    if unknown_face_vertices:
        blockers.append("panel_face_vertices_not_in_panel")
    if boundary_edges and not oriented_faces:
        blockers.append("inconsistent_panel_orientation")
    if not boundary_edges:
        blockers.append("unresolved_open_boundary")
    return {
        "connected_components": len(face_components),
        "component_face_counts": [len(component) for component in face_components],
        "nonmanifold_edges": len(nonmanifold_edges),
        "boundary_loops": boundary_loop_count,
        "boundary_edges": len(boundary_edges),
        "oriented_faces": oriented_faces,
        "duplicate_faces": duplicate_face_count,
        "degenerate_triangles": degenerate_triangle_count,
        "isolated_vertices": len(isolated_vertices),
        "unknown_face_vertices": len(unknown_face_vertices),
        "backend_serializable": not blockers,
        "blockers": blockers,
    }


def _panel_faces(panel: object) -> tuple[Face, ...]:
    faces: list[Face] = []
    for face in getattr(panel, "faces"):
        vertices = tuple(int(v) for v in face)
        if len(vertices) == 3:
            faces.append((vertices[0], vertices[1], vertices[2]))
    return tuple(faces)


def _triangle_area(vertices: np.ndarray, face: Face) -> float:
    a, b, c = face
    if max(face) >= len(vertices) or min(face) < 0:
        return 0.0
    return 0.5 * float(
        np.linalg.norm(np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a]))
    )


def _face_edges(face: Face) -> tuple[Edge, Edge, Edge]:
    a, b, c = face
    return (_normalize_edge(a, b), _normalize_edge(b, c), _normalize_edge(c, a))


def _normalize_edge(a: int, b: int) -> Edge:
    return (a, b) if a <= b else (b, a)


def _face_connected_components(faces: Sequence[Face]) -> list[set[int]]:
    if not faces:
        return []
    edge_to_faces: dict[Edge, list[int]] = defaultdict(list)
    for face_idx, face in enumerate(faces):
        for edge in _face_edges(face):
            edge_to_faces[edge].append(face_idx)
    graph: dict[int, set[int]] = defaultdict(set)
    for face_indices in edge_to_faces.values():
        if len(face_indices) < 2:
            continue
        for idx in face_indices:
            graph[idx].update(other for other in face_indices if other != idx)
    remaining = set(range(len(faces)))
    components: list[set[int]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        queue: deque[int] = deque([start])
        while queue:
            face_idx = queue.popleft()
            for neighbor in sorted(graph.get(face_idx, ())):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _boundary_loop_count(boundary_edges: Sequence[Edge]) -> int:
    if not boundary_edges:
        return 0
    graph: dict[int, set[int]] = defaultdict(set)
    for a, b in boundary_edges:
        graph[a].add(b)
        graph[b].add(a)
    remaining = set(graph)
    component_count = 0
    while remaining:
        component_count += 1
        start = min(remaining)
        remaining.remove(start)
        queue: deque[int] = deque([start])
        while queue:
            vertex = queue.popleft()
            for neighbor in sorted(graph.get(vertex, ())):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return component_count


def _faces_orientation_consistent(faces: Sequence[Face]) -> bool:
    directed_edges: dict[Edge, list[tuple[int, int]]] = defaultdict(list)
    for face in faces:
        a, b, c = face
        for u, v in ((a, b), (b, c), (c, a)):
            directed_edges[_normalize_edge(u, v)].append((u, v))
    for directions in directed_edges.values():
        if len(directions) != 2:
            continue
        if directions[0] == directions[1]:
            return False
    return True


def _uv_is_exportable(uv: np.ndarray, panel: object) -> bool:
    return (
        isinstance(uv, np.ndarray)
        and uv.ndim == 2
        and uv.shape == (len(tuple(getattr(panel, "vertices"))), 2)
        and bool(np.isfinite(uv).all())
    )


def _compute_distortion(vertices: np.ndarray, panel: object, uv: np.ndarray) -> float:
    edges = tuple(tuple(int(v) for v in edge) for edge in getattr(panel, "edges"))
    if not edges:
        return 0.0
    index = {int(vertex): idx for idx, vertex in enumerate(getattr(panel, "vertices"))}
    distortions: list[float] = []
    for a, b in edges:
        if a not in index or b not in index:
            continue
        length_3d = float(np.linalg.norm(vertices[a] - vertices[b]))
        length_2d = float(np.linalg.norm(uv[index[a]] - uv[index[b]]))
        if length_3d > 1e-12:
            distortions.append(abs(length_2d - length_3d) / length_3d)
    return float(sum(distortions) / len(distortions)) if distortions else 0.0


def _count_foldovers(panel: object, uv: np.ndarray) -> int:
    faces = tuple(tuple(int(v) for v in face) for face in getattr(panel, "faces"))
    index = {int(vertex): idx for idx, vertex in enumerate(getattr(panel, "vertices"))}
    signed_areas: list[float] = []
    for face in faces:
        try:
            a, b, c = (uv[index[vertex]] for vertex in face)
        except KeyError:
            continue
        area = 0.5 * float(np.cross(b - a, c - a))
        if abs(area) > 1e-12:
            signed_areas.append(area)
    if not signed_areas:
        return 0
    positive = sum(1 for area in signed_areas if area > 0.0)
    negative = sum(1 for area in signed_areas if area < 0.0)
    return int(min(positive, negative))


def _boundary_deviation(vertices: np.ndarray, panel: object, uv: np.ndarray) -> float:
    return _compute_distortion(vertices, panel, uv)


def _island_fragmentation(panel: object) -> int:
    faces = tuple(tuple(int(v) for v in face) for face in getattr(panel, "faces"))
    return 0 if faces else 1


def _has_triangle_topology(panel: object) -> bool:
    return bool(tuple(getattr(panel, "faces")))


def _score(
    *,
    distortion: float,
    foldovers: int,
    boundary_deviation: float,
    island_fragmentation: int,
    exportable: bool,
) -> float:
    return float(
        distortion
        + 10.0 * foldovers
        + 0.5 * boundary_deviation
        + 0.25 * island_fragmentation
        + (0.0 if exportable else 100.0)
    )


def _competition_blockers(candidates: Sequence[SerializationCandidateReceipt]) -> list[str]:
    blockers = ["no_serialization_backend_promoted"]
    for candidate in candidates:
        blockers.extend(candidate.blockers)
    return list(dict.fromkeys(blockers))


__all__ = [
    "PANEL_SERIALIZATION_BACKENDS",
    "PANEL_SERIALIZATION_SCHEMA",
    "SerializationCandidateReceipt",
    "XATLAS_BACKEND",
    "build_panel_serialization_competition_receipt",
    "panel_chart_diagnostics",
    "select_serialization_candidate",
    "serialize_panel",
]
