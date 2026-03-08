"""Repair helpers for fitted body meshes."""

from __future__ import annotations

import inspect
import warnings

import numpy as np

__all__ = ["repair_mesh_with_pymeshfix"]


def _supported_repair_kwargs(meshfix: object, *, verbose: bool, max_iters: int | None) -> dict[str, object]:
    """Filter optional repair kwargs against the installed PyMeshFix API."""

    try:
        signature = inspect.signature(meshfix.repair)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return {}

    supported: dict[str, object] = {}
    if "verbose" in signature.parameters:
        supported["verbose"] = verbose
    if max_iters is not None and "max_iters" in signature.parameters:
        supported["max_iters"] = max_iters
    return supported


def _split_components(mesh: object) -> list[object]:
    import trimesh

    if len(mesh.faces) == 0:
        return []

    face_indices = np.arange(len(mesh.faces))
    groups = trimesh.graph.connected_components(mesh.face_adjacency, nodes=face_indices)
    components = [
        mesh.submesh([np.asarray(group, dtype=int)], append=True, repair=False)
        for group in groups
    ]
    components.sort(key=lambda component: (component.area, len(component.faces)), reverse=True)
    return components


def _single_boundary_loop(component: object) -> list[int] | None:
    edges = np.asarray(component.edges_sorted, dtype=np.int64)
    if edges.size == 0:
        return None

    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique_edges[counts == 1]
    if len(boundary) == 0:
        return None

    adjacency: dict[int, list[int]] = {}
    for start, end in boundary:
        a = int(start)
        b = int(end)
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return None

    start = min(adjacency)
    loop = [start]
    prev: int | None = None
    current = start
    while True:
        neighbors = adjacency[current]
        nxt = neighbors[0] if neighbors[0] != prev else neighbors[1]
        if nxt == start:
            break
        if nxt in loop:
            return None
        loop.append(nxt)
        prev, current = current, nxt
    return loop if len(loop) == len(adjacency) else None


def _cap_boundary_loop(component: object) -> object | None:
    import trimesh

    loop = _single_boundary_loop(component)
    if loop is None or len(loop) < 3:
        return None

    center = np.asarray(component.vertices[np.asarray(loop, dtype=np.int64)], dtype=np.float64).mean(axis=0)
    center_idx = len(component.vertices)
    cap_faces = np.asarray(
        [[loop[idx], loop[(idx + 1) % len(loop)], center_idx] for idx in range(len(loop))],
        dtype=np.int64,
    )
    capped = trimesh.Trimesh(
        vertices=np.vstack((np.asarray(component.vertices, dtype=np.float64), center)),
        faces=np.vstack((np.asarray(component.faces, dtype=np.int64), cap_faces)),
        process=False,
    )
    trimesh.repair.fix_normals(capped, multibody=False)
    return capped


def repair_mesh_with_pymeshfix(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    verbose: bool = False,
    max_iters: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Attempt to repair a mesh via PyMeshFix.

    Parameters
    ----------
    vertices:
        Vertex array with shape ``(N, 3)``.
    faces:
        Triangular face array with shape ``(M, 3)``.
    verbose:
        Whether PyMeshFix should emit diagnostic output.
    max_iters:
        Optional maximum iteration hint passed to PyMeshFix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        Repaired mesh geometry if PyMeshFix is available and succeeds, otherwise
        ``None``.
    """

    try:
        from pymeshfix import MeshFix
    except ModuleNotFoundError:
        return None

    meshfix = MeshFix(
        np.asarray(vertices, dtype=np.float64, order="C", copy=True),
        np.asarray(faces, dtype=np.int64, order="C", copy=True),
    )
    repair_kwargs = _supported_repair_kwargs(meshfix, verbose=verbose, max_iters=max_iters)

    try:
        meshfix.repair(**repair_kwargs)
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.warn(
            f"PyMeshFix repair failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    raw_vertices = getattr(meshfix, "v", None)
    raw_faces = getattr(meshfix, "f", None)
    if raw_vertices is None or raw_faces is None:
        return None

    repaired_vertices = np.asarray(raw_vertices)
    repaired_faces = np.asarray(raw_faces)
    if repaired_vertices.size == 0 or repaired_faces.size == 0:
        return None

    return repaired_vertices.astype(np.float32, copy=False), repaired_faces.astype(np.int32, copy=False)


def repair_body_mesh_for_export(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    verbose: bool = False,
    max_iters: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Repair a fitted body mesh into a single watertight garment-ready surface.

    This prefers the primary connected component, since stock SMPL-X meshes may
    include small detached eye components that are irrelevant for downstream
    garment generation. If a single open boundary loop remains on the primary
    body surface, cap it to close the mouth opening.
    """

    try:
        import trimesh
    except ModuleNotFoundError:
        return repair_mesh_with_pymeshfix(vertices, faces, verbose=verbose, max_iters=max_iters)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = _split_components(mesh)
    if not components:
        return None

    primary = components[0]
    if primary.is_watertight:
        return (
            np.asarray(primary.vertices, dtype=np.float32),
            np.asarray(primary.faces, dtype=np.int32),
        )

    repaired = repair_mesh_with_pymeshfix(
        np.asarray(primary.vertices),
        np.asarray(primary.faces),
        verbose=verbose,
        max_iters=max_iters,
    )
    if repaired is not None:
        repaired_mesh = trimesh.Trimesh(vertices=repaired[0], faces=repaired[1], process=False)
        if repaired_mesh.is_watertight:
            return repaired

    candidate = primary.copy()
    trimesh.repair.fill_holes(candidate)
    trimesh.repair.fix_normals(candidate, multibody=False)
    if candidate.is_watertight:
        return (
            np.asarray(candidate.vertices, dtype=np.float32),
            np.asarray(candidate.faces, dtype=np.int32),
        )

    capped = _cap_boundary_loop(candidate)
    if capped is not None and capped.is_watertight:
        return (
            np.asarray(capped.vertices, dtype=np.float32),
            np.asarray(capped.faces, dtype=np.int32),
        )

    return None
