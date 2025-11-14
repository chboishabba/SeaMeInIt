"""Repair helpers for fitted body meshes."""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["repair_mesh_with_pymeshfix"]


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
    repair_kwargs: dict[str, object] = {"verbose": verbose}
    if max_iters is not None:
        repair_kwargs["max_iters"] = max_iters

    try:
        meshfix.repair(**repair_kwargs)
    except TypeError:
        meshfix.repair(verbose=verbose)
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.warn(
            f"PyMeshFix repair failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    repaired_vertices = np.asarray(getattr(meshfix, "v", None))
    repaired_faces = np.asarray(getattr(meshfix, "f", None))
    if repaired_vertices.size == 0 or repaired_faces.size == 0:
        return None

    return repaired_vertices.astype(np.float32, copy=False), repaired_faces.astype(np.int32, copy=False)
