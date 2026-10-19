"""Assembly of the discrete cotangent Laplace-Beltrami and lumped mass operators.

Provides sparse stiffness (L) and mass (M) matrices on triangular surface meshes
following the standard discrete differential geometry formulation:
    L * phi = lambda * M * phi
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CotangentLaplacianResult:
    """Stiffness and mass matrices with geometric diagnostics."""

    stiffness: sp.csr_matrix
    mass: sp.csr_matrix
    mass_diagonal: NDArray[np.float64]
    vertex_count: int
    face_count: int
    total_area: float
    degenerate_face_count: int
    negative_weight_count: int
    isolated_vertex_count: int


def build_cotangent_laplacian(
    vertices: NDArray[np.float64] | Sequence[Sequence[float]],
    faces: NDArray[np.int64] | Sequence[Sequence[int]],
    *,
    area_eps: float = 1e-12,
) -> CotangentLaplacianResult:
    """Assemble discrete cotangent Laplace-Beltrami stiffness and lumped mass matrices.

    Args:
        vertices: Array of shape (N, 3) vertex coordinates.
        faces: Array of shape (F, 3) vertex indices for triangles.
        area_eps: Threshold below which a triangle is treated as degenerate.

    Returns:
        CotangentLaplacianResult with sparse stiffness (L), mass (M), and diagnostics.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"Vertices must have shape (N, 3); got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"Faces must have shape (F, 3); got {f.shape}")
    if not np.isfinite(v).all():
        raise ValueError("Vertex coordinates contain NaN or Inf values")

    n_vertices = len(v)
    n_faces = len(f)

    if n_faces == 0 or n_vertices == 0:
        raise ValueError("Mesh must contain at least one vertex and one face")

    # Vertex indices for all triangles
    i0 = f[:, 0]
    i1 = f[:, 1]
    i2 = f[:, 2]

    # Edge vectors
    # e0 is opposite to vertex 0 (from v1 to v2)
    # e1 is opposite to vertex 1 (from v2 to v0)
    # e2 is opposite to vertex 2 (from v0 to v1)
    v0 = v[i0]
    v1 = v[i1]
    v2 = v[i2]

    # Vectors originating from vertices
    # at vertex 0: vectors to v1 and v2
    v01 = v1 - v0
    v02 = v2 - v0
    # at vertex 1: vectors to v2 and v0
    v12 = v2 - v1
    v10 = v0 - v1
    # at vertex 2: vectors to v0 and v1
    v20 = v0 - v2
    v21 = v1 - v2

    # Normal vectors and face areas (cross product)
    cross_normals = np.cross(v01, v02)
    two_areas = np.linalg.norm(cross_normals, axis=1)

    # Detect degenerate faces
    degenerate_mask = two_areas <= area_eps
    degenerate_count = int(np.sum(degenerate_mask))

    # Safe two_areas for division
    safe_two_areas = np.where(degenerate_mask, 1.0, two_areas)

    # Cotangents of triangle interior angles:
    # cot(angle) = (u . v) / ||u x v|| = (u . v) / (2 * Area)
    # cot0 opposite to edge (1, 2)
    cot0 = np.sum(v01 * v02, axis=1) / safe_two_areas
    # cot1 opposite to edge (2, 0)
    cot1 = np.sum(v12 * v10, axis=1) / safe_two_areas
    # cot2 opposite to edge (0, 1)
    cot2 = np.sum(v20 * v21, axis=1) / safe_two_areas

    # Zero out cotangents for degenerate faces
    cot0 = np.where(degenerate_mask, 0.0, cot0)
    cot1 = np.where(degenerate_mask, 0.0, cot1)
    cot2 = np.where(degenerate_mask, 0.0, cot2)

    # Weight per half-edge contribution is 0.5 * cot
    w0 = 0.5 * cot0  # edge (1, 2)
    w1 = 0.5 * cot1  # edge (2, 0)
    w2 = 0.5 * cot2  # edge (0, 1)

    negative_weight_count = int(np.sum(w0 < 0) + np.sum(w1 < 0) + np.sum(w2 < 0))

    # Assemble off-diagonal entries
    # Each edge appears symmetrically
    rows = np.concatenate([i1, i2, i2, i0, i0, i1])
    cols = np.concatenate([i2, i1, i0, i2, i1, i0])
    weights = np.concatenate([w0, w0, w1, w1, w2, w2])

    # Build adjacency with summed weights
    # L_ij = -w_ij for off-diagonals
    off_diag = sp.coo_matrix((-weights, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
    off_diag.sum_duplicates()

    # Diagonal entries L_ii = sum_{j != i} w_ij = -sum(off_diagonals)
    # Using off_diag row sums:
    diag_values = -np.asarray(off_diag.sum(axis=1)).flatten()

    diag_matrix = sp.diags(diag_values, 0, shape=(n_vertices, n_vertices), format="csr")
    stiffness = (diag_matrix + off_diag).tocsr()

    # Assemble lumped barycentric mass matrix M
    # Each vertex of triangle gets 1/3 of triangle area
    face_areas = np.where(degenerate_mask, 0.0, 0.5 * two_areas)
    total_area = float(np.sum(face_areas))

    one_third_areas = face_areas / 3.0
    mass_diag = np.zeros(n_vertices, dtype=np.float64)
    np.add.at(mass_diag, i0, one_third_areas)
    np.add.at(mass_diag, i1, one_third_areas)
    np.add.at(mass_diag, i2, one_third_areas)

    isolated_count = int(np.sum(mass_diag <= area_eps))

    # Construct sparse diagonal mass matrix
    mass = sp.diags(mass_diag, 0, shape=(n_vertices, n_vertices), format="csr")

    return CotangentLaplacianResult(
        stiffness=stiffness,
        mass=mass,
        mass_diagonal=mass_diag,
        vertex_count=n_vertices,
        face_count=n_faces,
        total_area=total_area,
        degenerate_face_count=degenerate_count,
        negative_weight_count=negative_weight_count,
        isolated_vertex_count=isolated_count,
    )
