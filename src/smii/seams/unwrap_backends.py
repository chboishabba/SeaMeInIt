"""Panel unwrap backends used by the receipted Gate 6 unwrapper."""

from __future__ import annotations

import numpy as np

Edge = tuple[int, int]
Face = tuple[int, int, int]

BOOTSTRAP_BACKEND = "bootstrap_projection"
LSCM_BACKEND = "lscm"
UNWRAP_BACKENDS = (BOOTSTRAP_BACKEND, LSCM_BACKEND)


def unwrap_panel_vertices(
    vertices: np.ndarray,
    *,
    panel_vertices: tuple[int, ...],
    panel_faces: tuple[Face, ...],
    method: str,
) -> np.ndarray:
    """Return UV coordinates ordered like ``panel_vertices``."""

    if method == BOOTSTRAP_BACKEND:
        return _bootstrap_projection(vertices, panel_vertices)
    if method == LSCM_BACKEND:
        return _lscm_unwrap(vertices, panel_vertices=panel_vertices, panel_faces=panel_faces)
    raise ValueError(f"Unknown unwrap backend '{method}'.")


def _bootstrap_projection(vertices: np.ndarray, panel_vertices: tuple[int, ...]) -> np.ndarray:
    coords = np.asarray(vertices[list(panel_vertices)], dtype=float)
    if len(coords) == 0:
        return np.empty((0, 2), dtype=float)
    centered = coords - coords.mean(axis=0)
    if len(coords) == 1:
        return np.zeros((1, 2), dtype=float)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt[:2]
    if axes.shape[0] < 2:
        axes = np.vstack([axes, np.array([[0.0, 1.0, 0.0]])])
    return centered @ axes.T


def _lscm_unwrap(
    vertices: np.ndarray,
    *,
    panel_vertices: tuple[int, ...],
    panel_faces: tuple[Face, ...],
) -> np.ndarray:
    local_vertices = np.asarray(vertices[list(panel_vertices)], dtype=float)
    if len(local_vertices) < 3:
        return _bootstrap_projection(vertices, panel_vertices)
    local_index = {vertex: idx for idx, vertex in enumerate(panel_vertices)}
    local_faces = [
        (local_index[a], local_index[b], local_index[c])
        for a, b, c in panel_faces
        if a in local_index and b in local_index and c in local_index
    ]
    if not local_faces:
        return _bootstrap_projection(vertices, panel_vertices)
    return _solve_lscm(local_vertices, np.asarray(local_faces, dtype=int))


def _triangle_complex(vertices: np.ndarray, i: int, j: int, k: int) -> tuple[complex, complex]:
    p_i = vertices[i]
    p_j = vertices[j]
    p_k = vertices[k]
    e1 = p_j - p_i
    e2 = p_k - p_i
    normal = np.cross(e1, e2)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        tangent = np.array([1.0, 0.0, 0.0])
        bitangent = np.array([0.0, 1.0, 0.0])
    else:
        tangent = e1 / float(np.linalg.norm(e1) or 1.0)
        bitangent = np.cross(normal, tangent)
        bitangent /= float(np.linalg.norm(bitangent) or 1.0)
    z1 = complex(float(np.dot(e1, tangent)), float(np.dot(e1, bitangent)))
    z2 = complex(float(np.dot(e2, tangent)), float(np.dot(e2, bitangent)))
    return z1, z2


def _choose_anchor_vertices(vertices: np.ndarray) -> tuple[int, int]:
    max_distance = -1.0
    anchor_pair = (0, 1 if len(vertices) > 1 else 0)
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            distance = float(np.linalg.norm(vertices[i] - vertices[j]))
            if distance > max_distance:
                max_distance = distance
                anchor_pair = (i, j)
    return anchor_pair


def _solve_lscm(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertex_count = int(len(vertices))
    if vertex_count < 2:
        return np.zeros((vertex_count, 2), dtype=float)
    i0, i1 = _choose_anchor_vertices(vertices)
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for tri in faces:
        i, j, k = (int(tri[0]), int(tri[1]), int(tri[2]))
        z1, z2 = _triangle_complex(vertices, i, j, k)
        coeffs = {i: -z1 + z2, j: -z2, k: z1}
        row_real = np.zeros(2 * vertex_count, dtype=float)
        row_imag = np.zeros(2 * vertex_count, dtype=float)
        for index, coeff in coeffs.items():
            a = coeff.real
            b = coeff.imag
            row_real[2 * index] += a
            row_real[2 * index + 1] += -b
            row_imag[2 * index] += b
            row_imag[2 * index + 1] += a
        rows.append(row_real)
        rows.append(row_imag)
        rhs.extend([0.0, 0.0])
    for index, target in ((i0, (0.0, 0.0)), (i1, (1.0, 0.0))):
        row_u = np.zeros(2 * vertex_count, dtype=float)
        row_v = np.zeros(2 * vertex_count, dtype=float)
        row_u[2 * index] = 1.0
        row_v[2 * index + 1] = 1.0
        rows.append(row_u)
        rows.append(row_v)
        rhs.extend([target[0], target[1]])
    matrix = np.vstack(rows)
    solution, *_ = np.linalg.lstsq(matrix, np.array(rhs, dtype=float), rcond=None)
    uv = solution.reshape(vertex_count, 2)
    edge_length = float(np.linalg.norm(vertices[i1] - vertices[i0])) or 1.0
    return uv * edge_length


__all__ = [
    "BOOTSTRAP_BACKEND",
    "LSCM_BACKEND",
    "UNWRAP_BACKENDS",
    "unwrap_panel_vertices",
]
