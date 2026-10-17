"""Conformal unwrapping and boundary loop extraction for garment pattern charts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .pattern_model import IntArray, PatternChart

Face = tuple[int, int, int]
Edge = tuple[int, int]


@dataclass(frozen=True)
class UnwrapConfig:
    """Configuration for conformal chart unwrapping."""

    method: str = "lscm"  # "lscm" or "bootstrap"
    anchor_pin_weight: float = 1000.0
    solver_atol: float = 1e-6
    solver_btol: float = 1e-6


def extract_boundary_loops(local_faces: np.ndarray) -> tuple[IntArray, ...]:
    """Extract ordered boundary loop cycles from a triangle mesh patch."""
    faces = np.asarray(local_faces, dtype=np.int64)
    if len(faces) == 0:
        return ()

    # Count undirected occurrences to find boundary half-edges
    edge_counts: dict[Edge, int] = {}
    for tri in faces:
        v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            norm = (a, b) if a < b else (b, a)
            edge_counts[norm] = edge_counts.get(norm, 0) + 1

    # Keep directed half-edges of boundary edges
    adj: dict[int, list[int]] = {}
    for tri in faces:
        v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            norm = (a, b) if a < b else (b, a)
            if edge_counts[norm] == 1:
                adj.setdefault(a, []).append(b)

    # Trace ordered cycles
    visited: set[int] = set()
    loops: list[IntArray] = []

    for start_node in list(adj.keys()):
        if start_node in visited:
            continue
        loop_nodes: list[int] = [start_node]
        visited.add(start_node)
        curr = start_node
        while True:
            next_candidates = [nxt for nxt in adj.get(curr, []) if nxt not in visited]
            if not next_candidates:
                break
            nxt = next_candidates[0]
            loop_nodes.append(nxt)
            visited.add(nxt)
            curr = nxt
        if len(loop_nodes) >= 3:
            loops.append(np.asarray(loop_nodes, dtype=np.int64))

    return tuple(loops)


def count_uv_foldovers(uv: np.ndarray, faces: np.ndarray) -> int:
    """Count inverted triangles (signed 2D area <= 0)."""
    p0 = uv[faces[:, 0]]
    p1 = uv[faces[:, 1]]
    p2 = uv[faces[:, 2]]
    # 2D cross product: (p1_x - p0_x)*(p2_y - p0_y) - (p1_y - p0_y)*(p2_x - p0_x)
    cross_2d = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (
        p2[:, 0] - p0[:, 0]
    )
    return int(np.sum(cross_2d <= 0.0))


def solve_sparse_lscm(
    local_verts: np.ndarray,
    local_faces: np.ndarray,
    config: UnwrapConfig | None = None,
) -> np.ndarray:
    """Solve Least Squares Conformal Mapping using sparse normal equations."""
    cfg = config or UnwrapConfig()
    nV = len(local_verts)
    nF = len(local_faces)
    if nV < 3 or nF == 0:
        return np.zeros((nV, 2), dtype=np.float64)

    # Select two anchor points along maximum coordinate spread
    coord_spans = local_verts.max(axis=0) - local_verts.min(axis=0)
    primary_axis = int(np.argmax(coord_spans))
    i0 = int(np.argmin(local_verts[:, primary_axis]))
    i1 = int(np.argmax(local_verts[:, primary_axis]))
    if i0 == i1:
        i1 = (i0 + 1) % nV

    rows: list[int] = []
    cols: list[int] = []
    data_vals: list[float] = []
    rhs = np.zeros(2 * nF + 4, dtype=np.float64)

    for row_idx, tri in enumerate(local_faces):
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        p_i, p_j, p_k = local_verts[i], local_verts[j], local_verts[k]
        e1 = p_j - p_i
        e2 = p_k - p_i
        normal = np.cross(e1, e2)
        norm = float(np.linalg.norm(normal))
        if norm > 1e-12:
            tangent = e1 / (float(np.linalg.norm(e1)) or 1.0)
            bitangent = np.cross(normal, tangent)
            bitangent /= float(np.linalg.norm(bitangent)) or 1.0
        else:
            tangent = np.array([1.0, 0.0, 0.0])
            bitangent = np.array([0.0, 1.0, 0.0])

        z1 = complex(float(np.dot(e1, tangent)), float(np.dot(e1, bitangent)))
        z2 = complex(float(np.dot(e2, tangent)), float(np.dot(e2, bitangent)))
        coeffs = {i: -z1 + z2, j: -z2, k: z1}

        r_real = 2 * row_idx
        r_imag = 2 * row_idx + 1
        for idx, c in coeffs.items():
            # Real equation: a * u - b * v
            rows.extend([r_real, r_real])
            cols.extend([2 * idx, 2 * idx + 1])
            data_vals.extend([c.real, -c.imag])
            # Imag equation: b * u + a * v
            rows.extend([r_imag, r_imag])
            cols.extend([2 * idx, 2 * idx + 1])
            data_vals.extend([c.imag, c.real])

    # Pin anchor vertices with penalty weights
    pin_offset = 2 * nF
    weight = cfg.anchor_pin_weight

    # u(i0) = 0, v(i0) = 0
    rows.extend([pin_offset, pin_offset + 1])
    cols.extend([2 * i0, 2 * i0 + 1])
    data_vals.extend([weight, weight])

    # u(i1) = 1, v(i1) = 0
    rows.extend([pin_offset + 2, pin_offset + 3])
    cols.extend([2 * i1, 2 * i1 + 1])
    data_vals.extend([weight, weight])
    rhs[pin_offset + 2] = weight

    A = sp.csr_matrix((data_vals, (rows, cols)), shape=(pin_offset + 4, 2 * nV))
    sol = spla.lsqr(A, rhs, atol=cfg.solver_atol, btol=cfg.solver_btol)
    uv = sol[0].reshape((nV, 2))

    # Scale back to physical Euclidean units (metres) based on anchor distance
    anchor_dist_3d = float(np.linalg.norm(local_verts[i1] - local_verts[i0])) or 1.0
    return uv * anchor_dist_3d


def unwrap_chart(
    vertices: np.ndarray,
    faces: np.ndarray,
    region_id: str,
    source_face_ids: Sequence[int],
    config: UnwrapConfig | None = None,
) -> PatternChart:
    """Unwrap a single region into a physical mm-scaled PatternChart."""
    verts = np.asarray(vertices, dtype=np.float64)
    all_faces = np.asarray(faces, dtype=np.int64)
    fids = np.asarray(source_face_ids, dtype=np.int64)

    sub_faces = all_faces[fids]
    unique_v, inv_map = np.unique(sub_faces, return_inverse=True)
    local_verts = verts[unique_v]
    local_faces = inv_map.reshape(sub_faces.shape).astype(np.int64)

    cfg = config or UnwrapConfig()
    if cfg.method == "lscm" and len(local_faces) >= 1:
        uv_m = solve_sparse_lscm(local_verts, local_faces, cfg)
    else:
        # SVD PCA projection fallback
        centered = local_verts - local_verts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        uv_m = centered @ vt[:2].T

    # Scale from metres to millimetres (1 m = 1000 mm)
    uv_mm = uv_m * 1000.0

    # Extract ordered boundary loops
    loops = extract_boundary_loops(local_faces)

    return PatternChart(
        chart_id=region_id,
        region_id=region_id,
        source_face_ids=fids,
        uv_mm=uv_mm,
        local_faces=local_faces,
        boundary_loops=loops,
        source_vertex_ids=unique_v.astype(np.int64),
    )


__all__ = [
    "UnwrapConfig",
    "count_uv_foldovers",
    "extract_boundary_loops",
    "solve_sparse_lscm",
    "unwrap_chart",
]
