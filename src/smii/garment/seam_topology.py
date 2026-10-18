"""Global seam topology and edge classification for full-suit garment construction."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .pattern_model import BoundaryKind, GarmentRegion, PatternChart, SeamConnection, SeamSide

Edge = tuple[int, int]


def _normalize_edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


class SeamTopologyExtractor:
    """Classifies mesh edges into interior, seam cuts, and intentional openings."""

    def __init__(self, allowance_mm: float = 10.0, default_seam_type: str = "overlock") -> None:
        self.allowance_mm = allowance_mm
        self.default_seam_type = default_seam_type

    def extract_seams(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        regions: Sequence[GarmentRegion],
    ) -> tuple[tuple[SeamConnection, ...], dict[str, BoundaryKind]]:
        """Extract all seam connections and boundary classifications across regions."""
        tri_faces = np.asarray(faces, dtype=np.int64)

        # Map each face ID to its region ID
        face_to_region: dict[int, str] = {}
        for region in regions:
            for fid in region.source_face_ids:
                face_to_region[int(fid)] = region.region_id

        # Index every edge across all faces: edge -> list of (face_id, region_id)
        edge_incidents: dict[Edge, list[tuple[int, str]]] = {}
        for fid, face in enumerate(tri_faces):
            r_id = face_to_region.get(fid, "")
            v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
            for e in (_normalize_edge(v0, v1), _normalize_edge(v1, v2), _normalize_edge(v2, v0)):
                edge_incidents.setdefault(e, []).append((fid, r_id))

        # Group seam edges by region pair
        pair_to_edges: dict[tuple[str, str], list[Edge]] = {}
        boundary_kinds: dict[str, BoundaryKind] = {}

        for edge, incidents in edge_incidents.items():
            if len(incidents) == 1:
                # Genuine boundary / opening edge of the mesh
                r_id = incidents[0][1]
                b_name = f"opening_{r_id}_{edge[0]}_{edge[1]}"
                boundary_kinds[b_name] = BoundaryKind.OPENING
            elif len(incidents) == 2:
                r1, r2 = incidents[0][1], incidents[1][1]
                if r1 != r2:
                    # Seam edge between two distinct regions
                    pair = (r1, r2) if r1 < r2 else (r2, r1)
                    pair_to_edges.setdefault(pair, []).append(edge)

        # Build SeamConnection for each region pair with shared seam edges
        seams: list[SeamConnection] = []
        for (r1, r2), edge_list in sorted(pair_to_edges.items()):
            seam_id = f"seam_{r1}_{r2}"
            boundary_kinds[seam_id] = BoundaryKind.SEAM

            # Unique ordered vertices in this seam
            seam_verts_set: set[int] = set()
            for u, v in edge_list:
                seam_verts_set.add(u)
                seam_verts_set.add(v)
            seam_verts = tuple(sorted(seam_verts_set))

            # Encode edge IDs as flat tuple of alternating vertices
            flat_edge_ids: list[int] = []
            for u, v in edge_list:
                flat_edge_ids.extend([u, v])

            left_side = SeamSide(
                chart_id=r1,
                source_edge_ids=tuple(flat_edge_ids),
                boundary_vertex_ids=seam_verts,
                reversed=False,
            )
            right_side = SeamSide(
                chart_id=r2,
                source_edge_ids=tuple(flat_edge_ids),
                boundary_vertex_ids=seam_verts,
                reversed=True,
            )

            seams.append(
                SeamConnection(
                    seam_id=seam_id,
                    left=left_side,
                    right=right_side,
                    allowance_mm=self.allowance_mm,
                    seam_type=self.default_seam_type,
                )
            )

        return tuple(seams), boundary_kinds

    @staticmethod
    def compute_3d_seam_lengths(
        vertices: np.ndarray,
        seam: SeamConnection,
    ) -> tuple[float, float, float]:
        """Compute 3D arc-length for left and right seam sides, and relative mismatch E_seam."""
        verts = np.asarray(vertices, dtype=np.float64)

        def _side_len(side: SeamSide) -> float:
            total = 0.0
            edges = side.source_edge_ids
            for i in range(0, len(edges) - 1, 2):
                u, v = edges[i], edges[i + 1]
                total += float(np.linalg.norm(verts[u] - verts[v]))
            return total

        l_left = _side_len(seam.left)
        l_right = _side_len(seam.right)
        max_l = max(l_left, l_right, 1e-12)
        e_seam = abs(l_left - l_right) / max_l
        return l_left, l_right, e_seam

    @staticmethod
    def compute_uv_seam_error(
        seam: SeamConnection,
        left_chart: PatternChart,
        right_chart: PatternChart,
    ) -> float:
        """Compute relative length mismatch E_seam in 2D flattened fabrication coordinates (mm)."""

        def _uv_length(side: SeamSide, chart: PatternChart) -> float:
            edge_to_len: dict[tuple[int, int], float] = {}
            for tri in chart.local_faces:
                s0 = int(chart.source_vertex_ids[tri[0]])
                s1 = int(chart.source_vertex_ids[tri[1]])
                s2 = int(chart.source_vertex_ids[tri[2]])
                uv0, uv1, uv2 = chart.uv_mm[tri[0]], chart.uv_mm[tri[1]], chart.uv_mm[tri[2]]
                for a, b, p1, p2 in ((s0, s1, uv0, uv1), (s1, s2, uv1, uv2), (s2, s0, uv2, uv0)):
                    norm = (min(a, b), max(a, b))
                    edge_to_len[norm] = float(np.linalg.norm(p1 - p2))

            total = 0.0
            edges = side.source_edge_ids
            for i in range(0, len(edges) - 1, 2):
                u, v = int(edges[i]), int(edges[i + 1])
                norm = (min(u, v), max(u, v))
                if norm in edge_to_len:
                    total += edge_to_len[norm]
            return total

        l_left = _uv_length(seam.left, left_chart)
        l_right = _uv_length(seam.right, right_chart)
        max_l = max(l_left, l_right, 1e-12)
        return abs(l_left - l_right) / max_l


__all__ = [
    "Edge",
    "SeamTopologyExtractor",
]
