"""Full-suit garment partitioner integrating region atlas, seam topology, and conformal unwrapping."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .chart_unwrap import UnwrapConfig, unwrap_chart
from .pattern_model import FullSuitPattern, PatternChart
from .region_atlas import RegionAtlas, RegionAtlasConfig
from .seam_topology import SeamTopologyExtractor


def _compute_mesh_sha256(vertices: np.ndarray, faces: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(vertices).tobytes())
    h.update(np.ascontiguousarray(faces).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class FullSuitPartitionConfig:
    atlas_config: RegionAtlasConfig | None = None
    unwrap_config: UnwrapConfig | None = None
    allowance_mm: float = 10.0
    seam_type: str = "overlock"


class FullSuitPartitioner:
    """End-to-end full-suit partitioner generating 2D pattern charts and global seam topology."""

    def __init__(self, config: FullSuitPartitionConfig | None = None) -> None:
        self.config = config or FullSuitPartitionConfig()
        self.atlas = RegionAtlas(self.config.atlas_config)
        self.seam_extractor = SeamTopologyExtractor(
            allowance_mm=self.config.allowance_mm,
            default_seam_type=self.config.seam_type,
        )

    def partition(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[FullSuitPattern, dict[str, Any]]:
        """Transform 3D body geometry into a validated FullSuitPattern."""
        verts = np.asarray(vertices, dtype=np.float64)
        tri_faces = np.asarray(faces, dtype=np.int64)
        mesh_sha = _compute_mesh_sha256(verts, tri_faces)

        # 1. Segment mesh faces into garment regions
        regions = self.atlas.build_regions(verts, tri_faces)

        # 2. Extract global seam graph and boundary edge classifications
        seams, boundary_kinds = self.seam_extractor.extract_seams(verts, tri_faces, regions)

        # 3. Conformal unwrap for each chart with physical scale in mm
        charts: list[PatternChart] = []
        chart_map: dict[str, PatternChart] = {}
        for r in regions:
            chart = unwrap_chart(
                verts,
                tri_faces,
                region_id=r.region_id,
                source_face_ids=r.source_face_ids,
                config=self.config.unwrap_config,
            )
            charts.append(chart)
            chart_map[r.region_id] = chart

        # 4. Measure seam arc-length errors across connected charts
        seam_diagnostics: list[dict[str, Any]] = []
        for s in seams:
            l_chart = chart_map.get(s.left.chart_id)
            r_chart = chart_map.get(s.right.chart_id)
            l3d, r3d, e_3d = self.seam_extractor.compute_3d_seam_lengths(verts, s)
            e_uv = (
                self.seam_extractor.compute_uv_seam_error(s, l_chart, r_chart)
                if l_chart and r_chart
                else 0.0
            )
            seam_diagnostics.append(
                {
                    "seam_id": s.seam_id,
                    "left_chart": s.left.chart_id,
                    "right_chart": s.right.chart_id,
                    "length_left_3d_mm": float(l3d * 1000.0),
                    "length_right_3d_mm": float(r3d * 1000.0),
                    "error_3d_relative": float(e_3d),
                    "error_uv_relative": float(e_uv),
                }
            )

        pattern = FullSuitPattern(
            source_mesh_sha256=mesh_sha,
            regions=regions,
            charts=tuple(charts),
            seams=seams,
            boundary_kinds=boundary_kinds,
        )

        diagnostics = {
            "source_mesh_sha256": mesh_sha,
            "region_count": len(regions),
            "chart_count": len(charts),
            "seam_count": len(seams),
            "total_faces": len(tri_faces),
            "total_vertices": len(verts),
            "seam_diagnostics": seam_diagnostics,
        }

        return pattern, diagnostics


def save_full_suit_pattern(
    pattern: FullSuitPattern,
    npz_path: Path | str,
) -> None:
    """Save pattern charts and geometry arrays into a compressed NPZ bundle."""
    target = Path(npz_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {
        "source_mesh_sha256": np.array([pattern.source_mesh_sha256]),
        "chart_count": np.array([len(pattern.charts)]),
    }

    for i, c in enumerate(pattern.charts):
        prefix = f"chart_{i}_{c.chart_id}_"
        arrays[f"{prefix}uv_mm"] = c.uv_mm
        arrays[f"{prefix}source_face_ids"] = c.source_face_ids
        arrays[f"{prefix}source_vertex_ids"] = c.source_vertex_ids
        arrays[f"{prefix}local_faces"] = c.local_faces

    np.savez_compressed(target, **arrays)


__all__ = [
    "FullSuitPartitionConfig",
    "FullSuitPartitioner",
    "save_full_suit_pattern",
]
