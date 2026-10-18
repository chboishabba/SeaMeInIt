"""Full-suit garment partitioner integrating region atlas, seam topology, and conformal unwrapping."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .chart_unwrap import (
    UnwrapConfig,
    extract_boundary_loops,
    solve_sparse_lscm,
    unwrap_chart,
)
from .pattern_model import FullSuitPattern, PatternChart
from .region_atlas import RegionAtlas, RegionAtlasConfig
from .rom_relief import minimum_rom_relief, open_mesh_along_route
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
    apply_rom_relief: bool = True


class FullSuitPartitioner:
    """End-to-end full-suit partitioner generating 2D pattern charts and global seam topology."""

    def __init__(self, config: FullSuitPartitionConfig | None = None) -> None:
        self.config = config or FullSuitPartitionConfig()
        atlas_cfg = self.config.atlas_config or RegionAtlasConfig(z_coronal=-0.035)
        self.atlas = RegionAtlas(atlas_cfg)
        self.seam_extractor = SeamTopologyExtractor(
            allowance_mm=self.config.allowance_mm,
            default_seam_type=self.config.seam_type,
        )

    def partition(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_costs: np.ndarray | None = None,
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
            sub_faces = tri_faces[r.source_face_ids]
            unique_v, inv_map = np.unique(sub_faces, return_inverse=True)
            local_verts = verts[unique_v]
            local_faces = inv_map.reshape(sub_faces.shape).astype(np.int64)

            chart: PatternChart | None = None
            if (
                self.config.apply_rom_relief
                and ("sleeve" in r.region_id or "leg" in r.region_id)
                and len(local_faces) >= 4
            ):
                try:
                    loops = extract_boundary_loops(local_faces)
                    if loops:
                        start_v = int(loops[0][0])
                        if "sleeve_left" in r.region_id:
                            end_v = int(np.argmax(local_verts[:, 0]))
                        elif "sleeve_right" in r.region_id:
                            end_v = int(np.argmin(local_verts[:, 0]))
                        else:
                            end_v = int(np.argmin(local_verts[:, 1]))

                        if start_v != end_v:
                            if vertex_costs is not None:
                                costs = np.asarray(vertex_costs[unique_v], dtype=np.float64)
                            else:
                                costs = np.zeros(len(local_verts), dtype=np.float64)

                            route = minimum_rom_relief(
                                local_verts, local_faces, costs, start_v, end_v
                            )
                            new_verts, new_faces, source_sub = open_mesh_along_route(
                                local_verts, local_faces, route
                            )
                            cfg_u = self.config.unwrap_config or UnwrapConfig()
                            uv_m = solve_sparse_lscm(new_verts, new_faces, cfg_u)
                            uv_mm = uv_m * 1000.0
                            new_loops = extract_boundary_loops(new_faces)
                            actual_source_ids = unique_v[source_sub].astype(np.int64)
                            chart = PatternChart(
                                chart_id=r.region_id,
                                region_id=r.region_id,
                                source_face_ids=r.source_face_ids,
                                uv_mm=uv_mm,
                                local_faces=new_faces,
                                boundary_loops=new_loops,
                                source_vertex_ids=actual_source_ids,
                            )
                except Exception:
                    chart = None

            if chart is None:
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
        arrays[f"{prefix}features_json"] = np.array(
            [json.dumps([feature.to_dict() for feature in c.features])]
        )

    np.savez_compressed(target, **arrays)


__all__ = [
    "FullSuitPartitionConfig",
    "FullSuitPartitioner",
    "save_full_suit_pattern",
]
