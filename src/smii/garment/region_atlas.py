"""Garment region atlas partitioning 3D body meshes into valid garment panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .pattern_model import GarmentRegion, IntArray


@dataclass(frozen=True)
class RegionAtlasConfig:
    """Configurable geometric thresholds for segmenting a 3D avatar body into garment regions."""

    y_crotch: float = -0.35
    x_shoulder: float = 0.22
    y_collar_max: float = 0.35
    z_coronal: float = 0.0
    default_material: str = "performance_stretch_textile"


class RegionAtlas:
    """Partitioning engine assigning 100% of body mesh faces to garment regions."""

    def __init__(self, config: RegionAtlasConfig | None = None) -> None:
        self.config = config or RegionAtlasConfig()

    def build_regions(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[GarmentRegion, ...]:
        """Partition mesh faces into canonical garment regions with 100% face coverage."""
        verts = np.asarray(vertices, dtype=np.float64)
        tri_faces = np.asarray(faces, dtype=np.int64)

        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(f"vertices must be (N, 3), got {verts.shape}")
        if tri_faces.ndim != 2 or tri_faces.shape[1] != 3:
            raise ValueError(f"faces must be (M, 3), got {tri_faces.shape}")
        if len(tri_faces) == 0:
            raise ValueError("Mesh has no faces.")

        # Compute 3D centroid for every face
        centroids = verts[tri_faces].mean(axis=1)
        total_faces = len(tri_faces)

        # Boolean masks based on geometric coordinates
        cfg = self.config
        is_legs = centroids[:, 1] < cfg.y_crotch
        is_leg_left = is_legs & (centroids[:, 0] >= 0.0)
        is_leg_right = is_legs & (centroids[:, 0] < 0.0)

        is_upper = ~is_legs
        is_sleeve_left = (
            is_upper & (centroids[:, 0] > cfg.x_shoulder) & (centroids[:, 1] < cfg.y_collar_max)
        )
        is_sleeve_right = (
            is_upper & (centroids[:, 0] < -cfg.x_shoulder) & (centroids[:, 1] < cfg.y_collar_max)
        )

        is_torso = is_upper & ~is_sleeve_left & ~is_sleeve_right
        is_torso_front = is_torso & (centroids[:, 2] >= cfg.z_coronal)
        is_torso_back = is_torso & (centroids[:, 2] < cfg.z_coronal)

        region_defs = [
            ("torso_front", is_torso_front, (0.0, 0.0, 1.0)),
            ("torso_back", is_torso_back, (0.0, 0.0, -1.0)),
            ("sleeve_left", is_sleeve_left, (1.0, 0.0, 0.0)),
            ("sleeve_right", is_sleeve_right, (-1.0, 0.0, 0.0)),
            ("leg_left", is_leg_left, (0.0, -1.0, 0.0)),
            ("leg_right", is_leg_right, (0.0, -1.0, 0.0)),
        ]

        assigned_indices: list[IntArray] = []
        regions: list[GarmentRegion] = []
        covered_set: set[int] = set()

        for region_id, mask, orientation in region_defs:
            face_ids = np.where(mask)[0].astype(np.int64)
            if len(face_ids) == 0:
                continue
            # Duplicate check
            overlap = covered_set.intersection(face_ids)
            if overlap:
                raise ValueError(f"Region {region_id} contains {len(overlap)} duplicate faces.")
            covered_set.update(face_ids)
            assigned_indices.append(face_ids)
            regions.append(
                GarmentRegion(
                    region_id=region_id,
                    source_face_ids=face_ids,
                    material_id=cfg.default_material,
                    orientation_hint=orientation,
                )
            )

        # Invariant verification: exact 100% coverage
        if len(covered_set) != total_faces:
            missing = set(range(total_faces)) - covered_set
            raise ValueError(
                f"Face partition coverage failure: {len(covered_set)}/{total_faces} covered "
                f"({len(missing)} unassigned)."
            )

        return tuple(regions)


def validate_region_coverage(
    regions: Sequence[GarmentRegion],
    total_face_count: int,
) -> bool:
    """Validate that regions cover 100% of source faces with zero overlap."""
    seen: set[int] = set()
    for r in regions:
        for fid in r.source_face_ids:
            if fid in seen:
                return False
            seen.add(fid)
    return len(seen) == total_face_count


__all__ = [
    "RegionAtlas",
    "RegionAtlasConfig",
    "validate_region_coverage",
]
