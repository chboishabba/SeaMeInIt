"""Tests for garment region atlas face partitioning and coverage invariants."""

from __future__ import annotations

import numpy as np
import pytest

from smii.garment.region_atlas import RegionAtlas, RegionAtlasConfig, validate_region_coverage


def _make_mock_body_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Create a simple synthetic humanoid mesh with torso, arms, and legs."""
    vertices = np.array(
        [
            # Torso front/back
            [0.0, 0.0, 0.1],  # 0: torso front
            [0.0, 0.0, -0.1],  # 1: torso back
            [0.0, 0.2, 0.0],  # 2: chest top
            [0.0, -0.2, 0.0],  # 3: waist bottom
            # Left sleeve
            [0.3, 0.1, 0.0],  # 4: left shoulder
            [0.5, 0.1, 0.0],  # 5: left hand
            # Right sleeve
            [-0.3, 0.1, 0.0],  # 6: right shoulder
            [-0.5, 0.1, 0.0],  # 7: right hand
            # Left leg
            [0.1, -0.4, 0.0],  # 8: left thigh
            [0.1, -0.8, 0.0],  # 9: left foot
            # Right leg
            [-0.1, -0.4, 0.0],  # 10: right thigh
            [-0.1, -0.8, 0.0],  # 11: right foot
        ],
        dtype=np.float64,
    )

    faces = np.array(
        [
            # Torso front
            [0, 2, 3],
            # Torso back
            [1, 3, 2],
            # Left sleeve
            [4, 5, 2],
            # Right sleeve
            [6, 2, 7],
            # Left leg
            [8, 9, 3],
            # Right leg
            [10, 3, 11],
        ],
        dtype=np.int64,
    )

    return vertices, faces


def test_region_atlas_exact_coverage() -> None:
    vertices, faces = _make_mock_body_mesh()
    atlas = RegionAtlas(RegionAtlasConfig(y_crotch=-0.3, x_shoulder=0.2))
    regions = atlas.build_regions(vertices, faces)

    assert len(regions) > 0
    # Invariant: validate exact 100% face coverage
    assert validate_region_coverage(regions, len(faces))

    region_names = {r.region_id for r in regions}
    assert "torso_front" in region_names
    assert "torso_back" in region_names
    assert "sleeve_left" in region_names
    assert "sleeve_right" in region_names
    assert "leg_left" in region_names
    assert "leg_right" in region_names


def test_region_atlas_rejects_empty_mesh() -> None:
    verts = np.empty((0, 3), dtype=np.float64)
    faces = np.empty((0, 3), dtype=np.int64)
    atlas = RegionAtlas()
    with pytest.raises(ValueError, match="no faces"):
        atlas.build_regions(verts, faces)
