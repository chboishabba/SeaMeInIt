"""Tests for end-to-end full-suit partitioning, unwrap coordinates, and artifact serialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smii.garment.full_suit_partition import (
    FullSuitPartitionConfig,
    FullSuitPartitioner,
    save_full_suit_pattern,
)
from smii.garment.region_atlas import RegionAtlasConfig


def _make_mock_suit_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic body mesh with non-degenerate faces for unwrap."""
    vertices = np.array(
        [
            # Torso front
            [-0.1, 0.0, 0.1],  # 0
            [0.1, 0.0, 0.1],  # 1
            [0.0, 0.2, 0.1],  # 2
            # Torso back
            [-0.1, 0.0, -0.1],  # 3
            [0.1, 0.0, -0.1],  # 4
            [0.0, 0.2, -0.1],  # 5
            # Left sleeve
            [0.3, 0.1, 0.0],  # 6
            [0.4, 0.1, 0.0],  # 7
            [0.35, 0.2, 0.0],  # 8
            # Right sleeve
            [-0.3, 0.1, 0.0],  # 9
            [-0.4, 0.1, 0.0],  # 10
            [-0.35, 0.2, 0.0],  # 11
            # Left leg
            [0.1, -0.5, 0.0],  # 12
            [0.2, -0.5, 0.0],  # 13
            [0.15, -0.7, 0.0],  # 14
            # Right leg
            [-0.1, -0.5, 0.0],  # 15
            [-0.2, -0.5, 0.0],  # 16
            [-0.15, -0.7, 0.0],  # 17
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],  # torso front
            [3, 5, 4],  # torso back
            [6, 7, 8],  # sleeve left
            [9, 11, 10],  # sleeve right
            [12, 13, 14],  # leg left
            [15, 17, 16],  # leg right
        ],
        dtype=np.int64,
    )
    return vertices, faces


def test_full_suit_partitioner_flow(tmp_path: Path) -> None:
    vertices, faces = _make_mock_suit_mesh()
    cfg = FullSuitPartitionConfig(atlas_config=RegionAtlasConfig(y_crotch=-0.3, x_shoulder=0.25))
    partitioner = FullSuitPartitioner(cfg)
    pattern, diagnostics = partitioner.partition(vertices, faces)

    assert (
        pattern.region_count() == 6
        if hasattr(pattern, "region_count")
        else len(pattern.regions) == 6
    )
    assert len(pattern.charts) == 6
    assert diagnostics["total_faces"] == 6

    for chart in pattern.charts:
        assert chart.uv_mm.shape[1] == 2
        assert len(chart.boundary_loops) >= 1

    # Save to NPZ and verify existence
    npz_out = tmp_path / "full_suit.npz"
    save_full_suit_pattern(pattern, npz_out)
    assert npz_out.exists()

    npz_data = np.load(npz_out)
    assert npz_data["chart_count"].item() == 6


@pytest.mark.skipif(
    not Path("outputs/demo/afflec_receipted_curated_20260605_143059/body/afflec_body.npz").exists(),
    reason="Real Afflec mesh fixture not found",
)
def test_full_suit_partition_on_real_afflec() -> None:
    data = np.load("outputs/demo/afflec_receipted_curated_20260605_143059/body/afflec_body.npz")
    vertices = data["vertices"]
    faces = data["faces"]

    partitioner = FullSuitPartitioner()
    pattern, diagnostics = partitioner.partition(vertices, faces)

    assert len(pattern.regions) == 6
    assert len(pattern.charts) == 6
    assert diagnostics["total_faces"] == len(faces)
    assert diagnostics["total_vertices"] == len(vertices)

    # Verify that every chart has valid physical UV coordinates in mm (non-zero area, span > 50mm)
    for chart in pattern.charts:
        span_x = float(chart.uv_mm[:, 0].max() - chart.uv_mm[:, 0].min())
        span_y = float(chart.uv_mm[:, 1].max() - chart.uv_mm[:, 1].min())
        assert span_x > 50.0, f"Chart {chart.chart_id} has degenerate X span: {span_x}"
        assert span_y > 50.0, f"Chart {chart.chart_id} has degenerate Y span: {span_y}"
