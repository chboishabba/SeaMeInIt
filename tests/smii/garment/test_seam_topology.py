"""Tests for seam graph extraction, source-edge indexing, and arc-length mismatch metrics."""

from __future__ import annotations

import numpy as np
import pytest

from smii.garment.pattern_model import BoundaryKind, GarmentRegion, PatternChart
from smii.garment.seam_topology import SeamTopologyExtractor


def _make_two_triangles() -> tuple[np.ndarray, np.ndarray, tuple[GarmentRegion, ...]]:
    # Two triangles sharing edge (1, 2)
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [2.0, 0.5, 0.0],  # 3
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],  # face 0
            [1, 3, 2],  # face 1
        ],
        dtype=np.int64,
    )
    regions = (
        GarmentRegion(
            region_id="panel_left",
            source_face_ids=np.array([0], dtype=np.int64),
            material_id="textile_A",
            orientation_hint=(0.0, 0.0, 1.0),
        ),
        GarmentRegion(
            region_id="panel_right",
            source_face_ids=np.array([1], dtype=np.int64),
            material_id="textile_B",
            orientation_hint=(0.0, 0.0, 1.0),
        ),
    )
    return vertices, faces, regions


def test_seam_extraction_and_classification() -> None:
    vertices, faces, regions = _make_two_triangles()
    extractor = SeamTopologyExtractor(allowance_mm=12.0)
    seams, boundary_kinds = extractor.extract_seams(vertices, faces, regions)

    assert len(seams) == 1
    seam = seams[0]
    assert seam.seam_id == "seam_panel_left_panel_right"
    assert seam.allowance_mm == 12.0
    assert seam.left.chart_id == "panel_left"
    assert seam.right.chart_id == "panel_right"
    # Shared edge between 1 and 2
    assert set(seam.left.boundary_vertex_ids) == {1, 2}
    assert boundary_kinds[seam.seam_id] == BoundaryKind.SEAM

    # 3D arc-length on shared mesh edge has 0.0 error
    l_left, l_right, e_seam = extractor.compute_3d_seam_lengths(vertices, seam)
    assert l_left == pytest.approx(1.0, abs=1e-6)
    assert l_right == pytest.approx(1.0, abs=1e-6)
    assert e_seam == pytest.approx(0.0, abs=1e-8)


def test_seam_uv_mismatch_metric() -> None:
    _verts, _faces, regions = _make_two_triangles()
    extractor = SeamTopologyExtractor()
    seams, _ = extractor.extract_seams(_verts, _faces, regions)
    seam = seams[0]

    # Mock 2D charts with intentional 10% length discrepancy along edge (1, 2)
    chart_left = PatternChart(
        chart_id="panel_left",
        region_id="panel_left",
        source_face_ids=np.array([0], dtype=np.int64),
        uv_mm=np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]], dtype=np.float64),
        local_faces=np.array([[0, 1, 2]], dtype=np.int64),
        boundary_loops=(np.array([0, 1, 2], dtype=np.int64),),
        source_vertex_ids=np.array([0, 1, 2], dtype=np.int64),
    )
    # Right chart has edge length 110mm instead of 100mm
    chart_right = PatternChart(
        chart_id="panel_right",
        region_id="panel_right",
        source_face_ids=np.array([1], dtype=np.int64),
        uv_mm=np.array([[0.0, 0.0], [50.0, 50.0], [0.0, 110.0]], dtype=np.float64),
        local_faces=np.array([[0, 1, 2]], dtype=np.int64),
        boundary_loops=(np.array([0, 1, 2], dtype=np.int64),),
        source_vertex_ids=np.array([1, 3, 2], dtype=np.int64),
    )

    e_uv = extractor.compute_uv_seam_error(seam, chart_left, chart_right)
    # Expected mismatch: |100 - 110| / 110 = 10 / 110 = 0.090909...
    assert e_uv == pytest.approx(10.0 / 110.0, rel=1e-3)
