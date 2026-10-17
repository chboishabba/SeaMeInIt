"""Tests for SVG vector layout rendering and calibration elements."""

from __future__ import annotations

import numpy as np

from smii.garment.pattern_model import FullSuitPattern, GarmentRegion, PatternChart
from smii.manufacturing.nesting import nest_charts_deterministic
from smii.manufacturing.svg_export import export_full_suit_svg
from smii.manufacturing.validation import validate_manufacturing_layout


def test_svg_export_contains_calibration_and_layers() -> None:
    pts = np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 300.0], [0.0, 300.0]], dtype=np.float64)
    chart = PatternChart(
        chart_id="torso_front",
        region_id="torso_front",
        source_face_ids=np.array([0, 1], dtype=np.int64),
        uv_mm=pts,
        local_faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        boundary_loops=(np.array([0, 1, 2, 3], dtype=np.int64),),
        source_vertex_ids=np.array([0, 1, 2, 3], dtype=np.int64),
    )
    region = GarmentRegion(
        region_id="torso_front",
        source_face_ids=np.array([0, 1], dtype=np.int64),
        material_id="spandex",
        orientation_hint=(0.0, 0.0, 1.0),
    )
    pattern = FullSuitPattern(
        source_mesh_sha256="test-sha",
        regions=(region,),
        charts=(chart,),
        seams=(),
        boundary_kinds={},
    )

    nesting = nest_charts_deterministic(pattern.charts)
    svg = export_full_suit_svg(pattern, nesting)

    assert "<svg" in svg
    assert "</svg>" in svg
    assert 'data-units="mm"' in svg
    # Calibration scale bar
    assert "calibration_scale" in svg
    assert "100 mm" in svg
    # Grainline
    assert 'data-grainline="true"' in svg
    # Chart group
    assert 'id="chart_torso_front"' in svg

    # Validation report
    report = validate_manufacturing_layout(pattern, nesting)
    assert report.is_valid
    assert report.gate7_promoted == 1
    assert report.collision_count == 0
