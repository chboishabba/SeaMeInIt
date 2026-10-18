"""Construction feature regression tests using tiny synthetic UV charts."""

import numpy as np
import pytest

from smii.garment.pattern_model import (
    FullSuitPattern,
    MarkKind,
    PatternChart,
    PatternFeature,
    PatternMark,
)
from smii.manufacturing.correction_bridge import materialize_dart
from smii.manufacturing.nesting import NestingResult, Placement
from smii.manufacturing.svg_export import export_full_suit_svg
from smii.manufacturing.validation import validate_manufacturing_layout


def _chart() -> PatternChart:
    return PatternChart(
        chart_id="panel",
        region_id="panel",
        source_face_ids=np.array([0, 1]),
        uv_mm=np.array(
            [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0],
             [0.0, 100.0], [50.0, 60.0], [40.0, 20.0], [60.0, 20.0]]
        ),
        local_faces=np.array([[0, 1, 4], [1, 2, 4]]),
        boundary_loops=(np.array([0, 1, 2, 3]),),
        source_vertex_ids=np.arange(7),
    )


def _pattern(chart: PatternChart) -> FullSuitPattern:
    return FullSuitPattern(
        source_mesh_sha256="synthetic",
        regions=(),
        charts=(chart,),
        seams=(),
        boundary_kinds={},
    )


def _nest(rotation: int = 0) -> NestingResult:
    return NestingResult(
        placements=(
            Placement(
                chart_id="panel",
                x_mm=20,
                y_mm=20,
                rotation_deg=rotation,
                mirrored=False,
                bounds_mm=(20, 20, 140, 140),
                offset_contour_mm=np.array(
                    [[20, 20], [140, 20], [140, 140], [20, 140]]
                ),
            ),
        ),
        fabric_length_mm=200,
        fabric_width_mm=200,
        utilization=0.5,
        collision_count=0,
        unplaced_chart_ids=(),
    )


def test_feature_roundtrip():
    mark = PatternMark(MarkKind.DRILL, ((12.0, 13.0),))
    feature = PatternFeature("d1", "op1", "dart", (mark,))
    assert PatternFeature.from_dict(feature.to_dict()) == feature
    chart = _chart()
    from dataclasses import replace

    with_feature = replace(chart, features=(feature,))
    assert PatternChart.from_dict(with_feature.to_dict()).features == (feature,)
    assert PatternChart.from_dict(chart.to_dict()).features == ()


def test_dart_geometry_and_drill_backoff():
    feature = materialize_dart(
        _chart(),
        feature_id="d1",
        operator_id="op1",
        apex_vertex=4,
        mouth_left_vertex=5,
        mouth_right_vertex=6,
    )
    assert [mark.kind for mark in feature.marks[:4]] == [
        MarkKind.SEW, MarkKind.SEW, MarkKind.FOLD, MarkKind.DRILL
    ]
    apex = np.array([50.0, 60.0])
    drill = np.array(feature.marks[3].points_mm[0])
    assert np.linalg.norm(drill - apex) == pytest.approx(5.0)


def test_svg_layers_and_rotation():
    from dataclasses import replace

    feature = materialize_dart(
        _chart(), feature_id="d1", operator_id="op1",
        apex_vertex=4, mouth_left_vertex=5, mouth_right_vertex=6,
    )
    chart = replace(_chart(), features=(feature,))
    svg0 = export_full_suit_svg(_pattern(chart), _nest(0))
    svg90 = export_full_suit_svg(_pattern(chart), _nest(90))
    assert 'id="features_panel"' in svg0
    assert 'data-layer="SEW"' in svg0
    assert 'data-layer="FOLD"' in svg0
    assert 'data-layer="DRILL"' in svg0
    assert svg90 != svg0


def test_gate7_accepts_valid_dart_and_blocks_mismatch():
    from dataclasses import replace

    feature = materialize_dart(
        _chart(), feature_id="d1", operator_id="op1",
        apex_vertex=4, mouth_left_vertex=5, mouth_right_vertex=6,
    )
    chart = replace(_chart(), features=(feature,))
    assert validate_manufacturing_layout(_pattern(chart), _nest()).gate7_promoted == 1
    marks = list(feature.marks)
    marks[1] = PatternMark(MarkKind.SEW, ((60.0, 20.0), (52.0, 60.0)))
    malformed = replace(feature, marks=tuple(marks))
    chart = replace(chart, features=(malformed,))
    result = validate_manufacturing_layout(_pattern(chart), _nest())
    assert result.gate7_promoted == 0
    assert any("mismatch" in reason for reason in result.blocking_reasons)


def test_unequal_dart_is_rejected():
    chart = _chart()
    chart.uv_mm[6] = [90.0, 20.0]
    with pytest.raises(ValueError, match="mismatch"):
        materialize_dart(
            chart, feature_id="bad", operator_id="op",
            apex_vertex=4, mouth_left_vertex=5, mouth_right_vertex=6,
        )
