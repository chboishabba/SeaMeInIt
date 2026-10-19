"""Tests for deterministic fabric sheet nesting and collision checks."""

from __future__ import annotations

import numpy as np

from smii.garment.pattern_model import PatternChart
from smii.manufacturing.nesting import NestingConfig, nest_charts_deterministic


def _make_dummy_chart(cid: str, width: float, height: float) -> PatternChart:
    pts = np.array(
        [
            [0.0, 0.0],
            [width, 0.0],
            [width, height],
            [0.0, height],
        ],
        dtype=np.float64,
    )
    return PatternChart(
        chart_id=cid,
        region_id=cid,
        source_face_ids=np.array([0, 1], dtype=np.int64),
        uv_mm=pts,
        local_faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        boundary_loops=(np.array([0, 1, 2, 3], dtype=np.int64),),
        source_vertex_ids=np.array([0, 1, 2, 3], dtype=np.int64),
    )


def test_nesting_layout_and_zero_collisions() -> None:
    charts = [
        _make_dummy_chart("chart_1", 300.0, 400.0),
        _make_dummy_chart("chart_2", 250.0, 350.0),
        _make_dummy_chart("chart_3", 400.0, 200.0),
    ]

    cfg = NestingConfig(fabric_width_mm=1000.0, clearance_mm=10.0, allowance_mm=5.0)
    res = nest_charts_deterministic(charts, cfg)

    assert len(res.placements) == 3
    assert res.collision_count == 0
    assert len(res.unplaced_chart_ids) == 0
    assert res.fabric_length_mm > 0.0
    assert 0.0 < res.utilization <= 1.0


def test_nesting_flags_unplaced_piece_exceeding_width() -> None:
    # A piece wider than fabric roll (1500mm piece on 1000mm fabric)
    charts = [_make_dummy_chart("oversized", 1500.0, 300.0)]
    cfg = NestingConfig(fabric_width_mm=1000.0)
    res = nest_charts_deterministic(charts, cfg)

    assert len(res.placements) == 0
    assert "oversized" in res.unplaced_chart_ids
