from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from smii.unwrap.bt369_manufacturing import (
    ManufacturingLimits,
    chart_diagnostics,
    regularize_partition,
)


def _mesh():
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0],
                           [0, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=float)
    faces = ((0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4))
    cells = [SimpleNamespace(residual=float(i) / 10, orientation_bin=i % 6)
             for i in range(4)]
    return vertices, faces, cells


def test_confetti_blocked_before_backend():
    vertices, faces, _ = _mesh()
    limits = ManufacturingLimits(max_charts=2, min_faces=2, min_area_fraction=0.1)
    receipt = chart_diagnostics(vertices=vertices, faces=faces,
                                partition=((0,), (1,), (2,), (3,)), limits=limits)
    assert "excessive_fragmentation" in receipt["manufacturing_blockers"]
    assert "subminimum_chart_area" in receipt["manufacturing_blockers"]


def test_micro_islands_merge_contiguously_without_losing_faces():
    vertices, faces, cells = _mesh()
    limits = ManufacturingLimits(max_charts=2, min_faces=2, min_area_fraction=0.1)
    merged = regularize_partition(vertices=vertices, faces=faces,
                                   partition=((0,), (1,), (2,), (3,)),
                                   cells=cells, limits=limits)
    assert sorted(f for chart in merged for f in chart) == [0, 1, 2, 3]
    assert all(len(chart) >= 2 for chart in merged)
    assert chart_diagnostics(vertices=vertices, faces=faces,
                             partition=merged, limits=limits)["manufacturing_admissible"]


def test_invalid_partition_fails_closed():
    vertices, faces, _ = _mesh()
    with pytest.raises(ValueError, match="exactly once"):
        chart_diagnostics(vertices=vertices, faces=faces,
                          partition=((0, 1), (1, 2, 3)), limits=ManufacturingLimits())


def test_seam_complexity_is_explicit():
    vertices, faces, _ = _mesh()
    receipt = chart_diagnostics(vertices=vertices, faces=faces,
                                partition=((0,), (1,), (2,), (3,)),
                                limits=ManufacturingLimits(max_charts=10, min_faces=1,
                                                           min_area_fraction=0,
                                                           max_seam_length_ratio=0.01))
    assert "excessive_seam_complexity" in receipt["manufacturing_blockers"]
