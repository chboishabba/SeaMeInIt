from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from smii.unwrap.bt369_local_repair import boundary_shortening, local_dart_candidates
from smii.unwrap.bt369_manufacturing import ManufacturingLimits, chart_diagnostics
from smii.unwrap.bt369_seam_tracing import flow_trace_partition


def _fixture():
    vertices = np.array([[x, y, 0.0] for y in range(6) for x in range(6)])
    faces = []
    for y in range(5):
        for x in range(5):
            a = y * 6 + x
            faces.extend([(a, a + 1, a + 7), (a, a + 7, a + 6)])
    cells = [SimpleNamespace(residual=0.0, orientation_bin=0) for _ in faces]
    return vertices, faces, cells


def test_boundary_shortening_never_increases_seam_ratio():
    v, f, cells = _fixture()
    limits = ManufacturingLimits(min_faces=2, min_area_fraction=0)
    original = flow_trace_partition(
        vertices=v, faces=f, cells=cells, limits=limits, target_charts=4
    )
    shortened, moves = boundary_shortening(
        vertices=v, faces=f, partition=original, min_faces=2
    )
    before = chart_diagnostics(vertices=v, faces=f, partition=original, limits=limits)
    after = chart_diagnostics(vertices=v, faces=f, partition=shortened, limits=limits)
    assert after["seam_length_over_sqrt_area"] <= before["seam_length_over_sqrt_area"] + 1e-10
    assert all(move["seam_length_saved"] > 0 for move in moves)


def test_local_splits_preserve_exact_coverage():
    v, f, _ = _fixture()
    candidates = local_dart_candidates(
        vertices=v, faces=f, partition=(tuple(range(len(f))),),
        chart_indices=(0,), min_faces=5
    )
    assert candidates
    for candidate in candidates:
        partition = candidate["partition"]
        assert sorted(face for chart in partition for face in chart) == list(range(len(f)))
        assert len(partition) > 1


def test_invalid_chart_target_is_rejected():
    v, f, _ = _fixture()
    with pytest.raises(IndexError):
        local_dart_candidates(
            vertices=v, faces=f, partition=(tuple(range(len(f))),),
            chart_indices=(19,)
        )
