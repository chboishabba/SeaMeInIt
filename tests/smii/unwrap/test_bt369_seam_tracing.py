from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from smii.unwrap.bt369_manufacturing import ManufacturingLimits, chart_diagnostics
from smii.unwrap.bt369_seam_tracing import (
    SeamWeights,
    flow_trace_partition,
    morphological_cleanup,
    sensitivity,
    surface_graph,
)


def _grid():
    vertices = np.array([[x, y, 0.0] for y in range(4) for x in range(4)], dtype=float)
    faces = []
    for y in range(3):
        for x in range(3):
            a = 4 * y + x
            faces.extend(((a, a + 1, a + 5), (a, a + 5, a + 4)))
    cells = [SimpleNamespace(residual=i / 100, orientation_bin=i % 6) for i in range(len(faces))]
    return vertices, tuple(faces), cells


def test_flow_tracing_preserves_face_coverage_and_connectivity():
    v, f, cells = _grid()
    limits = ManufacturingLimits(max_charts=8, min_faces=2, min_area_fraction=0)
    partition = flow_trace_partition(
        vertices=v, faces=f, cells=cells, limits=limits, target_charts=3
    )
    assert sorted(i for chart in partition for i in chart) == list(range(len(f)))
    assert len(partition) <= 3
    graph = surface_graph(vertices=v, faces=f, cells=cells)
    for chart in partition:
        reached = {chart[0]}
        pending = [chart[0]]
        while pending:
            node = pending.pop()
            for neighbor, _ in graph[node]:
                if neighbor in chart and neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        assert reached == set(chart)


def test_morphology_and_sensitivity_are_deterministic():
    v, f, cells = _grid()
    limits = ManufacturingLimits(max_charts=8, min_faces=2, min_area_fraction=0)
    source = tuple((i,) for i in range(len(f)))
    first = morphological_cleanup(vertices=v, faces=f, partition=source, cells=cells, limits=limits)
    second = morphological_cleanup(
        vertices=v, faces=f, partition=source, cells=cells, limits=limits
    )
    assert first == second
    report = chart_diagnostics(vertices=v, faces=f, partition=first, limits=limits)
    matrix = sensitivity(report)
    assert len(matrix) == 12
    assert matrix[0]["max_seam_ratio"] == 8
    assert matrix[-1]["max_charts"] == 128


def test_surface_cost_is_positive_and_symmetric():
    v, f, cells = _grid()
    graph = surface_graph(vertices=v, faces=f, cells=cells, weights=SeamWeights())
    for i, neighbors in enumerate(graph):
        for j, cost in neighbors:
            assert cost > 0
            assert any(k == i and back == cost for k, back in graph[j])
