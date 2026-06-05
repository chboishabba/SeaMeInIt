from __future__ import annotations

import numpy as np

from smii.seams.unwrap_benchmark import (
    GRAPH_ULTRAMETRIC_RECTANGLE,
    benchmark_sphere_rectangle_unwraps,
    build_uv_sphere_mesh,
)


def test_build_uv_sphere_mesh_has_rectangle_parameter_grid() -> None:
    mesh = build_uv_sphere_mesh(longitude_steps=12, latitude_steps=6)

    assert mesh.vertices.shape == ((12 + 1) * (6 + 1), 3)
    assert mesh.faces.shape == (12 * 6 * 2, 3)
    assert np.allclose(np.linalg.norm(mesh.vertices, axis=1), 1.0)
    assert np.isclose(mesh.lon.min(), -np.pi)
    assert np.isclose(mesh.lon.max(), np.pi)
    assert np.isclose(mesh.lat.min(), -np.pi / 2.0)
    assert np.isclose(mesh.lat.max(), np.pi / 2.0)


def test_graph_ultrametric_rectangle_wins_sphere_rectangle_benchmark() -> None:
    benchmark = benchmark_sphere_rectangle_unwraps(longitude_steps=16, latitude_steps=8)

    assert benchmark.winner.strategy == GRAPH_ULTRAMETRIC_RECTANGLE
    assert benchmark.winner.metrics.foldover_ratio == 0.0
    assert benchmark.winner.metrics.agreement_distance == 0
    assert benchmark.winner.metrics.aggregate_score < 0.2
    assert all(
        benchmark.winner.metrics.aggregate_score < candidate.metrics.aggregate_score
        for candidate in benchmark.candidates[1:]
    )


def test_sphere_rectangle_benchmark_keeps_numeric_backends_as_candidates() -> None:
    benchmark = benchmark_sphere_rectangle_unwraps(longitude_steps=12, latitude_steps=6)
    by_strategy = {candidate.strategy: candidate for candidate in benchmark.candidates}

    assert {"graph_ultrametric_rectangle", "lscm", "bootstrap_projection"}.issubset(by_strategy)
    assert (
        by_strategy[GRAPH_ULTRAMETRIC_RECTANGLE].metrics.agreement_depth
        > by_strategy["lscm"].metrics.agreement_depth
    )
    assert (
        by_strategy[GRAPH_ULTRAMETRIC_RECTANGLE].metrics.foldover_ratio
        <= by_strategy["bootstrap_projection"].metrics.foldover_ratio
    )
