from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smii.seams.panel_serialization_competition import (
    LSCM_BACKEND,
    panel_chart_diagnostics,
    serialize_panel,
)


@dataclass(frozen=True, slots=True)
class Panel:
    vertices: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    faces: tuple[tuple[int, int, int], ...]


def test_panel_chart_diagnostics_accepts_single_boundary_loop_chart() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    panel = Panel(
        vertices=(0, 1, 2, 3),
        edges=((0, 1), (1, 2), (2, 3), (0, 3), (0, 2)),
        faces=((0, 1, 2), (0, 2, 3)),
    )

    diagnostics = panel_chart_diagnostics(vertices, panel)

    assert diagnostics["backend_serializable"] is True
    assert diagnostics["connected_components"] == 1
    assert diagnostics["boundary_loops"] == 1
    assert diagnostics["blockers"] == []


def test_invalid_chart_domain_skips_production_backend() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    panel = Panel(
        vertices=(0, 1, 2, 3, 4, 5),
        edges=((0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)),
        faces=((0, 1, 2), (3, 4, 5)),
    )

    candidate, uv = serialize_panel(
        vertices=vertices,
        panel=panel,
        correction_tree=None,
        backend=LSCM_BACKEND,
        distortion_threshold=0.05,
    )

    assert uv is None
    assert candidate.promoted is False
    assert candidate.blockers == ("backend_skipped_invalid_chart_domain",)
    assert candidate.chart_diagnostics is not None
    assert candidate.chart_diagnostics["backend_serializable"] is False
    assert "panel_fragmentation_invalid" in candidate.chart_diagnostics["blockers"]
