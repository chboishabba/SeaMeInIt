from __future__ import annotations

import numpy as np

from smii.seams.metric_panelization import (
    MetricEnergyWeights,
    build_metric_panelization_payload,
    generate_correction_candidates,
)


def test_metric_panelization_selects_dart_for_cone_defect() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 1.4],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], dtype=int)
    labels = np.zeros(len(faces), dtype=int)

    payload = build_metric_panelization_payload(
        vertices=vertices,
        faces=faces,
        labels=labels,
        seam_edges=tuple(),
        families=("dart", "relief_cut", "stretch_zone", "bias_orientation"),
        max_corrections_per_panel=2,
        weights=MetricEnergyWeights(seam=0.0),
    )

    selected = payload["selected_corrections"]
    assert isinstance(selected, list)
    assert any(entry["family"] == "dart" for entry in selected)
    energy = payload["energy"]
    assert isinstance(energy, dict)
    assert energy["corrected_residual_total"] < energy["raw_residual_total"]


def test_metric_panelization_has_no_positive_corrections_for_flat_strip() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]], dtype=int)
    labels = np.zeros(len(faces), dtype=int)

    candidates, _reports = generate_correction_candidates(
        vertices=vertices,
        faces=faces,
        labels=labels,
    )

    assert not [candidate for candidate in candidates if candidate.gain > 0.0]


def test_metric_panelization_selects_gusset_for_inserted_gap_proxy() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.1, 0.6, 0.2],
            [3.1, 0.6, -0.2],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    labels = np.zeros(len(faces), dtype=int)

    payload = build_metric_panelization_payload(
        vertices=vertices,
        faces=faces,
        labels=labels,
        seam_edges=tuple(),
        families=("ease", "gusset", "stretch_zone"),
        max_corrections_per_panel=1,
        weights=MetricEnergyWeights(seam=0.0, complexity=0.2, manufacture=0.2),
    )

    selected = payload["selected_corrections"]
    assert isinstance(selected, list)
    assert selected
    assert selected[0]["family"] == "gusset"
