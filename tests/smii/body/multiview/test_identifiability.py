"""Tests for shape Jacobian spectrum and condition metrics."""

from __future__ import annotations

import numpy as np

from smii.body.multiview.identifiability import analyze_identifiability, compute_shape_jacobian
from smii.body.multiview.synthetic_rig import (
    build_four_view_orthogonal_rig,
    build_frontal_rig,
)


def _make_simple_landmarks() -> tuple[np.ndarray, np.ndarray]:
    pts = np.array(
        [
            [0.0, 0.5, 0.0],  # Head
            [0.2, 0.0, 0.0],  # Left arm
            [-0.2, 0.0, 0.0],  # Right arm
            [0.0, -0.5, 0.0],  # Feet
        ],
        dtype=np.float64,
    )
    # 2 shape modes: Mode 0 lateral (X), Mode 1 depth (Z)
    bases = np.zeros((4, 3, 2), dtype=np.float64)
    bases[:, 0, 0] = 0.1  # Mode 0 expands X
    bases[:, 2, 1] = 0.1  # Mode 1 expands Z
    return pts, bases


def test_orthogonal_improves_condition_number() -> None:
    pts, bases = _make_simple_landmarks()
    frontal_cams = build_frontal_rig(num_cameras=1)
    ortho_cams = build_four_view_orthogonal_rig()

    J_front = compute_shape_jacobian(frontal_cams, pts, bases)
    J_ortho = compute_shape_jacobian(ortho_cams, pts, bases)

    assert J_front.shape[1] == 2
    assert J_ortho.shape[1] == 2

    spec_front = analyze_identifiability(frontal_cams, pts, bases)
    spec_ortho = analyze_identifiability(ortho_cams, pts, bases)

    # In frontal view, sensitivity to depth mode 1 is near-zero, so condition number is high
    # In orthogonal 4-view, condition number is balanced and much lower
    assert spec_ortho.condition_number < spec_front.condition_number
    assert spec_ortho.min_singular_value > spec_front.min_singular_value
