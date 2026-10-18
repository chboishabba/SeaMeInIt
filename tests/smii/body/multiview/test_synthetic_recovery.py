"""Tests for shape recovery accuracy and beta explosion prevention."""

from __future__ import annotations

import numpy as np

from smii.body.multiview.fit_referee import run_shape_recovery_test
from smii.body.multiview.synthetic_rig import (
    build_four_view_orthogonal_rig,
    build_frontal_rig,
)


def test_orthogonal_four_view_recovers_ground_truth_beta() -> None:
    pts = np.array(
        [
            [0.0, 0.6, 0.0],  # Head
            [0.0, 0.3, 0.08],  # Chest
            [0.2, 0.3, 0.0],  # Left shoulder
            [-0.2, 0.3, 0.0],  # Right shoulder
            [0.1, -0.4, 0.0],  # Left leg
            [-0.1, -0.4, 0.0],  # Right leg
        ],
        dtype=np.float64,
    )
    # 3 shape modes: scale (Y), breadth (X), depth (Z)
    bases = np.zeros((6, 3, 3), dtype=np.float64)
    bases[:, 1, 0] = pts[:, 1] * 0.1
    bases[:, 0, 1] = pts[:, 0] * 0.1
    bases[:, 2, 2] = 0.05

    beta_true = np.array([1.0, -0.5, 0.8], dtype=np.float64)

    ortho_cams = build_four_view_orthogonal_rig()
    res_ortho = run_shape_recovery_test(
        configuration_name="orthogonal_4view",
        cameras=ortho_cams,
        mean_points=pts,
        shape_bases=bases,
        beta_ground_truth=beta_true,
        noise_sigma_px=0.5,
    )

    # Recovery RMSE should be low under orthogonal views
    assert res_ortho.beta_recovery_rmse < 0.20
    assert not res_ortho.green_goblin_detected
    assert res_ortho.beta_max_abs < 2.0


def test_frontal_only_suffers_high_error_or_drift() -> None:
    pts = np.array(
        [
            [0.0, 0.6, 0.0],
            [0.0, 0.3, 0.08],
            [0.2, 0.3, 0.0],
            [-0.2, 0.3, 0.0],
            [0.1, -0.4, 0.0],
            [-0.1, -0.4, 0.0],
        ],
        dtype=np.float64,
    )
    bases = np.zeros((6, 3, 3), dtype=np.float64)
    bases[:, 1, 0] = pts[:, 1] * 0.1
    bases[:, 0, 1] = pts[:, 0] * 0.1
    bases[:, 2, 2] = 0.05

    beta_true = np.array([1.0, -0.5, 0.8], dtype=np.float64)

    # Frontal-only camera has weak sensitivity to depth mode 2
    frontal_cams = build_frontal_rig(num_cameras=1)
    res_front = run_shape_recovery_test(
        configuration_name="frontal_1view",
        cameras=frontal_cams,
        mean_points=pts,
        shape_bases=bases,
        beta_ground_truth=beta_true,
        noise_sigma_px=0.5,
    )

    ortho_cams = build_four_view_orthogonal_rig()
    res_ortho = run_shape_recovery_test(
        configuration_name="orthogonal_4view",
        cameras=ortho_cams,
        mean_points=pts,
        shape_bases=bases,
        beta_ground_truth=beta_true,
        noise_sigma_px=0.5,
    )

    # Orthogonal 4-view must have lower recovery error than frontal 1-view
    assert res_ortho.beta_recovery_rmse < res_front.beta_recovery_rmse
