"""Tests for pinhole camera projection and ray unprojection."""

from __future__ import annotations

import numpy as np
import pytest

from smii.body.multiview.camera import CameraModel


def test_camera_projection_center() -> None:
    # Camera at (0, 0, 2.0) looking at (0, 0, 0)
    cam = CameraModel.look_at(
        camera_id="test_cam",
        eye=np.array([0.0, 0.0, 2.0], dtype=np.float64),
        target=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        focal_length=1000.0,
        image_size=(800, 600),
    )

    # Point exactly at target (0, 0, 0) should project to principal point (400, 300)
    pt = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    proj = cam.project(pt)
    assert proj.shape == (1, 2)
    assert proj[0, 0] == pytest.approx(400.0, abs=1e-4)
    assert proj[0, 1] == pytest.approx(300.0, abs=1e-4)


def test_camera_unproject_rays() -> None:
    cam = CameraModel.look_at(
        camera_id="test_cam",
        eye=np.array([0.0, 0.0, 3.0], dtype=np.float64),
        target=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        focal_length=1000.0,
        image_size=(800, 600),
    )

    # Center pixel (400, 300) should unproject ray along -Z towards target
    center_px = np.array([[400.0, 300.0]], dtype=np.float64)
    origins, dirs = cam.unproject_rays(center_px)

    assert origins[0, 2] == pytest.approx(3.0, abs=1e-4)
    # Direction in world frame: from (0,0,3) to (0,0,0) is [0, 0, -1]
    assert dirs[0, 0] == pytest.approx(0.0, abs=1e-4)
    assert dirs[0, 1] == pytest.approx(0.0, abs=1e-4)
    assert dirs[0, 2] == pytest.approx(-1.0, abs=1e-4)
