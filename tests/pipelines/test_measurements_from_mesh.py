from __future__ import annotations

import numpy as np

from smii.measurements.from_mesh import MeshMeasurementConfig, infer_measurements_from_mesh


def _prism_vertices(width: float = 1.0, depth: float = 1.0, height: float = 2.0) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [0.0, 0.0, depth],
            [width, 0.0, depth],
            [0.0, height, 0.0],
            [width, height, 0.0],
            [0.0, height, depth],
            [width, height, depth],
        ],
        dtype=float,
    )


def test_mesh_measurements_are_deterministic():
    vertices = _prism_vertices(width=1.0, depth=1.0, height=2.0)
    cfg = MeshMeasurementConfig(slice_band=0.5)  # widen band to capture sparse vertices

    first = infer_measurements_from_mesh(vertices, config=cfg)
    second = infer_measurements_from_mesh(vertices, config=cfg)

    assert first == second
    assert first["height"] == 2.0
    assert first["shoulder_width"] == 1.0
    assert first["hip_width"] == 1.0
    # Cross-section of a 1x1 square has perimeter 4
    assert first["chest_circumference"] == 4.0
    assert first["waist_circumference"] == 4.0
    assert first["hip_circumference"] == 4.0


def test_mesh_measurements_prefers_joints_when_available():
    vertices = _prism_vertices(width=1.0, depth=1.0, height=2.0)
    joints = {
        "left_shoulder": np.array([0.0, 1.6, 0.0]),
        "right_shoulder": np.array([1.5, 1.6, 0.0]),
        "left_hip": np.array([0.0, 0.9, 0.0]),
        "right_hip": np.array([1.2, 0.9, 0.0]),
    }

    measurements = infer_measurements_from_mesh(vertices, joints=joints)

    assert measurements["shoulder_width"] == 1.5
    assert measurements["hip_width"] == 1.2
