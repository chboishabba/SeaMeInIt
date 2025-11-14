from __future__ import annotations

import numpy as np
import pytest

from suit.body_axes import fit_body_axes


@pytest.fixture()
def simplified_body() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    joints = {
        "pelvis": np.array([0.0, 0.0, 0.0]),
        "spine": np.array([0.0, 0.0, 0.5]),
        "spine1": np.array([0.0, 0.0, 1.0]),
        "spine2": np.array([0.0, 0.0, 1.2]),
        "neck": np.array([0.0, 0.0, 1.45]),
        "head": np.array([0.0, 0.0, 1.65]),
        "left_shoulder": np.array([-0.2, 0.1, 1.45]),
        "left_elbow": np.array([-0.6, 0.1, 1.15]),
        "left_wrist": np.array([-0.95, 0.1, 0.85]),
        "left_hand": np.array([-1.1, 0.1, 0.85]),
        "right_shoulder": np.array([0.2, -0.1, 1.45]),
        "right_elbow": np.array([0.6, -0.1, 1.15]),
        "right_wrist": np.array([0.95, -0.1, 0.85]),
        "right_hand": np.array([1.1, -0.1, 0.85]),
        "left_hip": np.array([-0.2, 0.1, 0.0]),
        "left_knee": np.array([-0.2, 0.1, -0.85]),
        "left_ankle": np.array([-0.2, 0.1, -1.55]),
        "left_foot": np.array([-0.2, 0.25, -1.65]),
        "right_hip": np.array([0.2, -0.1, 0.0]),
        "right_knee": np.array([0.2, -0.1, -0.85]),
        "right_ankle": np.array([0.2, -0.1, -1.55]),
        "right_foot": np.array([0.2, -0.25, -1.65]),
    }
    vertices = np.stack(list(joints.values()), axis=0)
    return vertices, joints


def _polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def test_fit_body_axes_respects_measurement_confidence(simplified_body: tuple[np.ndarray, dict[str, np.ndarray]]) -> None:
    vertices, joints = simplified_body
    measurements = {"arm_length": 1.4, "inseam_length": 2.3}
    confidences = {"arm_length": 0.25, "inseam_length": 0.75}

    axes = fit_body_axes(vertices, joints=joints, measurements=measurements, measurement_confidences=confidences)

    left_arm_geom = _polyline_length(
        np.vstack([joints["left_shoulder"], joints["left_elbow"], joints["left_wrist"], joints["left_hand"]])
    )
    expected_left_arm = left_arm_geom + confidences["arm_length"] * (measurements["arm_length"] - left_arm_geom)
    assert axes.axis("left_arm").length == pytest.approx(expected_left_arm)

    left_leg_geom = _polyline_length(
        np.vstack([joints["left_hip"], joints["left_knee"], joints["left_ankle"], joints["left_foot"]])
    )
    expected_left_leg = left_leg_geom + confidences["inseam_length"] * (measurements["inseam_length"] - left_leg_geom)
    assert axes.axis("left_leg").length == pytest.approx(expected_left_leg)


@pytest.mark.parametrize(
    "axis_name, axis_index, increasing",
    [
        ("left_arm", 0, False),
        ("right_arm", 0, True),
        ("left_leg", 2, False),
        ("right_leg", 2, False),
        ("spine", 2, True),
    ],
)
def test_axis_sampling_is_monotonic_and_matches_measurements(
    simplified_body: tuple[np.ndarray, dict[str, np.ndarray]],
    axis_name: str,
    axis_index: int,
    increasing: bool,
) -> None:
    vertices, joints = simplified_body
    measurements = {"arm_length": 1.6, "inseam_length": 2.6, "height": 1.8}
    axes = fit_body_axes(vertices, joints=joints, measurements=measurements, measurement_confidences={})

    axis = axes.axis(axis_name)
    dense_samples = axis.sample(np.linspace(0.0, 1.0, 11))
    diffs = np.diff(dense_samples[:, axis_index])
    if increasing:
        assert np.all(diffs >= -1e-8)
    else:
        assert np.all(diffs <= 1e-8)

    polyline_samples = axis.sample(axis.parameters)
    sampled_length = np.linalg.norm(np.diff(polyline_samples, axis=0), axis=1).sum()
    target_measurement = measurements.get(axis.measurement_name, axis.length)
    assert sampled_length == pytest.approx(target_measurement, rel=1e-6, abs=1e-6)
