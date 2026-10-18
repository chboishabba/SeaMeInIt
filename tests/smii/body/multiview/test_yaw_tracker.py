"""Tests for circular alpha-beta yaw tracking and wrapped angle estimation."""

from __future__ import annotations

import math
import numpy as np

from smii.body.multiview.yaw_tracker import (
    CircularYawTracker,
    YawMeasurement,
    wrap_angle_pi,
)


def test_wrap_angle_pi():
    assert math.isclose(wrap_angle_pi(0.0), 0.0, abs_tol=1e-7)
    assert math.isclose(wrap_angle_pi(math.pi), -math.pi, abs_tol=1e-7)
    assert math.isclose(wrap_angle_pi(3.0 * math.pi), -math.pi, abs_tol=1e-7)
    assert math.isclose(wrap_angle_pi(-3.0 * math.pi), -math.pi, abs_tol=1e-7)
    assert math.isclose(wrap_angle_pi(0.5 * math.pi), 0.5 * math.pi, abs_tol=1e-7)
    assert math.isclose(wrap_angle_pi(-0.5 * math.pi), -0.5 * math.pi, abs_tol=1e-7)


def test_estimate_yaw_from_landmarks():
    # Construct synthetic 33-landmark skeleton
    landmarks = np.zeros((33, 3), dtype=float)
    visibility = np.ones(33, dtype=float)

    # Front-facing: Left shoulder at (+0.25, -0.4, 0.0), Right shoulder at (-0.25, -0.4, 0.0)
    landmarks[11] = [0.25, -0.4, 0.0]
    landmarks[12] = [-0.25, -0.4, 0.0]
    landmarks[23] = [0.15, 0.0, 0.0]
    landmarks[24] = [-0.15, 0.0, 0.0]
    landmarks[0] = [0.0, -0.6, -0.1]  # Nose in front

    meas = CircularYawTracker.estimate_yaw_from_landmarks(landmarks, visibility)
    assert meas is not None
    assert meas.facing_front is True
    assert math.isclose(meas.angle_rad, 0.0, abs_tol=0.1)

    # 90-degree profile: Left shoulder at (0.0, -0.4, 0.25), Right shoulder at (0.0, -0.4, -0.25)
    landmarks[11] = [0.0, -0.4, 0.25]
    landmarks[12] = [0.0, -0.4, -0.25]
    landmarks[23] = [0.0, 0.0, 0.15]
    landmarks[24] = [0.0, 0.0, -0.15]
    landmarks[0] = [-0.1, -0.6, 0.0]

    meas_profile = CircularYawTracker.estimate_yaw_from_landmarks(landmarks, visibility)
    assert meas_profile is not None
    assert math.isclose(meas_profile.angle_rad, 0.5 * math.pi, abs_tol=0.15)


def test_yaw_tracker_smooth_rotation():
    tracker = CircularYawTracker(alpha=0.5, gamma=0.1)
    timestamps = np.linspace(0.0, 4.0, 120)  # 30 fps, 4 seconds for 1 full turn
    true_yaws = (2.0 * math.pi * timestamps / 4.0) - math.pi  # -pi to +pi

    tracked_yaws = []
    for i, (t, true_yaw) in enumerate(zip(timestamps, true_yaws)):
        # Simulate noisy measurement
        noisy_yaw = wrap_angle_pi(true_yaw + 0.05 * math.sin(i))
        meas = YawMeasurement(
            angle_rad=noisy_yaw,
            uncertainty_rad=0.10,
            shoulder_yaw_rad=noisy_yaw,
            hip_yaw_rad=noisy_yaw,
            facing_front=abs(noisy_yaw) < math.pi / 2.0,
            confidence=0.9,
        )
        state = tracker.update(meas, timestamp_s=float(t), frame_index=i)
        tracked_yaws.append(state.angle_rad)

    # Check that tracking stayed close to true yaw across the turn
    diffs = [abs(wrap_angle_pi(tr - ty)) for tr, ty in zip(tracked_yaws[10:], true_yaws[10:])]
    assert np.mean(diffs) < 0.20
    assert np.max(diffs) < 0.40
