"""Unit tests for the hard-shell segmentation utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from suit_hard import HardShellSegmentationOptions, HardShellSegmenter


@pytest.fixture()
def synthetic_joint_map() -> dict[str, np.ndarray]:
    return {
        "pelvis": np.array([0.0, 1.02, 0.0]),
        "left_hip": np.array([-0.12, 1.0, 0.03]),
        "left_knee": np.array([-0.14, 0.62, 0.11]),
        "left_ankle": np.array([-0.06, 0.2, 0.15]),
        "right_hip": np.array([0.12, 1.0, -0.02]),
        "right_knee": np.array([0.16, 0.63, -0.1]),
        "right_ankle": np.array([0.08, 0.2, -0.16]),
        "spine": np.array([0.0, 1.28, 0.02]),
        "neck": np.array([0.0, 1.58, 0.0]),
        "left_shoulder": np.array([-0.18, 1.46, 0.04]),
        "left_elbow": np.array([-0.48, 1.2, 0.18]),
        "left_wrist": np.array([-0.64, 0.9, 0.28]),
        "right_shoulder": np.array([0.18, 1.46, -0.02]),
        "right_elbow": np.array([0.5, 1.2, -0.16]),
        "right_wrist": np.array([0.64, 0.91, -0.26]),
    }


@pytest.fixture()
def segmenter() -> HardShellSegmenter:
    return HardShellSegmenter()


def test_segmentation_generates_major_panels(segmenter: HardShellSegmenter, synthetic_joint_map: dict[str, np.ndarray]) -> None:
    segmentation = segmenter.segment(synthetic_joint_map)
    panel_names = {panel.name for panel in segmentation.panels}
    expected = {
        "left_shoulder_panel",
        "left_elbow_panel",
        "left_hip_panel",
        "left_knee_panel",
        "right_shoulder_panel",
        "right_elbow_panel",
        "right_hip_panel",
        "right_knee_panel",
    }
    assert panel_names == expected


def test_segment_boundaries_are_continuous(segmenter: HardShellSegmenter, synthetic_joint_map: dict[str, np.ndarray]) -> None:
    segmentation = segmenter.segment(synthetic_joint_map)
    for panel in segmentation.panels:
        assert panel.boundary.shape[0] >= 9
        np.testing.assert_allclose(panel.boundary[0], panel.boundary[-1])
        boundary_spans = np.linalg.norm(np.diff(panel.boundary, axis=0), axis=1)
        assert np.all(boundary_spans > 0)


def test_options_control_allowance(segmenter: HardShellSegmenter, synthetic_joint_map: dict[str, np.ndarray]) -> None:
    options = HardShellSegmentationOptions(hinge_allowance=0.012, boundary_points=12)
    segmentation = segmenter.segment(synthetic_joint_map, options=options)
    for panel in segmentation.panels:
        assert math.isclose(panel.allowance, options.hinge_allowance)
        assert panel.boundary.shape[0] == options.boundary_points + 1


def test_motion_axis_tracks_pose_changes(segmenter: HardShellSegmenter, synthetic_joint_map: dict[str, np.ndarray]) -> None:
    baseline = segmenter.segment(synthetic_joint_map)
    bent_pose = dict(synthetic_joint_map)
    bent_pose["left_elbow"] = synthetic_joint_map["left_elbow"] + np.array([0.0, 0.05, -0.12])
    bent_pose["left_wrist"] = synthetic_joint_map["left_wrist"] + np.array([0.08, -0.06, -0.18])

    bent = segmenter.segment(bent_pose)

    base_panel = baseline.panels_for_joint("left_elbow")[0]
    bent_panel = bent.panels_for_joint("left_elbow")[0]

    base_axis = base_panel.motion_axis
    bent_axis = bent_panel.motion_axis

    base_axis /= np.linalg.norm(base_axis)
    bent_axis /= np.linalg.norm(bent_axis)

    angle = math.degrees(math.acos(np.clip(np.dot(base_axis, bent_axis), -1.0, 1.0)))
    assert angle > 5.0

    assert not np.allclose(base_panel.cut_point, bent_panel.cut_point)


def test_segmentation_handles_array_inputs(segmenter: HardShellSegmenter, synthetic_joint_map: dict[str, np.ndarray]) -> None:
    names, positions = zip(*sorted(synthetic_joint_map.items()))
    positions_array = np.vstack(positions)
    segmentation = segmenter.segment(positions_array, joint_names=names)
    assert segmentation.panels_for_joint("right_knee")
