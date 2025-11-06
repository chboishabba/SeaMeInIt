"""Tests for the tent deployment module and exporter integration."""

from __future__ import annotations

from pathlib import Path

import json

import pytest

from exporters import (
    EngineConfig,
    PatternExporter,
    UnityUnrealExporter,
    export_suit_tent_bundle,
    load_smplx_template,
)
from modules.tent import (
    AttachmentAnchor,
    build_deployment_kinematics,
    default_attachment_anchors,
    generate_fold_paths,
    load_canopy_seams,
    load_canopy_template,
)


@pytest.fixture()
def sample_landmarks() -> dict[str, tuple[float, float, float]]:
    return {
        "c7_vertebra": (0.0, 1.62, -0.08),
        "sternum": (0.0, 1.45, 0.10),
        "left_acromion": (-0.18, 1.55, 0.02),
        "right_acromion": (0.18, 1.55, 0.02),
    }


def test_default_anchors_align_with_landmarks(
    sample_landmarks: dict[str, tuple[float, float, float]],
) -> None:
    anchors = default_attachment_anchors(sample_landmarks)

    assert set(anchors) == {
        "dorsal_mount",
        "ventral_mount",
        "left_shoulder_mount",
        "right_shoulder_mount",
    }

    dorsal = anchors["dorsal_mount"]
    assert isinstance(dorsal, AttachmentAnchor)
    assert dorsal.landmark == "c7_vertebra"
    expected_position = (
        sample_landmarks["c7_vertebra"][0] + 0.0,
        sample_landmarks["c7_vertebra"][1] + 0.08,
        sample_landmarks["c7_vertebra"][2] - 0.04,
    )
    assert pytest.approx(dorsal.position[0], rel=1e-3) == expected_position[0]
    assert pytest.approx(dorsal.position[1], rel=1e-3) == expected_position[1]
    assert pytest.approx(dorsal.position[2], rel=1e-3) == expected_position[2]

    left = anchors["left_shoulder_mount"]
    assert left.landmark == "left_acromion"
    assert left.normal[0] < 0  # offset aims laterally outward


def test_fold_paths_reference_valid_anchors(
    sample_landmarks: dict[str, tuple[float, float, float]],
) -> None:
    seams = load_canopy_seams()
    anchors = default_attachment_anchors(sample_landmarks)
    fold_paths = generate_fold_paths(anchors, seams)

    assert {"spine_roll", "left_rib_lock", "right_rib_lock"} == set(fold_paths)
    for path in fold_paths.values():
        path.validate(anchors)
        assert len(path.points) >= 2

    assert fold_paths["spine_roll"].anchors == ("dorsal_mount", "ventral_mount")
    assert fold_paths["left_rib_lock"].order < fold_paths["right_rib_lock"].order


def test_export_bundle_writes_glb_and_instructions(
    tmp_path: Path, sample_landmarks: dict[str, tuple[float, float, float]]
) -> None:
    canopy_mesh = load_canopy_template()
    seams = load_canopy_seams()
    deployment = build_deployment_kinematics(sample_landmarks, seams)

    template = load_smplx_template(
        vertices=[
            (0.0, 0.0, 0.0),
            (0.4, 0.0, 0.0),
            (0.0, 0.4, 0.0),
        ],
        faces=[(0, 1, 2)],
        joint_names=["root"],
        joint_parents=[-1],
        joint_positions=[(0.0, 0.0, 0.0)],
        skin_weights=[(1.0, 0.0, 0.0, 0.0)] * 3,
        skin_joints=[(0, 0, 0, 0)] * 3,
    )

    unity = UnityUnrealExporter(
        EngineConfig(name="Test", unit_scale=1.0, up_axis="Y", forward_axis="Z")
    )
    patterns = PatternExporter()

    result = export_suit_tent_bundle(
        suit_template=template,
        canopy_mesh=canopy_mesh,
        seam_plan=seams,
        deployment=deployment,
        unity_exporter=unity,
        pattern_exporter=patterns,
        output_dir=tmp_path,
        instruction_formats=("pdf",),
    )

    glb_path = result["glb_path"]
    assert glb_path.exists()

    instructions = result["instructions"]
    assert set(instructions) == {"pdf"}
    pdf_path = instructions["pdf"]
    assert pdf_path.exists()

    metadata_path = result["metadata_path"]
    assert metadata_path.exists()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["anchors"]["dorsal_mount"]["landmark"] == "c7_vertebra"
    assert payload["sequence"][0]["path"] == "spine_roll"
    assert payload["instructions"]["pdf"].endswith(".pdf")
