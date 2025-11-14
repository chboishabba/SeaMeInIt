from __future__ import annotations

import numpy as np
import pytest

from suit import LAYER_REFERENCE_MEASUREMENTS, UnderSuitGenerator, UnderSuitOptions


@pytest.fixture()
def tetra_body() -> dict[str, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3) / 2.0, 0.0],
            [0.5, np.sqrt(3) / 6.0, np.sqrt(6) / 3.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ],
        dtype=int,
    )
    return {"vertices": vertices, "faces": faces}


@pytest.fixture()
def representative_measurements() -> dict[str, float]:
    return {
        "chest_circumference": 102.0,
        "waist_circumference": 82.0,
        "hip_circumference": 100.0,
    }


@pytest.fixture()
def tetra_body_with_joints(tetra_body: dict[str, np.ndarray]) -> dict[str, object]:
    joints = {
        "left_shoulder": np.array([0.3, 0.2, 0.3]),
        "left_elbow": np.array([0.25, 0.2, 0.15]),
        "left_wrist": np.array([0.2, 0.2, 0.05]),
        "left_hand": np.array([0.15, 0.2, 0.05]),
        "right_shoulder": np.array([0.7, 0.2, 0.3]),
        "right_elbow": np.array([0.75, 0.2, 0.15]),
        "right_wrist": np.array([0.8, 0.2, 0.05]),
        "right_hand": np.array([0.85, 0.2, 0.05]),
        "left_hip": np.array([0.45, 0.2, 0.05]),
        "left_knee": np.array([0.45, 0.25, -0.15]),
        "left_ankle": np.array([0.45, 0.25, -0.35]),
        "left_foot": np.array([0.45, 0.3, -0.4]),
        "right_hip": np.array([0.55, 0.2, 0.05]),
        "right_knee": np.array([0.55, 0.25, -0.15]),
        "right_ankle": np.array([0.55, 0.25, -0.35]),
        "right_foot": np.array([0.55, 0.3, -0.4]),
        "pelvis": np.array([0.5, 0.2, 0.1]),
        "spine": np.array([0.5, 0.2, 0.25]),
        "spine1": np.array([0.5, 0.2, 0.35]),
        "spine2": np.array([0.5, 0.2, 0.45]),
        "neck": np.array([0.5, 0.2, 0.55]),
        "head": np.array([0.5, 0.2, 0.65]),
    }
    payload = dict(tetra_body)
    payload["joints"] = joints
    return payload


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    tri_vertices = vertices[faces]
    tri_normals = np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0])
    for index, face in enumerate(faces):
        normals[face] += tri_normals[index]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return normals / lengths


def _expected_scale(measurements: dict[str, float], ease_percent: float) -> float:
    ratios: list[float] = []
    for name, reference in LAYER_REFERENCE_MEASUREMENTS.items():
        if name in measurements:
            ratios.append(measurements[name] / reference)
    scale = sum(ratios) / len(ratios) if ratios else 1.0
    return scale * (1.0 + ease_percent)


def test_base_layer_respects_measurement_scaling(tetra_body: dict[str, np.ndarray], representative_measurements: dict[str, float]) -> None:
    generator = UnderSuitGenerator()
    options = UnderSuitOptions(
        base_thickness=0.002,
        include_insulation=False,
        include_comfort_liner=False,
        ease_percent=0.05,
    )
    result = generator.generate(
        tetra_body,
        options=options,
        measurements=representative_measurements,
    )

    base_vertices = result.base_layer.vertices
    original_vertices = tetra_body["vertices"]
    normals = _vertex_normals(original_vertices, tetra_body["faces"])

    expected_scale = _expected_scale(representative_measurements, options.ease_percent)
    centroid = original_vertices.mean(axis=0, keepdims=True)
    scaled = (original_vertices - centroid) * expected_scale + centroid
    expected_vertices = scaled + normals * options.base_thickness

    np.testing.assert_allclose(base_vertices, expected_vertices, rtol=1e-6, atol=1e-6)


def test_watertight_mesh_preserves_seams(tetra_body: dict[str, np.ndarray], representative_measurements: dict[str, float]) -> None:
    generator = UnderSuitGenerator()
    result = generator.generate(tetra_body, measurements=representative_measurements)

    assert result.metadata["watertight"] is True
    assert result.metadata["seam_max_deviation"] == pytest.approx(0.0, rel=1e-6, abs=1e-6)


def test_layering_toggles_adjust_offsets(tetra_body: dict[str, np.ndarray], representative_measurements: dict[str, float]) -> None:
    generator = UnderSuitGenerator()
    normals = _vertex_normals(tetra_body["vertices"], tetra_body["faces"])

    enabled = generator.generate(tetra_body, measurements=representative_measurements)
    assert enabled.insulation_layer is not None
    assert enabled.comfort_layer is not None

    np.testing.assert_allclose(
        enabled.insulation_layer.vertices - enabled.base_layer.vertices,
        normals * enabled.insulation_layer.thickness,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        enabled.comfort_layer.vertices - enabled.insulation_layer.vertices,
        normals * enabled.comfort_layer.thickness,
        rtol=1e-6,
        atol=1e-6,
    )

    disabled = generator.generate(
        tetra_body,
        options=UnderSuitOptions(include_insulation=False, include_comfort_liner=False),
        measurements=representative_measurements,
    )
    assert disabled.insulation_layer is None
    assert disabled.comfort_layer is None


def test_generator_emits_body_axes_metadata(tetra_body_with_joints: dict[str, object]) -> None:
    generator = UnderSuitGenerator()
    measurements = {
        "arm_length": {"value": 0.8, "confidence": 0.3},
        "inseam_length": {"value": 0.95, "confidence": 0.7},
    }

    result = generator.generate(tetra_body_with_joints, measurements=measurements)

    metadata_axes = result.metadata["body_axes"]
    assert set(metadata_axes) >= {"left_arm", "right_arm", "left_leg", "right_leg", "spine"}

    centroid = tetra_body_with_joints["vertices"].mean(axis=0)
    scale_factor = (1.0 + UnderSuitOptions().ease_percent)  # measurement_scale is 1.0 in this fixture

    def _scale(point: np.ndarray) -> np.ndarray:
        return (point - centroid) * scale_factor + centroid

    left_shoulder = _scale(tetra_body_with_joints["joints"]["left_shoulder"])
    left_elbow = _scale(tetra_body_with_joints["joints"]["left_elbow"])
    left_wrist = _scale(tetra_body_with_joints["joints"]["left_wrist"])
    left_hand = _scale(tetra_body_with_joints["joints"]["left_hand"])

    geom_length = (
        np.linalg.norm(left_elbow - left_shoulder)
        + np.linalg.norm(left_wrist - left_elbow)
        + np.linalg.norm(left_hand - left_wrist)
    )
    expected_length = geom_length + 0.3 * (measurements["arm_length"]["value"] - geom_length)

    left_arm_meta = metadata_axes["left_arm"]
    assert left_arm_meta["measurement"] == "arm_length"
    assert left_arm_meta["parameters"][0] == pytest.approx(0.0)
    assert left_arm_meta["parameters"][-1] == pytest.approx(1.0)
    assert left_arm_meta["length"] == pytest.approx(expected_length)
