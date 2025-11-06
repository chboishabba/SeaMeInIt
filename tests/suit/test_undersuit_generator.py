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
