from __future__ import annotations

import numpy as np
import pytest

from suit_hard.shell_generator import (
    ShellGenerator,
    ShellOptions,
    _vertex_normals,
)


@pytest.fixture()
def tetra_mesh() -> dict[str, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=int,
    )
    return {"vertices": vertices, "faces": faces}


@pytest.fixture()
def region_masks(tetra_mesh: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mask_apex = np.array([True, False, False, False])
    mask_base = np.array([False, True, True, True])
    return {"apex": mask_apex, "base": mask_base}


def test_uniform_thickness_offsets_vertices(tetra_mesh: dict[str, np.ndarray]) -> None:
    generator = ShellGenerator()
    result = generator.generate(tetra_mesh, thickness_profile=0.01)

    vertices = tetra_mesh["vertices"]
    normals = _vertex_normals(vertices, tetra_mesh["faces"])
    displacement = result.vertices - vertices
    projected = np.einsum("ij,ij->i", displacement, normals)

    assert np.allclose(projected, 0.01)
    assert np.allclose(result.thickness, 0.01)
    assert result.metadata["profile_type"] == "uniform"


def test_regional_thickness_overrides(
    region_masks: dict[str, np.ndarray], tetra_mesh: dict[str, np.ndarray]
) -> None:
    options = ShellOptions(default_thickness=0.005, region_masks=region_masks)
    generator = ShellGenerator()

    result = generator.generate(
        tetra_mesh,
        thickness_profile={"apex": 0.02, "base": 0.01},
        options=options,
    )

    assert np.isclose(result.thickness[0], 0.02)
    assert np.allclose(result.thickness[1:], 0.01)
    assert result.metadata["profile_type"] == "regional"


def test_exclusion_zeroes_thickness(
    region_masks: dict[str, np.ndarray], tetra_mesh: dict[str, np.ndarray]
) -> None:
    options = ShellOptions(default_thickness=0.01, region_masks=region_masks)
    generator = ShellGenerator()

    result = generator.generate(tetra_mesh, exclusions=["apex"], options=options)

    assert result.thickness[0] == pytest.approx(0.0)
    assert np.allclose(result.vertices[0], tetra_mesh["vertices"][0])
    assert result.metadata["excluded_vertices"] == 1
    assert result.metadata["exclusion_ratio"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "kwargs, expected_exception",
    [
        ({"thickness_profile": -0.1}, ValueError),
        ({"thickness_profile": {"unknown": 0.01}}, ValueError),
        ({"thickness_profile": np.full(5, 0.01)}, ValueError),
        ({"exclusions": ["unknown"]}, ValueError),
    ],
)
def test_invalid_profiles_raise(
    kwargs: dict[str, object],
    expected_exception: type[Exception],
    region_masks: dict[str, np.ndarray],
    tetra_mesh: dict[str, np.ndarray],
) -> None:
    options = ShellOptions(region_masks=region_masks)
    generator = ShellGenerator()

    with pytest.raises(expected_exception):
        generator.generate(tetra_mesh, options=options, **kwargs)


def test_non_watertight_mesh_rejected(tetra_mesh: dict[str, np.ndarray]) -> None:
    faces = tetra_mesh["faces"][1:]
    invalid = {"vertices": tetra_mesh["vertices"], "faces": faces}
    generator = ShellGenerator()

    with pytest.raises(ValueError):
        generator.generate(invalid)
