"""Referee tests for ROM-relief routing and topological opening."""
import numpy as np
import pytest

from smii.garment.rom_relief import minimum_rom_relief, open_mesh_along_route


def fixture():
    # Four-triangle strip around a square center; center is high-ROM.
    vertices = np.array([
        [0., 0., 0.], [1., 0., 0.], [1., 1., 0.],
        [0., 1., 0.], [0.5, 0.5, 0.],
    ])
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return vertices, faces


def test_avoids_high_rom_valley():
    xyz, faces = fixture()
    costs = np.array([0., 0., 0., 0., 100.])
    route = minimum_rom_relief(xyz, faces, costs, 0, 2)
    assert 4 not in route.vertex_path
    assert route.vertex_path in ((0, 1, 2), (0, 3, 2))


def test_cut_duplicates_vertices_and_preserves_faces():
    xyz, faces = fixture()
    route = minimum_rom_relief(xyz, faces, np.zeros(len(xyz)), 0, 2)
    new_xyz, new_faces, source = open_mesh_along_route(xyz, faces, route)
    assert new_faces.shape == faces.shape
    np.testing.assert_array_equal(source[new_faces], faces)
    np.testing.assert_allclose(new_xyz, xyz[source])
    assert len(new_xyz) > len(xyz)


def test_invalid_rom_cost_rejected():
    xyz, faces = fixture()
    with pytest.raises(ValueError, match="ROM costs"):
        minimum_rom_relief(xyz, faces, np.array([0., -1., 0., 0., 0.]), 0, 2)


def test_nonexistent_cut_edge_rejected():
    from smii.garment.rom_relief import ReliefRoute

    xyz, faces = fixture()
    route = ReliefRoute((0, 2), ((0, 2),), 1.0)
    with pytest.raises(ValueError, match="absent"):
        open_mesh_along_route(xyz, faces, route)
