from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from smii.meshing.repair import repair_body_mesh_for_export, repair_mesh_with_pymeshfix


def _install_dummy_pymeshfix(monkeypatch: pytest.MonkeyPatch, *, fail: bool = False) -> None:
    class DummyMeshFix:
        def __init__(self, vertices, faces) -> None:
            self._vertices = np.asarray(vertices, dtype=np.float64)
            self._faces = np.asarray(faces, dtype=np.int64)
            self.v = self._vertices.copy()
            self.f = self._faces.copy()
            self._fail = fail

        def repair(self, **_: object) -> None:
            if self._fail:
                raise RuntimeError("repair failed")
            self.v = self._vertices + 1.0
            self.f = self._faces

    module = ModuleType("pymeshfix")
    module.MeshFix = DummyMeshFix
    monkeypatch.setitem(sys.modules, "pymeshfix", module)


def _install_signature_limited_pymeshfix(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyMeshFix:
        def __init__(self, vertices, faces) -> None:
            self._vertices = np.asarray(vertices, dtype=np.float64)
            self._faces = np.asarray(faces, dtype=np.int64)
            self.v = self._vertices.copy()
            self.f = self._faces.copy()

        def repair(self, joincomp: bool = False, remove_smallest_components: bool = True) -> None:
            assert joincomp is False
            assert remove_smallest_components is True
            self.v = self._vertices + 2.0
            self.f = self._faces

    module = ModuleType("pymeshfix")
    module.MeshFix = DummyMeshFix
    monkeypatch.setitem(sys.modules, "pymeshfix", module)


def test_repair_mesh_with_pymeshfix_returns_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy_pymeshfix(monkeypatch)
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    repaired = repair_mesh_with_pymeshfix(vertices, faces)

    assert repaired is not None
    fixed_vertices, fixed_faces = repaired
    np.testing.assert_allclose(fixed_vertices, vertices + 1.0)
    np.testing.assert_array_equal(fixed_faces, faces)
    assert fixed_vertices.dtype == np.float32
    assert fixed_faces.dtype == np.int32


def test_repair_mesh_with_pymeshfix_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pymeshfix", None)
    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    assert repair_mesh_with_pymeshfix(vertices, faces) is None


def test_repair_mesh_with_pymeshfix_warns_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy_pymeshfix(monkeypatch, fail=True)
    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with pytest.warns(RuntimeWarning):
        assert repair_mesh_with_pymeshfix(vertices, faces) is None


def test_repair_mesh_with_pymeshfix_handles_signature_limited_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_signature_limited_pymeshfix(monkeypatch)
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    repaired = repair_mesh_with_pymeshfix(vertices, faces, verbose=True, max_iters=8)

    assert repaired is not None
    fixed_vertices, fixed_faces = repaired
    np.testing.assert_allclose(fixed_vertices, vertices + 2.0)
    np.testing.assert_array_equal(fixed_faces, faces)


def test_repair_body_mesh_for_export_caps_primary_loop_and_drops_satellites() -> None:
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.update_faces(np.arange(len(box.faces) - 2))
    satellite_left = trimesh.creation.icosphere(subdivisions=1, radius=0.08)
    satellite_left.apply_translation((-1.5, 0.0, 0.0))
    satellite_right = trimesh.creation.icosphere(subdivisions=1, radius=0.08)
    satellite_right.apply_translation((1.5, 0.0, 0.0))

    vertices = np.vstack(
        (box.vertices, satellite_left.vertices, satellite_right.vertices)
    )
    faces = np.vstack(
        (
            box.faces,
            satellite_left.faces + len(box.vertices),
            satellite_right.faces + len(box.vertices) + len(satellite_left.vertices),
        )
    )

    repaired = repair_body_mesh_for_export(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
    )

    assert repaired is not None
    repaired_mesh = trimesh.Trimesh(vertices=repaired[0], faces=repaired[1], process=False)
    assert repaired_mesh.is_watertight
    assert len(repaired_mesh.split(only_watertight=False)) == 1
