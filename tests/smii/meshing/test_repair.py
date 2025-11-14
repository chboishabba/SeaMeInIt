from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from smii.meshing.repair import repair_mesh_with_pymeshfix


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
    monkeypatch.delitem(sys.modules, "pymeshfix", raising=False)
    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    assert repair_mesh_with_pymeshfix(vertices, faces) is None


def test_repair_mesh_with_pymeshfix_warns_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy_pymeshfix(monkeypatch, fail=True)
    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with pytest.warns(RuntimeWarning):
        assert repair_mesh_with_pymeshfix(vertices, faces) is None
