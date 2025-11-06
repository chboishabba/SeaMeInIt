"""Lightweight fakes for the :mod:`open3d` API used in tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, List

import numpy as np


class FakePointCloud:
    """Minimal point cloud implementation used for testing."""

    def __init__(self, *, empty: bool = False) -> None:
        self._empty = empty
        self.removed_non_finite = False
        self.normals_estimated = False

    def is_empty(self) -> bool:
        return self._empty

    def remove_non_finite_points(self) -> "FakePointCloud":
        self.removed_non_finite = True
        return self

    def estimate_normals(self) -> "FakePointCloud":
        self.normals_estimated = True
        return self


class FakeTriangleMesh:
    """Simplified triangle mesh matching the subset of methods we exercise."""

    def __init__(self, radius: float = 1.0) -> None:
        self.radius = radius
        self.normals_computed = False
        self.subdivision_calls: List[int] = []
        self.applied_transformation = np.eye(4)

    @classmethod
    def create_sphere(cls, radius: float) -> "FakeTriangleMesh":
        return cls(radius)

    def compute_vertex_normals(self) -> "FakeTriangleMesh":
        self.normals_computed = True
        return self

    def subdivide_midpoint(self, number_of_iterations: int = 0) -> "FakeTriangleMesh":
        self.subdivision_calls.append(number_of_iterations)
        return self

    def sample_points_uniformly(self, number_of_points: int) -> FakePointCloud:
        return FakePointCloud()

    def transform(self, matrix: np.ndarray) -> "FakeTriangleMesh":
        self.applied_transformation = np.asarray(matrix)
        return self


@dataclass
class FakeRegistrationResult:
    """Container mirroring :class:`open3d.pipelines.registration.RegistrationResult`."""

    transformation: np.ndarray = field(default_factory=lambda: np.eye(4))
    fitness: float = 0.9
    inlier_rmse: float = 0.001


class FakeOpen3DModule:
    """Bundle the minimal pieces of the :mod:`open3d` API that the pipeline touches."""

    def __init__(
        self,
        *,
        point_cloud: FakePointCloud | None = None,
        registration_result: FakeRegistrationResult | None = None,
    ) -> None:
        self._point_cloud = point_cloud or FakePointCloud()
        self.registration_result = registration_result or FakeRegistrationResult()
        self.written_meshes: list[tuple[str, FakeTriangleMesh, bool]] = []
        self.read_paths: list[str] = []
        self.geometry = SimpleNamespace(TriangleMesh=FakeTriangleMesh, PointCloud=FakePointCloud)
        self.io = SimpleNamespace(
            read_point_cloud=self._read_point_cloud,
            write_triangle_mesh=self._write_triangle_mesh,
        )
        self.pipelines = SimpleNamespace(
            registration=SimpleNamespace(
                registration_icp=self._registration_icp,
                TransformationEstimationPointToPlane=lambda: object(),
                ICPConvergenceCriteria=lambda **kwargs: SimpleNamespace(**kwargs),
            )
        )

    def set_point_cloud(self, cloud: FakePointCloud) -> None:
        self._point_cloud = cloud

    def set_registration_result(self, result: FakeRegistrationResult) -> None:
        self.registration_result = result

    # -- API callbacks -------------------------------------------------
    def _read_point_cloud(self, path: str) -> FakePointCloud:
        self.read_paths.append(path)
        return self._point_cloud

    def _write_triangle_mesh(self, path: str, mesh: FakeTriangleMesh, write_ascii: bool = False) -> bool:
        self.written_meshes.append((path, mesh, write_ascii))
        return True

    def _registration_icp(
        self,
        source: FakePointCloud,
        target: FakePointCloud,
        max_correspondence_distance: float,
        *_: Any,
    ) -> FakeRegistrationResult:
        return self.registration_result


__all__ = [
    "FakeOpen3DModule",
    "FakePointCloud",
    "FakeRegistrationResult",
    "FakeTriangleMesh",
]
