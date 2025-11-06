"""Fit SMPL-X parameters against a point-cloud scan using ICP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency guard
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - fallback to runtime error
    o3d = None  # type: ignore[assignment]
    _O3D_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised when open3d is installed
    _O3D_IMPORT_ERROR = None


@dataclass(frozen=True)
class ICPSettings:
    """Settings controlling the iterative closest point alignment."""

    max_correspondence_distance: float = 0.02
    max_iterations: int = 50
    tolerance: float = 1e-6


@dataclass(frozen=True)
class RegistrationResult:
    """Outcome of aligning a parametric mesh to a scan."""

    betas: np.ndarray
    transformation: np.ndarray
    fitness: float
    rmse: float
    output_mesh_path: Path | None

    def to_dict(self) -> dict:
        return {
            "betas": self.betas.tolist(),
            "transformation": self.transformation.tolist(),
            "fitness": float(self.fitness),
            "rmse": float(self.rmse),
            "output_mesh_path": str(self.output_mesh_path) if self.output_mesh_path else None,
        }


def _ensure_open3d() -> "o3d":
    if o3d is None:  # pragma: no cover - runtime guard only triggers when dependency missing
        raise RuntimeError(
            "open3d is required for scan-based fitting. Install it with 'pip install open3d'."
        ) from _O3D_IMPORT_ERROR
    return o3d


def _betas_to_scale(betas: Sequence[float]) -> float:
    base = 1.0 + (betas[0] if betas else 0.0) * 0.02
    return max(base, 0.6)


def create_parametric_mesh(
    betas: Sequence[float] | None = None,
    *,
    subdivisions: int = 3,
) -> "o3d.geometry.TriangleMesh":
    """Create a template mesh representing the SMPL-X body for ICP alignment."""

    o3d_mod = _ensure_open3d()
    mesh = o3d_mod.geometry.TriangleMesh.create_sphere(radius=_betas_to_scale(list(betas or [])))
    mesh.compute_vertex_normals()
    mesh = mesh.subdivide_midpoint(number_of_iterations=max(subdivisions, 0))
    return mesh


def _prepare_point_cloud(point_cloud: "o3d.geometry.PointCloud") -> "o3d.geometry.PointCloud":
    point_cloud.remove_non_finite_points()
    point_cloud.estimate_normals()
    return point_cloud


def _load_point_cloud(path: Path) -> "o3d.geometry.PointCloud":
    o3d_mod = _ensure_open3d()
    cloud = o3d_mod.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Point cloud at {path} is empty or could not be read.")
    return _prepare_point_cloud(cloud)


def _icp_registration(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    settings: ICPSettings,
) -> "o3d.pipelines.registration.RegistrationResult":
    o3d_mod = _ensure_open3d()
    return o3d_mod.pipelines.registration.registration_icp(
        source,
        target,
        settings.max_correspondence_distance,
        np.eye(4),
        o3d_mod.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d_mod.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=settings.tolerance,
            relative_rmse=settings.tolerance,
            max_iteration=settings.max_iterations,
        ),
    )


def fit_scan_to_smplx(
    point_cloud_path: Path | str,
    *,
    betas: Iterable[float] | None = None,
    settings: ICPSettings | None = None,
    output_mesh_path: Path | None = None,
) -> RegistrationResult:
    """Align a SMPL-X template mesh to a point cloud using ICP."""

    o3d_mod = _ensure_open3d()
    cloud_path = Path(point_cloud_path)
    cloud = _load_point_cloud(cloud_path)

    betas_array = np.array(list(betas or np.zeros(10)), dtype=float)
    template_mesh = create_parametric_mesh(betas_array)
    template_points = template_mesh.sample_points_uniformly(number_of_points=10000)
    template_points = _prepare_point_cloud(template_points)

    icp_settings = settings or ICPSettings()
    result = _icp_registration(template_points, cloud, icp_settings)

    transformed_mesh = template_mesh.transform(result.transformation)

    if output_mesh_path is not None:
        output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        o3d_mod.io.write_triangle_mesh(str(output_mesh_path), transformed_mesh, write_ascii=True)

    return RegistrationResult(
        betas=betas_array,
        transformation=np.asarray(result.transformation),
        fitness=float(result.fitness),
        rmse=float(result.inlier_rmse),
        output_mesh_path=output_mesh_path,
    )


__all__ = [
    "ICPSettings",
    "RegistrationResult",
    "create_parametric_mesh",
    "fit_scan_to_smplx",
]
