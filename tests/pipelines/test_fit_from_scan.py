from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smii.pipelines import fit_from_scan as scan_pipeline
from tests.helpers import FakeOpen3DModule, FakePointCloud, FakeRegistrationResult


def test_ensure_open3d_raises_when_dependency_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scan_pipeline, "o3d", None)
    monkeypatch.setattr(scan_pipeline, "_O3D_IMPORT_ERROR", ImportError("open3d not installed"))

    with pytest.raises(RuntimeError, match="open3d is required"):
        scan_pipeline._ensure_open3d()


def test_load_point_cloud_rejects_empty_scans(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_o3d = FakeOpen3DModule(point_cloud=FakePointCloud(empty=True))
    monkeypatch.setattr(scan_pipeline, "o3d", fake_o3d)
    monkeypatch.setattr(scan_pipeline, "_O3D_IMPORT_ERROR", None)

    with pytest.raises(ValueError, match="Point cloud.*empty"):
        scan_pipeline._load_point_cloud(tmp_path / "scan.ply")


def test_fit_scan_to_smplx_returns_registration_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    transformation = np.eye(4)
    transformation[0, 3] = 0.05
    registration = FakeRegistrationResult(transformation=transformation, fitness=0.87, inlier_rmse=0.004)
    point_cloud = FakePointCloud(empty=False)
    fake_o3d = FakeOpen3DModule(point_cloud=point_cloud, registration_result=registration)
    monkeypatch.setattr(scan_pipeline, "o3d", fake_o3d)
    monkeypatch.setattr(scan_pipeline, "_O3D_IMPORT_ERROR", None)

    original_create_mesh = scan_pipeline.create_parametric_mesh

    def safe_create_mesh(betas=None, *, subdivisions: int = 3):
        if isinstance(betas, np.ndarray):
            betas = betas.tolist()
        return original_create_mesh(betas, subdivisions=subdivisions)

    monkeypatch.setattr(scan_pipeline, "create_parametric_mesh", safe_create_mesh)

    output_path = tmp_path / "aligned_mesh.ply"
    result = scan_pipeline.fit_scan_to_smplx(
        tmp_path / "input_scan.ply",
        betas=[0.1, -0.2],
        output_mesh_path=output_path,
    )

    assert isinstance(result, scan_pipeline.RegistrationResult)
    assert result.betas.shape == (2,)
    assert result.betas[0] == pytest.approx(0.1)
    assert result.betas[1] == pytest.approx(-0.2)
    assert np.allclose(result.transformation, transformation)
    assert result.fitness == pytest.approx(registration.fitness)
    assert result.rmse == pytest.approx(registration.inlier_rmse)
    assert result.output_mesh_path == output_path

    assert fake_o3d.read_paths == [str(tmp_path / "input_scan.ply")]
    assert fake_o3d.written_meshes and fake_o3d.written_meshes[0][0] == str(output_path)
    assert point_cloud.removed_non_finite is True
    assert point_cloud.normals_estimated is True
