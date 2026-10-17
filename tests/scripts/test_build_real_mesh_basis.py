from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scripts.build_real_mesh_basis import build_real_mesh_basis
from smii.meshing.body_carrier_receipt import BodyCarrierReceipt
from smii.rom.basis import load_basis


def _write_mock_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # Octahedron mesh
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int64,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vertices=vertices, faces=faces)
    return vertices, faces


def _write_carrier_receipt(path: Path, promotion: int = 1) -> None:
    receipt = BodyCarrierReceipt(
        source_hash="a" * 64,
        raw_reprojection_hash="b" * 64,
        refined_pre_repair_hash="c" * 64,
        repaired_export_hash="d" * 64,
        vertex_count=6,
        face_count=8,
        topology_label="octahedron_v6",
        landmark_residuals={},
        skull_rigidity_residual=0.01,
        body_fit_confidence=0.95,
        promotion=promotion,
        blocked_consumers=[],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt.to_json(path)


def test_build_real_mesh_basis_promotes(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.npz"
    receipt_in = tmp_path / "body_carrier_receipt.json"
    basis_out = tmp_path / "basis.npz"
    receipt_out = tmp_path / "basis_receipt.json"

    _write_mock_mesh(mesh_path)
    _write_carrier_receipt(receipt_in, promotion=1)

    res = build_real_mesh_basis(
        mesh_path=mesh_path,
        body_receipt_path=receipt_in,
        components=4,
        out_basis=basis_out,
        out_receipt=receipt_out,
    )

    assert res["promotion"] == 1
    assert res["components"] == 4
    assert res["orthogonality_error"] < 1e-8
    assert basis_out.exists()
    assert receipt_out.exists()

    # Load back with standard KernelBasis loader
    basis = load_basis(basis_out)
    assert basis.metadata.vertex_count == 6
    assert basis.metadata.component_count == 4
    assert basis.metadata.normalization == "mass_orthonormal_cotangent_laplacian"

    # Verify JSON receipt
    data = json.loads(receipt_out.read_text(encoding="utf-8"))
    assert data["promotion"] == 1
    assert data["construction_method"] == "cotangent_laplace_beltrami_spectral_v1"
    assert data["basis_dimension"] == 4
