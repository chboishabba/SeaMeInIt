from __future__ import annotations

import numpy as np
import pytest
from smii.rom.basis import KernelBasis, KernelProjector
from smii.rom.cotangent_laplacian import build_cotangent_laplacian
from smii.rom.spectral_basis import SpectralBasisConfig, build_spectral_basis


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    """Generate vertices and faces of a regular icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    # Normalize vertices to unit sphere
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)

    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _planar_grid(nx: int = 5, ny: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Generate vertices and faces of an open planar square grid."""
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xv, yv = np.meshgrid(x, y)
    vertices = np.column_stack([xv.flatten(), yv.flatten(), np.zeros(nx * ny)])

    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v0 = j * nx + i
            v1 = v0 + 1
            v2 = (j + 1) * nx + i
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    return vertices, np.array(faces, dtype=np.int64)


def test_cotangent_laplacian_properties() -> None:
    vertices, faces = _icosahedron()
    res = build_cotangent_laplacian(vertices, faces)

    assert res.vertex_count == 12
    assert res.face_count == 20
    assert res.degenerate_face_count == 0
    assert res.isolated_vertex_count == 0
    assert res.total_area > 0.0

    # L is symmetric
    l_mat = res.stiffness.toarray()
    np.testing.assert_allclose(l_mat, l_mat.T, atol=1e-12)

    # Row sums of L are identically zero (L * 1 = 0)
    row_sums = np.sum(l_mat, axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    # Diagonal mass is positive and sums to total area
    assert np.all(res.mass_diagonal > 0.0)
    np.testing.assert_allclose(np.sum(res.mass_diagonal), res.total_area, rtol=1e-10)


def test_spectral_basis_closed_surface() -> None:
    vertices, faces = _icosahedron()
    config = SpectralBasisConfig(components=8)
    res = build_spectral_basis(vertices, faces, config=config)

    assert len(res.eigenvalues) == 8
    assert res.eigenvectors.shape == (12, 8)

    # Eigenvalues are non-negative and monotonically ascending
    assert res.eigenvalues[0] == pytest.approx(0.0, abs=1e-8)
    assert np.all(np.diff(res.eigenvalues) >= -1e-10)

    # First eigenvector (ground state) is constant
    ground_mode = res.eigenvectors[:, 0]
    std_val = float(np.std(ground_mode))
    assert std_val < 1e-10

    # Mass-orthonormality: Phi^T * M * Phi = I
    assert res.orthogonality_error < 1e-10

    # Residuals: || L * phi - lambda * M * phi || / scale < 1e-8
    assert np.all(res.residuals < 1e-8)


def test_spectral_basis_open_surface() -> None:
    vertices, faces = _planar_grid(nx=6, ny=6)
    config = SpectralBasisConfig(components=10)
    res = build_spectral_basis(vertices, faces, config=config)

    assert len(res.eigenvalues) == 10
    assert res.eigenvectors.shape == (36, 10)

    # Neumann ground state is 0.0
    assert res.eigenvalues[0] == pytest.approx(0.0, abs=1e-8)
    assert np.all(np.diff(res.eigenvalues) >= -1e-10)

    # M-orthonormality holds on open surfaces
    assert res.orthogonality_error < 1e-10
    assert np.all(res.residuals < 1e-8)


def test_kernel_basis_from_mesh_and_projector() -> None:
    vertices, faces = _planar_grid(nx=5, ny=5)
    basis = KernelBasis.from_mesh(vertices, faces, components=8)

    assert basis.metadata.vertex_count == 25
    assert basis.metadata.component_count == 8
    assert basis.metadata.normalization == "mass_orthonormal_cotangent_laplacian"
    assert basis.mass_diagonal is not None

    projector = KernelProjector(basis)

    # Generate synthetic coefficients and test round-trip reconstruction
    c_orig = np.array([1.5, -0.8, 0.4, 0.1, -0.2, 0.9, -0.5, 0.3])
    field = projector.project(c_orig)
    c_recon = projector.encode(field)

    np.testing.assert_allclose(c_recon, c_orig, atol=1e-10)

    # Batch round-trip
    c_batch = np.random.RandomState(42).randn(8, 5)
    fields_batch = projector.project_batch(c_batch)
    c_batch_recon = projector.encode_batch(fields_batch)

    np.testing.assert_allclose(c_batch_recon, c_batch, atol=1e-10)


def test_degenerate_triangle_handling() -> None:
    # Triangle with collinear vertices (area = 0)
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # Collinear!
            [0.5, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],  # Degenerate (flat line)
            [0, 1, 3],  # Valid
            [1, 2, 3],  # Valid
        ],
        dtype=np.int64,
    )

    res = build_cotangent_laplacian(vertices, faces)
    assert res.degenerate_face_count == 1
    assert np.isfinite(res.stiffness.toarray()).all()
    assert np.isfinite(res.mass_diagonal).all()
