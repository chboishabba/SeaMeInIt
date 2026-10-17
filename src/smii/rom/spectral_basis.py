"""Spectral eigenmode solver using discrete cotangent Laplace-Beltrami operator.

Computes intrinsic mass-orthonormal harmonic eigenmodes on 3D meshes:
    L * phi_k = lambda_k * M * phi_k
    phi_i^T * M * phi_j = delta_ij
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from .cotangent_laplacian import CotangentLaplacianResult, build_cotangent_laplacian


@dataclass(frozen=True, slots=True)
class SpectralBasisConfig:
    """Configuration for spectral Laplace-Beltrami eigenmode extraction."""

    components: int = 64
    mass_scheme: str = "barycentric_lumped"
    boundary_condition: str = "neumann"
    eigen_tolerance: float = 1e-8
    degeneracy_tolerance: float = 1e-12
    shift: float = 1e-6  # small shift to avoid singularity at lambda=0


@dataclass(frozen=True, slots=True)
class SpectralBasisResult:
    """Computed spectral eigenmodes and numerical verification diagnostics."""

    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    mass_diagonal: NDArray[np.float64]
    residuals: NDArray[np.float64]
    orthogonality_error: float
    mesh_sha256: str
    config: SpectralBasisConfig
    laplacian_diagnostics: CotangentLaplacianResult


def _sha256_arrays(v: NDArray[np.float64], f: NDArray[np.int64]) -> str:
    h = hashlib.sha256()
    h.update(v.tobytes())
    h.update(f.tobytes())
    return h.hexdigest()


def build_spectral_basis(
    vertices: NDArray[np.float64] | Sequence[Sequence[float]],
    faces: NDArray[np.int64] | Sequence[Sequence[int]],
    config: SpectralBasisConfig | None = None,
) -> SpectralBasisResult:
    """Compute the discrete cotangent Laplace-Beltrami spectral basis on a triangular mesh.

    Args:
        vertices: Array of shape (N, 3) 3D vertex positions.
        faces: Array of shape (F, 3) triangle indices.
        config: SpectralBasisConfig options.

    Returns:
        SpectralBasisResult with eigenvalues, mass-orthonormal eigenvectors, and diagnostics.
    """
    if config is None:
        config = SpectralBasisConfig()

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    lap_res = build_cotangent_laplacian(v, f, area_eps=config.degeneracy_tolerance)
    n = lap_res.vertex_count
    k = min(config.components, n)

    if k < 1:
        raise ValueError(
            f"Number of requested components must be positive; got {config.components}"
        )

    # Use lumped barycentric mass diagonal m
    m_diag = lap_res.mass_diagonal.copy()
    # Guard against isolated or zero-area vertices
    zero_mass_mask = m_diag <= config.degeneracy_tolerance
    if np.any(zero_mass_mask):
        mean_mass = np.mean(m_diag[~zero_mass_mask]) if np.any(~zero_mass_mask) else 1.0
        m_diag[zero_mass_mask] = mean_mass * 1e-6

    # Symmetric standard transform:
    # Let L_sym = M^(-1/2) * L * M^(-1/2)
    # Then L_sym * psi = lambda * psi
    # and phi = M^(-1/2) * psi satisfies phi^T * M * phi = psi^T * psi = I
    inv_sqrt_m = 1.0 / np.sqrt(m_diag)
    inv_sqrt_m_mat = sp.diags(inv_sqrt_m, 0, shape=(n, n), format="csr")

    l_sym = inv_sqrt_m_mat @ lap_res.stiffness @ inv_sqrt_m_mat
    l_sym = 0.5 * (l_sym + l_sym.T)  # Ensure exact numerical symmetry

    if n <= 300:
        # For small meshes, solve dense standard symmetric eigenvalue problem
        dense_l = l_sym.toarray()
        evals_all, evecs_all = la.eigh(dense_l)
        # Sort ascending
        idx = np.argsort(evals_all)
        eigenvalues = evals_all[idx[:k]]
        psi = evecs_all[:, idx[:k]]
    else:
        # For large meshes, use shift-invert Arnoldi / Lanczos
        # L_sym is positive semi-definite; shift slightly to avoid zero-division
        shift = config.shift
        shifted_l = l_sym + shift * sp.eye(n, format="csr")
        try:
            evals, psi = spla.eigsh(
                shifted_l,
                k=k,
                sigma=-shift,
                which="LM",
                tol=config.eigen_tolerance,
            )
            eigenvalues = evals - shift
        except Exception:
            # Fallback to direct smallest algebraic if shift-invert encounters singular factorization
            evals, psi = spla.eigsh(
                l_sym,
                k=k,
                which="SA",
                tol=config.eigen_tolerance,
            )
            eigenvalues = evals

        # Ensure sorted order
        sort_order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_order]
        psi = psi[:, sort_order]

    # Convert normalized psi back to phi: phi = M^(-1/2) * psi
    eigenvectors = inv_sqrt_m[:, np.newaxis] * psi

    # Clamp near-zero ground state eigenvalue to 0.0 (constant mode on closed or Neumann mesh)
    if len(eigenvalues) > 0 and abs(eigenvalues[0]) < 1e-9:
        eigenvalues[0] = 0.0

    # Ensure consistent sign convention: first non-zero entry is positive
    for j in range(k):
        col = eigenvectors[:, j]
        max_idx = np.argmax(np.abs(col))
        if col[max_idx] < 0:
            eigenvectors[:, j] = -col

    # Verify M-orthogonality error: || Phi^T * M * Phi - I ||_max
    m_mat = sp.diags(m_diag, 0, shape=(n, n), format="csr")
    ortho_matrix = eigenvectors.T @ (m_mat @ eigenvectors)
    orthogonality_error = float(np.max(np.abs(ortho_matrix - np.eye(k))))

    # Compute residual norms || L * phi_j - lambda_j * M * phi_j || / max(1, lambda_j)
    residuals = np.zeros(k, dtype=np.float64)
    l_phi = lap_res.stiffness @ eigenvectors
    m_phi = m_mat @ eigenvectors
    for j in range(k):
        diff = l_phi[:, j] - eigenvalues[j] * m_phi[:, j]
        scale = max(1.0, float(abs(eigenvalues[j])))
        residuals[j] = float(np.linalg.norm(diff) / scale)

    mesh_hash = _sha256_arrays(v, f)

    return SpectralBasisResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        mass_diagonal=m_diag,
        residuals=residuals,
        orthogonality_error=orthogonality_error,
        mesh_sha256=mesh_hash,
        config=config,
        laplacian_diagnostics=lap_res,
    )
