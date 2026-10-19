"""Kernel basis loading and projection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping

import numpy as np

ArrayLike = np.ndarray | list[float]


@dataclass(frozen=True, slots=True)
class BasisMetadata:
    """Describes the canonical kernel basis."""

    vertex_count: int
    component_count: int
    source_mesh: str | None = None
    normalization: str | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class KernelBasis:
    """In-memory representation of the body kernel basis."""

    matrix: np.ndarray
    vertices: np.ndarray | None
    metadata: BasisMetadata
    mass_diagonal: np.ndarray | None = None

    @classmethod
    def from_arrays(
        cls,
        matrix: ArrayLike,
        *,
        vertices: ArrayLike | None = None,
        mass_diagonal: ArrayLike | None = None,
        source_mesh: str | None = None,
        normalization: str | None = None,
        notes: str | None = None,
    ) -> "KernelBasis":
        basis = np.asarray(matrix, dtype=float)
        if basis.ndim != 2:
            raise ValueError("Basis matrix must be 2-D (N_vertices, K_components).")
        if not np.isfinite(basis).all():
            raise ValueError("Basis matrix contains non-finite entries.")

        vertices_arr = None
        if vertices is not None:
            vertices_arr = np.asarray(vertices, dtype=float)
            if vertices_arr.shape != (basis.shape[0], 3):
                raise ValueError(
                    "Vertices must be shaped (N_vertices, 3) and align with the basis rows."
                )
            if not np.isfinite(vertices_arr).all():
                raise ValueError("Vertex coordinates must be finite.")

        mass_arr = None
        if mass_diagonal is not None:
            mass_arr = np.asarray(mass_diagonal, dtype=float)
            if mass_arr.shape != (basis.shape[0],):
                raise ValueError("Mass diagonal must match the number of vertices.")

        metadata = BasisMetadata(
            vertex_count=basis.shape[0],
            component_count=basis.shape[1],
            source_mesh=source_mesh,
            normalization=normalization,
            notes=notes,
        )
        return cls(matrix=basis, vertices=vertices_arr, metadata=metadata, mass_diagonal=mass_arr)

    @classmethod
    def from_mesh(
        cls,
        vertices: ArrayLike,
        faces: ArrayLike,
        *,
        components: int = 64,
        config: object | None = None,
        source_mesh: str | None = None,
        notes: str | None = None,
    ) -> "KernelBasis":
        from .spectral_basis import SpectralBasisConfig, build_spectral_basis

        cfg = (
            config
            if isinstance(config, SpectralBasisConfig)
            else SpectralBasisConfig(components=components)
        )
        spectral_res = build_spectral_basis(vertices, faces, config=cfg)
        return cls.from_arrays(
            spectral_res.eigenvectors,
            vertices=vertices,
            mass_diagonal=spectral_res.mass_diagonal,
            source_mesh=source_mesh,
            normalization="mass_orthonormal_cotangent_laplacian",
            notes=notes
            or f"spectral cotangent laplace-beltrami, ortho_err={spectral_res.orthogonality_error:.2e}",
        )


def load_basis(path: str | Path) -> KernelBasis:
    """Load a kernel basis from an NPZ file."""

    npz_path = Path(path)
    payload = np.load(npz_path, allow_pickle=True)
    if "basis" not in payload:
        raise KeyError(f"Basis NPZ '{npz_path}' missing required 'basis' array.")
    basis = payload["basis"]

    vertices = payload["vertices"] if "vertices" in payload else None
    meta_dict: MutableMapping[str, str] = {}
    if "meta" in payload:
        meta_raw = (
            payload["meta"].item()
            if getattr(payload["meta"], "shape", ()) == ()
            else payload["meta"]
        )
        if isinstance(meta_raw, Mapping):
            meta_dict.update({str(k): str(v) for k, v in meta_raw.items()})

    return KernelBasis.from_arrays(
        basis,
        vertices=vertices,
        source_mesh=meta_dict.get("source_mesh"),
        normalization=meta_dict.get("normalization"),
        notes=meta_dict.get("notes"),
    )


class KernelProjector:
    """Fast projection of kernel coefficients to body fields."""

    def __init__(self, basis: KernelBasis) -> None:
        self.basis = basis

    @property
    def vertex_count(self) -> int:
        return self.basis.metadata.vertex_count

    @property
    def component_count(self) -> int:
        return self.basis.metadata.component_count

    def project(self, coeffs: ArrayLike) -> np.ndarray:
        """Project a single coefficient vector to a body field."""

        arr = np.asarray(coeffs, dtype=float)
        if arr.ndim != 1:
            raise ValueError("Coefficient vector must be 1-D.")
        if arr.shape[0] != self.component_count:
            raise ValueError(
                f"Coefficient length {arr.shape[0]} does not match basis components "
                f"{self.component_count}."
            )
        return self.basis.matrix @ arr

    def project_batch(self, coeffs: ArrayLike) -> np.ndarray:
        """Project a batch of coefficient vectors shaped (K, M)."""

        arr = np.asarray(coeffs, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Batch coefficients must be 2-D (K_components, batch).")
        if arr.shape[0] != self.component_count:
            raise ValueError(
                f"Batch coefficient dimension {arr.shape[0]} does not match basis components "
                f"{self.component_count}."
            )
        return self.basis.matrix @ arr

    def project_mapping(self, coeffs: Mapping[str, ArrayLike]) -> dict[str, np.ndarray]:
        """Project multiple named coefficient vectors."""

        return {name: self.project(vector) for name, vector in coeffs.items()}

    def encode(self, field: ArrayLike) -> np.ndarray:
        """Encode a body field back into coefficient space."""
        arr = np.asarray(field, dtype=float)
        if arr.ndim != 1:
            raise ValueError("Field vector must be 1-D.")
        if arr.shape[0] != self.vertex_count:
            raise ValueError(
                f"Field length {arr.shape[0]} does not match basis vertices {self.vertex_count}."
            )
        if self.basis.mass_diagonal is not None:
            return self.basis.matrix.T @ (self.basis.mass_diagonal * arr)
        return self.basis.matrix.T @ arr

    def encode_batch(self, fields: ArrayLike) -> np.ndarray:
        """Encode a batch of body fields shaped (N_vertices, batch)."""
        arr = np.asarray(fields, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Batch fields must be 2-D (N_vertices, batch).")
        if arr.shape[0] != self.vertex_count:
            raise ValueError(
                f"Batch field vertex dimension {arr.shape[0]} does not match basis vertices "
                f"{self.vertex_count}."
            )
        if self.basis.mass_diagonal is not None:
            return self.basis.matrix.T @ (self.basis.mass_diagonal[:, np.newaxis] * arr)
        return self.basis.matrix.T @ arr
