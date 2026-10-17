#!/usr/bin/env python3
"""Generate an intrinsic cotangent Laplace-Beltrami spectral basis from real 3D mesh geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from smii.meshing import load_body_carrier_receipt
from smii.rom import BasisReceipt
from smii.rom.spectral_basis import SpectralBasisConfig, build_spectral_basis

DEFAULT_COMPONENTS = 64
DEFAULT_ORTHOGONALITY_TOLERANCE = 1e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Mesh file '{path}' does not exist.")

    payload = np.load(path)
    if "vertices" in payload and "faces" in payload:
        vertices = np.asarray(payload["vertices"], dtype=np.float64)
        faces = np.asarray(payload["faces"], dtype=np.int64)
    elif "v" in payload and "f" in payload:
        vertices = np.asarray(payload["v"], dtype=np.float64)
        faces = np.asarray(payload["f"], dtype=np.int64)
    else:
        raise KeyError(f"NPZ at '{path}' must contain ('vertices', 'faces') or ('v', 'f').")

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Vertices must be shaped (N, 3); got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Faces must be shaped (F, 3); got {faces.shape}")
    if not np.isfinite(vertices).all():
        raise ValueError("Vertices contain non-finite values.")

    return vertices, faces


def build_real_mesh_basis(
    *,
    mesh_path: Path,
    body_receipt_path: Path,
    components: int = DEFAULT_COMPONENTS,
    out_basis: Path,
    out_receipt: Path,
    orthogonality_tolerance: float = DEFAULT_ORTHOGONALITY_TOLERANCE,
) -> dict[str, Any]:
    vertices, faces = _load_mesh(mesh_path)
    carrier_receipt = load_body_carrier_receipt(body_receipt_path)
    carrier_hash = _sha256(body_receipt_path)

    config = SpectralBasisConfig(components=components)
    spectral_res = build_spectral_basis(vertices, faces, config=config)

    # Validate against carrier receipt
    blocked_consumers: list[str] = []
    if carrier_receipt.promotion != 1:
        blocked_consumers.append("unpromoted_carrier_receipt")
    if spectral_res.orthogonality_error > orthogonality_tolerance:
        blocked_consumers.append("orthogonality_error_exceeds_tolerance")
    if spectral_res.laplacian_diagnostics.isolated_vertex_count > 0:
        blocked_consumers.append("isolated_vertices_present")

    promotion = 1 if not blocked_consumers else 0

    meta = {
        "source_mesh": str(mesh_path),
        "source_mesh_hash": _sha256(mesh_path),
        "carrier_receipt_hash": carrier_hash,
        "normalization": "mass_orthonormal_cotangent_laplacian",
        "construction_method": "cotangent_laplace_beltrami_spectral_v1",
        "components": len(spectral_res.eigenvalues),
        "total_area": spectral_res.laplacian_diagnostics.total_area,
        "orthogonality_error": spectral_res.orthogonality_error,
        "max_residual": float(np.max(spectral_res.residuals)),
    }

    # Save NPZ bundle
    out_basis.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_basis,
        basis=spectral_res.eigenvectors,
        vertices=vertices,
        eigenvalues=spectral_res.eigenvalues,
        mass_diagonal=spectral_res.mass_diagonal,
        meta=meta,
    )

    # Save BasisReceipt
    basis_receipt = BasisReceipt(
        carrier_receipt_hash=carrier_hash,
        basis_vertex_count=len(vertices),
        basis_dimension=len(spectral_res.eigenvalues),
        construction_method="cotangent_laplace_beltrami_spectral_v1",
        reconstruction_error=float(spectral_res.orthogonality_error),
        promotion=promotion,
        blocked_consumers=blocked_consumers,
    )

    out_receipt.parent.mkdir(parents=True, exist_ok=True)
    basis_receipt.to_json(out_receipt)

    return {
        "promotion": promotion,
        "components": len(spectral_res.eigenvalues),
        "orthogonality_error": spectral_res.orthogonality_error,
        "max_residual": float(np.max(spectral_res.residuals)),
        "blocked_consumers": blocked_consumers,
        "basis_path": str(out_basis),
        "receipt_path": str(out_receipt),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh", type=Path, required=True, help="Input mesh NPZ containing vertices and faces"
    )
    parser.add_argument(
        "--body-receipt", type=Path, required=True, help="Path to body_carrier_receipt.json"
    )
    parser.add_argument(
        "--components", type=int, default=DEFAULT_COMPONENTS, help="Number of spectral modes"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ basis file")
    parser.add_argument(
        "--receipt-output", type=Path, required=True, help="Output JSON basis receipt file"
    )
    args = parser.parse_args()

    summary = build_real_mesh_basis(
        mesh_path=args.mesh,
        body_receipt_path=args.body_receipt,
        components=args.components,
        out_basis=args.output,
        out_receipt=args.receipt_output,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["promotion"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
