"""Generate a canonical kernel basis from mesh vertices using sinusoidal features."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from smii.meshing import load_body_carrier_receipt
from smii.rom import BasisReceipt

DEFAULT_RECONSTRUCTION_ERROR_THRESHOLD_RATIO = 0.05


def _load_vertices(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Vertex file '{path}' does not exist.")

    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        if "vertices" in payload:
            vertices = payload["vertices"]
        elif "v" in payload:
            vertices = payload["v"]
        else:
            raise KeyError("NPZ must contain a 'vertices' or 'v' array.")
    else:
        vertices = payload

    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Vertices must be shaped (N, 3).")
    if not np.isfinite(vertices).all():
        raise ValueError("Vertices must be finite.")
    return vertices


def _build_features(vertices: np.ndarray, harmonics: int) -> np.ndarray:
    features = [
        np.ones((vertices.shape[0], 1), dtype=float),
        vertices,
        np.linalg.norm(vertices, axis=1, keepdims=True),
    ]
    for k in range(1, harmonics + 1):
        scaled = vertices * float(k)
        features.append(np.sin(scaled))
        features.append(np.cos(scaled))
    return np.concatenate(features, axis=1)


def _orthonormalize(features: np.ndarray, component_count: int) -> np.ndarray:
    q, _ = np.linalg.qr(features)
    usable = min(component_count, q.shape[1])
    return q[:, :usable]


def generate_basis(vertices: np.ndarray, *, harmonics: int, components: int) -> np.ndarray:
    if components <= 0:
        raise ValueError("components must be positive.")
    if harmonics < 0:
        raise ValueError("harmonics must be non-negative.")
    features = _build_features(vertices, harmonics)
    if components > features.shape[1]:
        components = features.shape[1]
    return _orthonormalize(features, components)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _static_contact_pressure_proxy(vertices: np.ndarray) -> np.ndarray:
    y = vertices[:, 1]
    pressure = float(np.max(y)) - y
    scale = float(np.max(np.abs(pressure)))
    if scale <= 0.0:
        pressure = np.linalg.norm(vertices - np.mean(vertices, axis=0), axis=1)
        scale = float(np.max(np.abs(pressure)))
    if scale <= 0.0:
        return np.ones(vertices.shape[0], dtype=float)
    return pressure / scale


def _relative_reconstruction_error(basis: np.ndarray, field: np.ndarray) -> float:
    coeffs = np.linalg.lstsq(basis, field, rcond=None)[0]
    residual = field - basis @ coeffs
    denominator = float(np.linalg.norm(field))
    if denominator <= 0.0:
        return float(np.linalg.norm(residual))
    return float(np.linalg.norm(residual) / denominator)


def emit_basis_receipt(
    *,
    body_receipt_path: Path,
    basis_path: Path,
    receipt_path: Path,
    vertices: np.ndarray,
    basis: np.ndarray,
    construction_method: str,
    reconstruction_error_threshold_ratio: float = DEFAULT_RECONSTRUCTION_ERROR_THRESHOLD_RATIO,
) -> BasisReceipt:
    body_receipt = load_body_carrier_receipt(body_receipt_path)
    if body_receipt.promotion != 1:
        raise ValueError(
            "BodyCarrierReceipt not promoted "
            f"(status={body_receipt.promotion}). "
            f"Blocked by: {body_receipt.blocked_consumers}"
        )
    if int(vertices.shape[0]) != int(body_receipt.vertex_count):
        raise ValueError(
            "Vertex count mismatch with BodyCarrierReceipt: "
            f"basis vertices={vertices.shape[0]}, receipt vertices={body_receipt.vertex_count}."
        )
    if reconstruction_error_threshold_ratio < 0.0:
        raise ValueError("reconstruction error threshold ratio must be non-negative.")

    test_field = _static_contact_pressure_proxy(vertices)
    reconstruction_error = _relative_reconstruction_error(basis, test_field)
    promotion = 1 if reconstruction_error <= reconstruction_error_threshold_ratio else 0

    receipt = BasisReceipt(
        carrier_receipt_hash=_sha256_file(body_receipt_path),
        basis_vertex_count=int(vertices.shape[0]),
        basis_dimension=int(basis.shape[1]),
        construction_method=construction_method,
        reconstruction_error=reconstruction_error,
        promotion=promotion,
        blocked_consumers=[] if promotion == 1 else [],
        basis_hash=_sha256_file(basis_path),
        promotion_threshold=float(reconstruction_error_threshold_ratio),
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt.to_json(receipt_path)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", required=True, type=Path, help="Path to vertex array (.npy or .npz).")
    parser.add_argument(
        "--components", type=int, default=64, help="Number of basis components to retain (default: 64)."
    )
    parser.add_argument(
        "--harmonics",
        type=int,
        default=3,
        help="Number of sinusoidal harmonics to include when building features (default: 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rom/canonical_basis.npz"),
        help="Output NPZ path for the generated basis (default: outputs/rom/canonical_basis.npz).",
    )
    parser.add_argument(
        "--body-receipt",
        type=Path,
        default=None,
        help="Promoted BodyCarrierReceipt JSON required to emit a BasisReceipt.",
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=None,
        help="Output path for basis_receipt.json (default: next to --output when --body-receipt is set).",
    )
    parser.add_argument(
        "--reconstruction-error-threshold-ratio",
        type=float,
        default=DEFAULT_RECONSTRUCTION_ERROR_THRESHOLD_RATIO,
        help="Relative reconstruction error threshold for BasisReceipt promotion (default: 0.05).",
    )
    parser.add_argument("--source-mesh", type=str, default=None, help="Optional source mesh identifier for metadata.")
    parser.add_argument("--notes", type=str, default=None, help="Optional notes string to store alongside the basis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.receipt_output is not None and args.body_receipt is None:
        raise ValueError("--receipt-output requires --body-receipt.")
    if args.body_receipt is not None:
        body_receipt = load_body_carrier_receipt(args.body_receipt)
        if body_receipt.promotion != 1:
            raise ValueError(
                "BodyCarrierReceipt not promoted "
                f"(status={body_receipt.promotion}). "
                f"Blocked by: {body_receipt.blocked_consumers}"
            )

    vertices = _load_vertices(args.vertices)
    basis = generate_basis(vertices, harmonics=args.harmonics, components=args.components)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "source_mesh": args.source_mesh or args.vertices.name,
        "source_path": str(args.vertices),
        "vertex_count": int(vertices.shape[0]),
        "normalization": "qr-orthonormalized",
        "notes": args.notes or "Sinusoidal features with harmonics",
        "construction_method": f"sinusoidal_qr_h{args.harmonics}",
    }
    np.savez_compressed(args.output, basis=basis, vertices=vertices, meta=meta)
    print(f"Saved basis with shape {basis.shape} to {args.output}")

    if args.body_receipt is not None:
        receipt_path = args.receipt_output or (args.output.parent / "basis_receipt.json")
        receipt = emit_basis_receipt(
            body_receipt_path=args.body_receipt,
            basis_path=args.output,
            receipt_path=receipt_path,
            vertices=vertices,
            basis=basis,
            construction_method=meta["construction_method"],
            reconstruction_error_threshold_ratio=args.reconstruction_error_threshold_ratio,
        )
        print(
            "Wrote basis receipt "
            f"(promotion={receipt.promotion}, reconstruction_error={receipt.reconstruction_error:.6f}) "
            f"to {receipt_path}"
        )


if __name__ == "__main__":
    main()
