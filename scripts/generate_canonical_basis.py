"""Generate a canonical kernel basis from mesh vertices using sinusoidal features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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
    features = _build_features(vertices, harmonics)
    if components > features.shape[1]:
        components = features.shape[1]
    return _orthonormalize(features, components)


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
    parser.add_argument("--source-mesh", type=str, default=None, help="Optional source mesh identifier for metadata.")
    parser.add_argument("--notes", type=str, default=None, help="Optional notes string to store alongside the basis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vertices = _load_vertices(args.vertices)
    basis = generate_basis(vertices, harmonics=args.harmonics, components=args.components)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "source_mesh": args.source_mesh or args.vertices.name,
        "normalization": "qr-orthonormalized",
        "notes": args.notes or "Sinusoidal features with harmonics",
    }
    np.savez_compressed(args.output, basis=basis, vertices=vertices, meta=meta)
    print(f"Saved basis with shape {basis.shape} to {args.output}")


if __name__ == "__main__":
    main()
