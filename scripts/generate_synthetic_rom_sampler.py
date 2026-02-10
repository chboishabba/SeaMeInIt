#!/usr/bin/env python3
"""Generate a deterministic synthetic ROM sampler for plumbing tests.

This script produces a sampler JSON that matches the ROM aggregator schema but
marks itself synthetic. Coefficients are random draws from a fixed seed to
exercise basis projection paths without real ROM data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_vertices(path: Path) -> np.ndarray:
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
    return vertices


def generate_sampler(
    body_path: Path, *, components: int, sample_count: int, seed: int
) -> dict:
    vertices = _load_vertices(body_path)
    vertex_count = int(vertices.shape[0])
    rng = np.random.default_rng(seed)

    samples: list[dict] = []
    for idx in range(sample_count):
        coeffs = {
            "shear": rng.normal(0.0, 1.0, size=components).tolist(),
            "tension": rng.normal(0.0, 0.5, size=components).tolist(),
        }
        samples.append(
            {
                "id": f"synthetic_{idx:03d}",
                "accepted": True,
                "coeffs": coeffs,
            }
        )

    payload = {
        "meta": {
            "mesh": body_path.stem,
            "vertex_count": vertex_count,
            "component_count": components,
            "synthetic": True,
            "notes": "Plumbing-only synthetic ROM sampler",
        },
        "samples": samples,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--body",
        required=True,
        type=Path,
        help="Path to body npz/npy with vertices (and faces optional).",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=32,
        help="Coefficient length K for the synthetic sampler (default: 32).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="Number of synthetic samples to emit (default: 8).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output path for sampler JSON (gitignored recommended).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for deterministic coefficients (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = generate_sampler(
        args.body, components=args.components, sample_count=args.samples, seed=args.seed
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Wrote synthetic sampler → {args.out} "
        f"(components={args.components}, samples={args.samples}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()
