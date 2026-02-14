#!/usr/bin/env python3
"""Build persistent vertex correspondence maps between two mesh topologies."""

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
            raise KeyError(f"{path} does not contain 'vertices' or 'v'.")
    else:
        vertices = payload
    arr = np.asarray(vertices, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path} vertices must be shaped (N,3).")
    return arr


def _nearest_map(source: np.ndarray, target: np.ndarray, *, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Map each source vertex to nearest target vertex."""

    source_arr = np.asarray(source, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    indices = np.empty(source_arr.shape[0], dtype=np.int64)
    dists = np.empty(source_arr.shape[0], dtype=np.float64)
    step = max(1, int(batch_size))
    for start in range(0, source_arr.shape[0], step):
        end = min(source_arr.shape[0], start + step)
        block = source_arr[start:end]
        diff = block[:, None, :] - target_arr[None, :, :]
        dist2 = np.einsum("bij,bij->bi", diff, diff)
        nearest = np.argmin(dist2, axis=1)
        indices[start:end] = nearest.astype(np.int64)
        dists[start:end] = np.sqrt(dist2[np.arange(len(nearest)), nearest])
    return indices, dists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mesh", type=Path, required=True, help="Source mesh (.npz/.npy with vertices).")
    parser.add_argument("--target-mesh", type=Path, required=True, help="Target mesh (.npz/.npy with vertices).")
    parser.add_argument("--out", type=Path, required=True, help="Output map artifact (.npz).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for nearest-neighbor distance blocks (default: 256).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = _load_vertices(args.source_mesh)
    target = _load_vertices(args.target_mesh)

    source_to_target, source_to_target_dist = _nearest_map(source, target, batch_size=args.batch_size)
    target_to_source, target_to_source_dist = _nearest_map(target, source, batch_size=args.batch_size)

    source_unique_target = int(len(set(int(v) for v in source_to_target.tolist())))
    target_unique_source = int(len(set(int(v) for v in target_to_source.tolist())))

    source_collision_ratio = float(max(0, len(source_to_target) - source_unique_target) / max(1, len(source_to_target)))
    target_collision_ratio = float(max(0, len(target_to_source) - target_unique_source) / max(1, len(target_to_source)))

    meta = {
        "source_mesh": str(args.source_mesh),
        "target_mesh": str(args.target_mesh),
        "source_vertex_count": int(source.shape[0]),
        "target_vertex_count": int(target.shape[0]),
        "source_to_target_max_distance": float(np.max(source_to_target_dist)) if len(source_to_target_dist) else 0.0,
        "source_to_target_mean_distance": float(np.mean(source_to_target_dist)) if len(source_to_target_dist) else 0.0,
        "source_to_target_unique_targets": source_unique_target,
        "source_to_target_collision_ratio": source_collision_ratio,
        "target_to_source_max_distance": float(np.max(target_to_source_dist)) if len(target_to_source_dist) else 0.0,
        "target_to_source_mean_distance": float(np.mean(target_to_source_dist)) if len(target_to_source_dist) else 0.0,
        "target_to_source_unique_sources": target_unique_source,
        "target_to_source_collision_ratio": target_collision_ratio,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        source_to_target_indices=source_to_target,
        source_to_target_distances=source_to_target_dist,
        target_to_source_indices=target_to_source,
        target_to_source_distances=target_to_source_dist,
        meta=np.array(meta, dtype=object),
    )
    print(f"Wrote map artifact to {args.out}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
