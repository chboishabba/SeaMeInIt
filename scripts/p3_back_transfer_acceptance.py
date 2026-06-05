#!/usr/bin/env python3
"""Emit a P3 back-transfer acceptance receipt for a fixed forward object."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from smii.meshing import CorrespondenceReceipt, load_body_carrier_receipt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    payload = np.load(path, allow_pickle=True)
    if "vertices" not in payload:
        raise KeyError(f"Mesh NPZ '{path}' must contain a 'vertices' array.")
    vertices = np.asarray(payload["vertices"], dtype=float)
    faces = np.asarray(payload["faces"], dtype=int) if "faces" in payload else None
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Mesh vertices in '{path}' must be shaped (N, 3).")
    if faces is not None and (faces.ndim != 2 or faces.shape[1] != 3):
        raise ValueError(f"Mesh faces in '{path}' must be shaped (M, 3).")
    if not np.isfinite(vertices).all():
        raise ValueError(f"Mesh vertices in '{path}' must be finite.")
    return vertices, faces


def _nearest_map(
    source: np.ndarray,
    target: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.empty(source.shape[0], dtype=np.int64)
    distances = np.empty(source.shape[0], dtype=np.float64)
    step = max(1, int(batch_size))
    for start in range(0, source.shape[0], step):
        end = min(source.shape[0], start + step)
        block = source[start:end]
        diff = block[:, None, :] - target[None, :, :]
        dist2 = np.einsum("bij,bij->bi", diff, diff)
        nearest = np.argmin(dist2, axis=1)
        indices[start:end] = nearest.astype(np.int64)
        distances[start:end] = np.sqrt(dist2[np.arange(len(nearest)), nearest])
    return indices, distances


def _bbox_diagonal(vertices: np.ndarray) -> float:
    if len(vertices) == 0:
        return 0.0
    return float(np.linalg.norm(np.ptp(vertices, axis=0)))


def emit_p3_back_transfer_receipt(
    *,
    source_mesh_path: Path,
    target_mesh_path: Path,
    target_body_receipt_path: Path,
    output_receipt_path: Path,
    output_map_path: Path | None = None,
    source_topology_label: str = "B_v9438",
    target_topology_label: str | None = None,
    transfer_mode: str = "approximate_correspondence",
    max_distance_ratio_threshold: float = 0.08,
    collision_ratio_threshold: float = 0.80,
    seam_retention_threshold: float = 0.20,
    round_trip_retention_threshold: float = 0.20,
    batch_size: int = 256,
) -> CorrespondenceReceipt:
    """Compute nearest-neighbor transfer metrics and write a receipt."""

    source_vertices, source_faces = _load_mesh(source_mesh_path)
    target_vertices, target_faces = _load_mesh(target_mesh_path)
    target_body_receipt = load_body_carrier_receipt(target_body_receipt_path)
    if target_body_receipt.promotion != 1:
        raise ValueError(
            "Target BodyCarrierReceipt is not promoted "
            f"(status={target_body_receipt.promotion})."
        )
    if int(target_vertices.shape[0]) != int(target_body_receipt.vertex_count):
        raise ValueError(
            "Target mesh vertex count does not match BodyCarrierReceipt: "
            f"mesh={target_vertices.shape[0]}, receipt={target_body_receipt.vertex_count}."
        )

    source_to_target, source_to_target_dist = _nearest_map(
        source_vertices,
        target_vertices,
        batch_size=batch_size,
    )
    target_to_source, target_to_source_dist = _nearest_map(
        target_vertices,
        source_vertices,
        batch_size=batch_size,
    )

    unique_targets = int(len(set(int(v) for v in source_to_target.tolist())))
    source_count = int(source_vertices.shape[0])
    target_count = int(target_vertices.shape[0])
    collision_ratio = float(max(0, source_count - unique_targets) / max(1, source_count))
    retention_ratio = float(unique_targets / max(1, source_count))
    round_trip_indices = target_to_source[source_to_target]
    round_trip_retention = float(
        np.mean(round_trip_indices == np.arange(source_count, dtype=np.int64))
    )
    round_trip_distances = np.linalg.norm(
        source_vertices - source_vertices[round_trip_indices],
        axis=1,
    )
    bbox_diag = max(_bbox_diagonal(source_vertices), _bbox_diagonal(target_vertices), 1e-12)
    load_metric = float(np.percentile(source_to_target_dist, 95) / bbox_diag)

    promotes = (
        load_metric <= max_distance_ratio_threshold
        and collision_ratio <= collision_ratio_threshold
        and retention_ratio >= seam_retention_threshold
        and round_trip_retention >= round_trip_retention_threshold
    )
    notes = [
        "P3 back-transfer acceptance against exact historical forward object.",
        "Transfer is approximate correspondence/reprojection, not a geometric inverse.",
        f"source_faces={0 if source_faces is None else int(source_faces.shape[0])}; "
        f"target_faces={0 if target_faces is None else int(target_faces.shape[0])}",
    ]
    receipt = CorrespondenceReceipt(
        source_mesh_hash=_sha256_file(source_mesh_path),
        target_mesh_hash=_sha256_file(target_mesh_path),
        transform_type="nearest_neighbor_back_transfer_acceptance",
        mean_distance=float(np.mean(source_to_target_dist)) if source_count else 0.0,
        max_distance=float(np.max(source_to_target_dist)) if source_count else 0.0,
        collision_ratio=collision_ratio,
        seam_transfer_collapse=collision_ratio,
        retention_ratio=retention_ratio,
        unique_targets_used=unique_targets,
        total_target_vertices=target_count,
        edge_retention_ratio=retention_ratio,
        promotion=1 if promotes else 0,
        notes=notes,
        blocked_consumers=[] if promotes else [],
        source_topology_label=source_topology_label,
        target_topology_label=target_topology_label or target_body_receipt.topology_label,
        source_topology_hash=_sha256_file(source_mesh_path),
        target_body_receipt_hash=_sha256_file(target_body_receipt_path),
        forward_object_hash=_sha256_file(source_mesh_path),
        transfer_mode=transfer_mode,
        approximate_transfer=(transfer_mode == "approximate_correspondence"),
        round_trip_mean_distance=float(np.mean(round_trip_distances)) if source_count else 0.0,
        round_trip_max_distance=float(np.max(round_trip_distances)) if source_count else 0.0,
        round_trip_retention_ratio=round_trip_retention,
        load_metric=load_metric,
        output_label=f"{source_topology_label}_to_{target_topology_label or target_body_receipt.topology_label}",
    )

    output_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt.to_json(output_receipt_path)
    if output_map_path is not None:
        output_map_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_map_path,
            source_to_target_indices=source_to_target,
            source_to_target_distances=source_to_target_dist,
            target_to_source_indices=target_to_source,
            target_to_source_distances=target_to_source_dist,
            meta=np.array(
                {
                    "source_mesh": str(source_mesh_path),
                    "target_mesh": str(target_mesh_path),
                    "source_topology_label": source_topology_label,
                    "target_topology_label": target_topology_label
                    or target_body_receipt.topology_label,
                    "receipt": str(output_receipt_path),
                    "transfer_mode": transfer_mode,
                },
                dtype=object,
            ),
        )

    print(f"Wrote P3 back-transfer receipt to {output_receipt_path}")
    print(
        json.dumps(
            {
                "promotion": receipt.promotion,
                "transfer_mode": receipt.transfer_mode,
                "approximate_transfer": receipt.approximate_transfer,
                "collision_ratio": receipt.collision_ratio,
                "retention_ratio": receipt.retention_ratio,
                "round_trip_retention_ratio": receipt.round_trip_retention_ratio,
                "load_metric": receipt.load_metric,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return receipt


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mesh", type=Path, required=True)
    parser.add_argument("--target-mesh", type=Path, required=True)
    parser.add_argument("--target-body-receipt", type=Path, required=True)
    parser.add_argument("--out-receipt", type=Path, required=True)
    parser.add_argument("--out-map", type=Path, default=None)
    parser.add_argument("--source-topology-label", default="B_v9438")
    parser.add_argument("--target-topology-label", default=None)
    parser.add_argument(
        "--transfer-mode",
        choices=("true_inverse", "pseudo_inverse", "approximate_correspondence"),
        default="approximate_correspondence",
    )
    parser.add_argument("--max-distance-ratio-threshold", type=float, default=0.08)
    parser.add_argument("--collision-ratio-threshold", type=float, default=0.80)
    parser.add_argument("--seam-retention-threshold", type=float, default=0.20)
    parser.add_argument("--round-trip-retention-threshold", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    emit_p3_back_transfer_receipt(
        source_mesh_path=args.source_mesh,
        target_mesh_path=args.target_mesh,
        target_body_receipt_path=args.target_body_receipt,
        output_receipt_path=args.out_receipt,
        output_map_path=args.out_map,
        source_topology_label=args.source_topology_label,
        target_topology_label=args.target_topology_label,
        transfer_mode=args.transfer_mode,
        max_distance_ratio_threshold=args.max_distance_ratio_threshold,
        collision_ratio_threshold=args.collision_ratio_threshold,
        seam_retention_threshold=args.seam_retention_threshold,
        round_trip_retention_threshold=args.round_trip_retention_threshold,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
