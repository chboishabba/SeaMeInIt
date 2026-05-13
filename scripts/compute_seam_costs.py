#!/usr/bin/env python3
"""Compute receipted seam edge costs from promoted ROM fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from smii.meshing import (
    can_consume_correspondence_receipt,
    can_consume_receipt,
    load_body_carrier_receipt,
    load_correspondence_receipt,
)
from smii.rom import (
    SeamCostField,
    can_consume_rom_field_receipt,
    load_rom_field_receipt,
    save_seam_cost_field,
)
from smii.seams import SeamCostReceipt

DEFAULT_WEIGHTS = {
    "w_P": 1.0,
    "w_S": 0.8,
    "w_T": 0.6,
    "w_H": 0.4,
    "w_C": 0.3,
    "w_kappa": 0.2,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if "vertices" not in payload or "faces" not in payload:
        raise KeyError("Mesh NPZ must contain 'vertices' and 'faces' arrays.")
    vertices = np.asarray(payload["vertices"], dtype=float)
    faces = np.asarray(payload["faces"], dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Mesh vertices must be shaped (N, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Mesh faces must be shaped (M, 3).")
    return vertices, faces


def _mesh_edges(faces: np.ndarray) -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = set()
    for a, b, c in np.asarray(faces, dtype=int):
        for u, v in ((a, b), (b, c), (c, a)):
            lo, hi = sorted((int(u), int(v)))
            if lo != hi:
                edges.add((lo, hi))
    return tuple(sorted(edges))


def _field(payload: Mapping[str, np.ndarray], name: str, vertex_count: int) -> np.ndarray:
    key = f"{name}_peak"
    if key not in payload:
        return np.zeros(vertex_count, dtype=float)
    arr = np.asarray(payload[key], dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"ROM field '{key}' must be one-dimensional.")
    if len(arr) != vertex_count:
        raise ValueError(
            f"ROM field '{key}' vertex count ({len(arr)}) does not match mesh ({vertex_count})."
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"ROM field '{key}' contains non-finite values.")
    return arr


def _load_weights(path: Path | None) -> dict[str, float]:
    if path is None:
        return dict(DEFAULT_WEIGHTS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Weights JSON must contain an object.")
    weights = dict(DEFAULT_WEIGHTS)
    for key, value in payload.items():
        if key not in DEFAULT_WEIGHTS:
            raise KeyError(f"Unsupported seam-cost weight: {key}")
        weights[key] = float(value)
    return weights


def _edge_lengths(vertices: np.ndarray, edges: tuple[tuple[int, int], ...]) -> np.ndarray:
    if not edges:
        return np.zeros(0, dtype=float)
    edge_arr = np.asarray(edges, dtype=int)
    lengths = np.linalg.norm(vertices[edge_arr[:, 0]] - vertices[edge_arr[:, 1]], axis=1)
    maximum = float(np.max(lengths)) if lengths.size else 0.0
    if maximum <= 1e-12:
        return np.zeros_like(lengths)
    return lengths / maximum


def _cost_uniformity(costs: np.ndarray) -> float:
    finite = np.asarray(costs, dtype=float)
    if finite.size == 0:
        return 1.0
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    spread = float(np.std(finite))
    magnitude = float(np.max(np.abs(finite)))
    if magnitude <= 1e-12:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - (spread / (magnitude + 1e-8)))))


def compute_seam_costs(
    *,
    body_receipt_path: Path,
    rom_field_receipt_path: Path,
    rom_fields_path: Path,
    mesh_path: Path,
    costs_path: Path,
    receipt_path: Path,
    correspondence_receipt_path: Path | None = None,
    solve_domain: str = "A_v3240",
    weights: Mapping[str, float] | None = None,
    coverage_threshold: float = 0.99,
    uniformity_threshold: float = 0.95,
) -> SeamCostReceipt:
    """Compute edge costs and emit their promotion receipt."""

    body_receipt = load_body_carrier_receipt(body_receipt_path)
    rom_receipt = load_rom_field_receipt(rom_field_receipt_path)

    if not can_consume_receipt(body_receipt, "seam_cost_field"):
        raise ValueError(
            "BodyCarrierReceipt not promoted for seam costs "
            f"(status={body_receipt.promotion})."
        )
    if not can_consume_rom_field_receipt(rom_receipt, "seam_cost_field"):
        raise ValueError(
            "ROMFieldReceipt not promoted for seam costs "
            f"(status={rom_receipt.promotion})."
        )
    if rom_receipt.fields_hash != _sha256_file(rom_fields_path):
        raise ValueError("ROM fields hash does not match ROMFieldReceipt.fields_hash.")

    correspondence_hash: str | None = None
    if solve_domain == "B_v9438":
        if correspondence_receipt_path is None:
            raise ValueError("B_v9438 solve requires --correspondence-receipt.")
        corr = load_correspondence_receipt(correspondence_receipt_path)
        if not can_consume_correspondence_receipt(corr, "seam_cost_field"):
            raise ValueError(
                f"CorrespondenceReceipt not promoted ({corr.promotion}). "
                "Use solve_domain=A_v3240 or fix transfer first."
            )
        correspondence_hash = _sha256_file(correspondence_receipt_path)
    elif solve_domain != "A_v3240":
        raise ValueError("solve_domain must be A_v3240 or B_v9438.")

    vertices, faces = _load_mesh(mesh_path)
    vertex_count = int(vertices.shape[0])
    if vertex_count != int(body_receipt.vertex_count):
        raise ValueError(
            "Mesh vertex count mismatch with BodyCarrierReceipt: "
            f"mesh={vertex_count}, receipt={body_receipt.vertex_count}."
        )
    if vertex_count != int(rom_receipt.vertex_count):
        raise ValueError(
            "Mesh vertex count mismatch with ROMFieldReceipt: "
            f"mesh={vertex_count}, receipt={rom_receipt.vertex_count}."
        )

    fields = np.load(rom_fields_path)
    pressure = _field(fields, "pressure", vertex_count)
    shear = _field(fields, "shear", vertex_count)
    tension = _field(fields, "tension", vertex_count)
    thermal = _field(fields, "thermal", vertex_count)
    cooling = _field(fields, "cooling", vertex_count)

    edge_list = _mesh_edges(faces)
    if not edge_list:
        raise ValueError("Mesh produced no edges for seam-cost computation.")
    edge_arr = np.asarray(edge_list, dtype=int)
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    kappa = _edge_lengths(vertices, edge_list)

    i = edge_arr[:, 0]
    j = edge_arr[:, 1]
    edge_costs = (
        w["w_P"] * np.maximum(pressure[i], pressure[j])
        + w["w_S"] * np.maximum(shear[i], shear[j])
        + w["w_T"] * np.abs(tension[i] - tension[j])
        + w["w_H"] * np.abs(thermal[i] - thermal[j])
        + w["w_C"] * np.maximum(cooling[i], cooling[j])
        + w["w_kappa"] * kappa
    )
    vertex_costs = (
        w["w_P"] * pressure
        + w["w_S"] * shear
        + w["w_C"] * cooling
        + w["w_kappa"] * np.zeros(vertex_count, dtype=float)
    )
    finite_cost_coverage = float(np.isfinite(edge_costs).mean())
    cost_uniformity = _cost_uniformity(edge_costs)
    finite_edge_costs = edge_costs[np.isfinite(edge_costs)]
    peak_cost = float(np.max(finite_edge_costs)) if finite_edge_costs.size else 0.0
    mean_cost = float(np.mean(finite_edge_costs)) if finite_edge_costs.size else 0.0

    cost_field = SeamCostField(
        field="combined",
        vertex_costs=vertex_costs,
        edge_costs=edge_costs,
        edges=edge_list,
        samples_used=int(rom_receipt.pose_count),
        metadata={
            **{key: float(value) for key, value in w.items()},
            "finite_cost_coverage": finite_cost_coverage,
            "cost_uniformity": cost_uniformity,
        },
    )
    save_seam_cost_field(cost_field, costs_path)

    promotes = (
        finite_cost_coverage > coverage_threshold
        and cost_uniformity < uniformity_threshold
    )
    receipt = SeamCostReceipt(
        rom_field_receipt_hash=_sha256_file(rom_field_receipt_path),
        body_receipt_hash=_sha256_file(body_receipt_path),
        correspondence_receipt_hash=correspondence_hash,
        solve_domain=solve_domain,
        vertex_count=vertex_count,
        edge_count=len(edge_list),
        finite_cost_coverage=finite_cost_coverage,
        cost_uniformity=cost_uniformity,
        peak_cost=peak_cost,
        mean_cost=mean_cost,
        weight_vector={key: float(value) for key, value in w.items()},
        costs_hash=_sha256_file(costs_path),
        promotion=1 if promotes else 0,
        blocked_consumers=[] if promotes else [],
    )
    receipt.to_json(receipt_path)
    print(f"Wrote seam costs to {costs_path}")
    print(f"Wrote seam cost receipt to {receipt_path}")
    return receipt


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-receipt", type=Path, required=True)
    parser.add_argument("--rom-field-receipt", type=Path, required=True)
    parser.add_argument("--rom-fields", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out-costs", type=Path, default=Path("outputs/rom/seam_costs.npz"))
    parser.add_argument(
        "--out-seam-cost-receipt",
        type=Path,
        default=Path("outputs/rom/seam_cost_receipt.json"),
    )
    parser.add_argument("--correspondence-receipt", type=Path, default=None)
    parser.add_argument(
        "--solve-domain",
        choices=("A_v3240", "B_v9438"),
        default="A_v3240",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional JSON object overriding seam-cost weights.",
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.99)
    parser.add_argument("--uniformity-threshold", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    compute_seam_costs(
        body_receipt_path=args.body_receipt,
        rom_field_receipt_path=args.rom_field_receipt,
        rom_fields_path=args.rom_fields,
        mesh_path=args.mesh,
        costs_path=args.out_costs,
        receipt_path=args.out_seam_cost_receipt,
        correspondence_receipt_path=args.correspondence_receipt,
        solve_domain=args.solve_domain,
        weights=_load_weights(args.weights),
        coverage_threshold=args.coverage_threshold,
        uniformity_threshold=args.uniformity_threshold,
    )


if __name__ == "__main__":
    main()
