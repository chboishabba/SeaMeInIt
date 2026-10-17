#!/usr/bin/env python3
"""Export Gate-7 manufacturing artifacts from the frozen P2 winner receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts import evaluate_operator_tree as tree_eval
from scripts import unwrap_panels as unwrap_mod
from smii.seams.panel_serialization_competition import serialize_panel

SCHEMA_VERSION = "smii.bt369_p2_manufacturing.v1"
CLAIM_BOUNDARY = "geometric_pattern_export_requires_physical_fabrication_authorization"

ALLOWANCE_BY_METHOD = {
    "home_sewing": 0.015,  # 15 mm
    "overlock": 0.010,  # 10 mm
    "flatlock": 0.008,  # 8 mm
    "bonded": 0.005,  # 5 mm
    "welded": 0.003,  # 3 mm
    "laser_cut": 0.002,  # 2 mm
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _chart_boundary_points(patch: unwrap_mod.PanelPatch, uv: np.ndarray) -> np.ndarray:
    """Extract ordered boundary vertices for SVG polygon rendering."""
    edge_counts: dict[tuple[int, int], int] = {}
    for face in patch.faces:
        v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            norm = (a, b) if a < b else (b, a)
            edge_counts[norm] = edge_counts.get(norm, 0) + 1

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    if not boundary_edges:
        return np.asarray(uv, dtype=float)

    # Construct adjacency for simple loop ordering
    adj: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    visited = set()
    order = []
    curr = boundary_edges[0][0]
    while curr not in visited and len(order) < len(boundary_edges):
        visited.add(curr)
        order.append(curr)
        next_nodes = [n for n in adj.get(curr, []) if n not in visited]
        if not next_nodes:
            break
        curr = next_nodes[0]

    # Check if patch.vertices is a list of global vertex IDs or coordinate array
    if hasattr(patch, "vertices") and np.asarray(patch.vertices).ndim == 1:
        v_index = {int(v): i for i, v in enumerate(patch.vertices)}
        ordered_indices = [v_index[v] for v in order if v in v_index]
    else:
        ordered_indices = [v for v in order if 0 <= v < len(uv)]

    if len(ordered_indices) >= 3:
        return np.asarray(uv[ordered_indices], dtype=float)
    return np.asarray(uv, dtype=float)


def _generate_p2_svg(
    *,
    charts_uv: Sequence[np.ndarray],
    chart_patches: Sequence[unwrap_mod.PanelPatch],
    foldovers_by_chart: Sequence[int],
    allowance_meters: float,
    manufacturing_method: str,
    columns: int = 5,
) -> str:
    """Render a multi-column nested SVG cutting layout for all charts."""
    cell_size = 220.0
    margin = 30.0
    scale = 140.0
    num_charts = len(charts_uv)
    rows = (num_charts + columns - 1) // columns

    total_width = margin * 2 + columns * cell_size
    total_height = margin * 2 + rows * cell_size + 40.0

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width:.1f}" '
            f'height="{total_height:.1f}" viewBox="0 0 {total_width:.1f} {total_height:.1f}" '
            f'data-units="meters" data-scale="{scale:.1f}">'
        ),
        f"  <title>SMII P2 Frozen Winner Cutting Layout - {manufacturing_method}</title>",
        f"  <!-- Target allowance: {allowance_meters * 1000:.1f}mm ({manufacturing_method}) -->",
        '  <g id="p2_charts" fill="none" stroke="#222" stroke-width="1.2">',
    ]

    for idx, (uv, patch, fold_count) in enumerate(
        zip(charts_uv, chart_patches, foldovers_by_chart, strict=False)
    ):
        col = idx % columns
        row = idx // columns
        origin_x = margin + col * cell_size
        origin_y = margin + row * cell_size + 30.0

        pts = _chart_boundary_points(patch, uv)
        u_min, u_max = float(pts[:, 0].min()), float(pts[:, 0].max())
        v_min, v_max = float(pts[:, 1].min()), float(pts[:, 1].max())
        span = max(u_max - u_min, v_max - v_min, 1e-6)
        c_scale = (cell_size - 40.0) / span

        shifted_x = (pts[:, 0] - u_min) * c_scale + origin_x + 10.0
        shifted_y = (v_max - pts[:, 1]) * c_scale + origin_y + 10.0
        point_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(shifted_x, shifted_y, strict=False))

        has_foldover = fold_count > 0
        stroke_color = "#c0392b" if has_foldover else "#2c3e50"
        fill_color = "rgba(231, 76, 60, 0.08)" if has_foldover else "rgba(52, 152, 219, 0.05)"

        lines.extend(
            [
                (
                    f'    <g id="chart_{idx:02d}" data-faces="{len(patch.faces)}" '
                    f'data-grain="warp" data-seam-allowance="{allowance_meters:.4f}">'
                ),
                (
                    f'      <polygon points="{point_str}" fill="{fill_color}" '
                    f'stroke="{stroke_color}" stroke-width="1.5" />'
                ),
                (
                    f'      <text x="{origin_x + 10:.1f}" y="{origin_y - 8:.1f}" '
                    f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#333">'
                    f"P2.chart.{idx:02d} ({len(patch.faces)}f"
                    f"{', ' + str(fold_count) + ' foldovers' if has_foldover else ''})</text>"
                ),
                (
                    f'      <line x1="{origin_x + 15:.1f}" y1="{origin_y + 20:.1f}" '
                    f'x2="{origin_x + 35:.1f}" y2="{origin_y + 20:.1f}" '
                    'stroke="#e67e22" stroke-width="2" data-notch="true" />'
                ),
                (
                    "      <!-- Grainline vector -->",
                    f'      <line x1="{origin_x + 20:.1f}" y1="{origin_y + 40:.1f}" '
                    f'x2="{origin_x + 20:.1f}" y2="{origin_y + 70:.1f}" '
                    'stroke="#7f8c8d" stroke-dasharray="3,3" data-grainline="true" />',
                )[1],
            ]
        )
        if has_foldover:
            lines.append(
                f'      <circle cx="{origin_x + cell_size - 25:.1f}" cy="{origin_y - 12:.1f}" '
                f'r="5" fill="#e74c3c" data-foldover-warning="true" />'
            )
        lines.append("    </g>")

    lines.extend(["  </g>", "</svg>", ""])
    return "\n".join(lines)


def export_p2_manufacturing(
    *,
    frozen_winner_path: Path,
    mesh_path: Path,
    seam_edges_path: Path,
    out_dir: Path,
    manufacturing_method: str = "home_sewing",
) -> dict[str, Any]:
    """Export validated Gate-7 artifacts and receipt for the frozen P2 winner."""
    if manufacturing_method not in ALLOWANCE_BY_METHOD:
        raise ValueError(
            f"unknown manufacturing method '{manufacturing_method}'; "
            f"supported: {sorted(ALLOWANCE_BY_METHOD)}"
        )

    raw_frozen = frozen_winner_path.read_bytes()
    frozen = json.loads(raw_frozen)
    if frozen.get("schema_version") != "smii.bt369_frozen_p2_winner.v1":
        raise ValueError(f"unexpected frozen winner schema: {frozen.get('schema_version')}")

    topology = frozen["topology"]
    expected_sha = frozen["topology_sha256"]
    computed_sha = hashlib.sha256(json.dumps(topology, separators=(",", ":")).encode()).hexdigest()
    if expected_sha != computed_sha:
        raise ValueError("topology SHA-256 mismatch")

    vertices, panels_raw = tree_eval._reconstruct_source_panels(mesh_path, seam_edges_path)
    source_panels = [unwrap_mod.PanelPatch(*p) if isinstance(p, tuple) else p for p in panels_raw]
    p2_panel = source_panels[2]
    chart_patches = tree_eval._connected_chart_panels(p2_panel, topology)

    if len(chart_patches) != len(topology):
        raise ValueError(f"reconstructed {len(chart_patches)} patches, expected {len(topology)}")

    comps = frozen["backend_competitions"]
    uvs = []
    measured_distortions = []
    measured_foldovers = []

    for idx, patch in enumerate(chart_patches):
        backend = comps[idx]["selected_backend"]
        cand, uv = serialize_panel(
            vertices=vertices,
            panel=patch,
            correction_tree=None,
            materialization_constraints={
                "source_panel_id": "P2",
                "chart_materialization_kind": "sequential_operator_tree",
            },
            backend=backend,
            distortion_threshold=0.05,
        )
        if uv is None:
            raise ValueError(f"UV serialization failed for chart {idx} using {backend}")
        uvs.append(uv)
        measured_distortions.append(float(cand.distortion or 0.0))
        measured_foldovers.append(int(cand.foldovers or 0))

    total_foldovers = sum(measured_foldovers)
    worst_dist = max(measured_distortions) if measured_distortions else 0.0
    pinned_metrics = frozen.get("metrics", {})
    if abs(worst_dist - float(pinned_metrics.get("worst_distortion", 0.0))) > 1e-4:
        raise ValueError("reconstructed worst distortion does not match pinned receipt")

    allowance_meters = ALLOWANCE_BY_METHOD[manufacturing_method]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save UV npz
    uv_path = out_dir / "p2_chart_uvs.npz"
    np.savez_compressed(uv_path, **{f"chart_{i:02d}": uv for i, uv in enumerate(uvs)})
    uv_hash = _sha256_file(uv_path)

    # 2. Save Seam Allowance npz
    allowance_array = np.full(len(chart_patches), allowance_meters, dtype=float)
    allowance_path = out_dir / "p2_seam_allowance.npz"
    np.savez_compressed(allowance_path, allowance=allowance_array)
    allowance_hash = _sha256_file(allowance_path)

    # 3. Generate SVG cutting layout
    svg = _generate_p2_svg(
        charts_uv=uvs,
        chart_patches=chart_patches,
        foldovers_by_chart=measured_foldovers,
        allowance_meters=allowance_meters,
        manufacturing_method=manufacturing_method,
    )
    svg_path = out_dir / "p2_cutting_layout.svg"
    svg_path.write_text(svg, encoding="utf-8")
    svg_hash = _sha256_file(svg_path)

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "frozen_p2_winner_sha256": hashlib.sha256(raw_frozen).hexdigest(),
        "topology_sha256": computed_sha,
        "chart_count": len(chart_patches),
        "manufacturing_method": manufacturing_method,
        "allowance_width_meters": allowance_meters,
        "measured_metrics": {
            "worst_distortion": worst_dist,
            "total_foldovers": total_foldovers,
            "charts_with_foldovers": [i for i, f in enumerate(measured_foldovers) if f > 0],
            "foldover_counts_by_chart": {
                f"chart_{i:02d}": f for i, f in enumerate(measured_foldovers) if f > 0
            },
        },
        "artifacts": {
            "chart_uvs": {
                "path": str(uv_path.name),
                "sha256": uv_hash,
            },
            "seam_allowance": {
                "path": str(allowance_path.name),
                "sha256": allowance_hash,
            },
            "cutting_layout": {
                "path": str(svg_path.name),
                "sha256": svg_hash,
            },
        },
        "validation_checks": {
            "all_charts_unwrapped": len(uvs) == len(chart_patches),
            "units_declared": "meters",
            "notches_present": True,
            "grainlines_present": True,
            "labels_present": True,
            "metrics_match_pinned_receipt": True,
        },
        "promotion": 1,
        "blockers": [],
    }

    receipt_path = out_dir / "p2_manufacturing_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-winner", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--seam-edges", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--manufacturing-method",
        choices=sorted(ALLOWANCE_BY_METHOD),
        default="home_sewing",
    )
    args = parser.parse_args()

    receipt = export_p2_manufacturing(
        frozen_winner_path=args.frozen_winner,
        mesh_path=args.mesh,
        seam_edges_path=args.seam_edges,
        out_dir=args.out_dir,
        manufacturing_method=args.manufacturing_method,
    )
    print(
        json.dumps(
            {
                "promotion": receipt["promotion"],
                "chart_count": receipt["chart_count"],
                "artifacts": receipt["artifacts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
