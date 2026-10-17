#!/usr/bin/env python3
"""CLI script to generate a full-suit garment pattern and seam graph from 3D body geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from smii.garment.full_suit_partition import (
    FullSuitPartitionConfig,
    FullSuitPartitioner,
    save_full_suit_pattern,
)
from smii.manufacturing.nesting import NestingConfig, nest_charts_deterministic
from smii.manufacturing.svg_export import export_full_suit_svg
from smii.manufacturing.validation import validate_manufacturing_layout
from smii.meshing.body_carrier_receipt import BodyCarrierReceipt


def generate_full_suit_patterns(
    *,
    mesh_path: Path,
    out_dir: Path,
    body_receipt_path: Path | None = None,
    allowance_mm: float = 10.0,
) -> dict[str, Any]:
    """Execute full-suit partitioning and emit validated pattern artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found at: {mesh_path}")

    mesh_data = np.load(mesh_path)
    if "vertices" not in mesh_data or "faces" not in mesh_data:
        raise ValueError(f"Mesh NPZ must contain 'vertices' and 'faces', found {mesh_data.files}")

    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]

    # Optional upstream body carrier receipt validation
    body_receipt = None
    if body_receipt_path and body_receipt_path.exists():
        body_receipt = BodyCarrierReceipt.from_json(body_receipt_path)
        if body_receipt.promotion == 0:
            raise ValueError(
                f"Body carrier receipt at {body_receipt_path} is rejected (promotion=0)."
            )

    cfg = FullSuitPartitionConfig(allowance_mm=allowance_mm)
    partitioner = FullSuitPartitioner(cfg)
    pattern, diagnostics = partitioner.partition(vertices, faces)

    # Nest charts onto fabric roll
    nesting_cfg = NestingConfig(allowance_mm=allowance_mm)
    nesting_res = nest_charts_deterministic(pattern.charts, nesting_cfg)

    # Render vector SVG cutting pattern layout
    svg_content = export_full_suit_svg(pattern, nesting_res)
    svg_out = out_dir / "cutting_layout.svg"
    svg_out.write_text(svg_content, encoding="utf-8")

    # Validate manufacturing layout
    validation_report = validate_manufacturing_layout(pattern, nesting_res)
    mfg_receipt_out = out_dir / "manufacturing_receipt.json"
    mfg_receipt_out.write_text(
        json.dumps(validation_report.to_dict(), indent=2),
        encoding="utf-8",
    )

    # Save artifacts
    npz_out = out_dir / "full_suit_charts.npz"
    manifest_out = out_dir / "pattern_manifest.json"
    seam_out = out_dir / "seam_topology.json"
    diag_out = out_dir / "partition_diagnostics.json"

    save_full_suit_pattern(pattern, npz_out)

    manifest_data = pattern.to_dict()
    if body_receipt:
        manifest_data["body_carrier_source_hash"] = body_receipt.source_hash

    manifest_out.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    seam_out.write_text(
        json.dumps([s.to_dict() for s in pattern.seams], indent=2),
        encoding="utf-8",
    )
    diag_out.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    return {
        "status": "success",
        "mesh_sha256": pattern.source_mesh_sha256,
        "region_count": len(pattern.regions),
        "chart_count": len(pattern.charts),
        "seam_count": len(pattern.seams),
        "fabric_width_mm": nesting_res.fabric_width_mm,
        "fabric_length_mm": nesting_res.fabric_length_mm,
        "fabric_utilization": round(nesting_res.utilization, 4),
        "gate7_promoted": validation_report.gate7_promoted,
        "svg_layout_path": str(svg_out),
        "output_directory": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate full-suit 2D garment patterns and seam topology from a 3D body mesh."
    )
    parser.add_argument("--mesh", type=Path, required=True, help="Input body mesh NPZ")
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Output directory for pattern artifacts"
    )
    parser.add_argument(
        "--body-receipt", type=Path, default=None, help="Optional body_carrier_receipt.json"
    )
    parser.add_argument(
        "--allowance-mm", type=float, default=10.0, help="Seam allowance width in mm"
    )

    args = parser.parse_args()
    res = generate_full_suit_patterns(
        mesh_path=args.mesh,
        out_dir=args.out_dir,
        body_receipt_path=args.body_receipt,
        allowance_mm=args.allowance_mm,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
