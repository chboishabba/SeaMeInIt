#!/usr/bin/env python3
"""Generate final receipted manufacturing artifacts from promoted panel UVs."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
from typing import Iterable

import numpy as np

from smii.seams import (
    ManufacturingReceipt,
    can_consume_panel_unwrap_receipt,
    load_panel_unwrap_receipt,
)

ALLOWANCE_BY_METHOD = {
    "home_sewing": 0.015,
    "overlock": 0.010,
    "flatlock": 0.008,
    "bonded": 0.005,
    "welded": 0.003,
    "laser_cut": 0.002,
    "3d_print": 0.000,
    "eva_foam_cut": 0.005,
}
ACCESSIBILITY_BY_METHOD = {
    "home_sewing": "consumer",
    "overlock": "consumer",
    "flatlock": "advanced",
    "bonded": "advanced",
    "welded": "industrial",
    "laser_cut": "industrial",
    "3d_print": "advanced",
    "eva_foam_cut": "consumer",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _load_panel_uvs(path: Path, panel_count: int) -> list[np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    panels: list[np.ndarray] = []
    for idx in range(panel_count):
        key = f"panel_{idx}"
        if key not in payload:
            raise KeyError(f"Panel UV artifact is missing '{key}'.")
        uv = np.asarray(payload[key], dtype=float)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError(f"{key} must be shaped (N, 2).")
        panels.append(uv)
    return panels


def _load_rom_field(path: Path, key: str) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if key not in payload:
        raise KeyError(f"ROM fields artifact is missing '{key}'.")
    field = np.asarray(payload[key], dtype=float).reshape(-1)
    if field.size == 0:
        raise ValueError(f"ROM field '{key}' must be non-empty.")
    if not np.isfinite(field).all():
        raise ValueError(f"ROM field '{key}' must be finite.")
    return field


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    max_value = float(np.max(values)) if values.size else 0.0
    if max_value <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return values / max_value


def _compute_allowance_field(
    *,
    pressure_peak: np.ndarray,
    shear_peak: np.ndarray,
    base_allowance: float,
    variable_allowance: bool,
) -> np.ndarray:
    if pressure_peak.shape != shear_peak.shape:
        raise ValueError("pressure_peak and shear_peak must have the same shape.")
    if not variable_allowance:
        return np.full(pressure_peak.shape, float(base_allowance), dtype=float)
    pressure_gradient = np.abs(np.gradient(pressure_peak))
    shear_gradient = np.abs(np.gradient(shear_peak))
    field_norm = _normalise(pressure_gradient + shear_gradient)
    return float(base_allowance) * (1.0 + 0.5 * field_norm)


def _panel_points_for_svg(
    panel: np.ndarray,
    *,
    offset_x: float,
    offset_y: float,
    scale: float,
) -> str:
    if panel.size == 0:
        return ""
    shifted = np.asarray(panel, dtype=float).copy()
    shifted[:, 0] = (shifted[:, 0] - float(shifted[:, 0].min())) * scale + offset_x
    shifted[:, 1] = (float(shifted[:, 1].max()) - shifted[:, 1]) * scale + offset_y
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in shifted)


def _generate_cutting_layout_svg(
    *,
    panels: list[np.ndarray],
    grain_directions: list[str],
    allowance_mean: float,
    manufacturing_method: str,
) -> str:
    cell_width = 160.0
    margin = 20.0
    scale = 100.0
    width = max(cell_width, cell_width * max(1, len(panels)))
    height = 180.0
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.1f}" '
            f'height="{height:.1f}" viewBox="0 0 {width:.1f} {height:.1f}">'
        ),
        f'  <title>SMII cutting layout - {manufacturing_method}</title>',
        '  <g id="panels" fill="none" stroke="#222" stroke-width="1.2">',
    ]
    for idx, panel in enumerate(panels):
        offset_x = margin + idx * cell_width
        offset_y = margin + 40.0
        points = _panel_points_for_svg(
            panel,
            offset_x=offset_x,
            offset_y=offset_y,
            scale=scale,
        )
        direction = grain_directions[idx]
        lines.extend(
            [
                f'    <g id="panel_{idx}" data-grain="{direction}">',
                (
                    f'      <polyline points="{points}" data-seam-allowance='
                    f'"{allowance_mean:.6f}" />'
                ),
                (
                    f'      <text x="{offset_x:.1f}" y="{margin:.1f}" '
                    f'font-size="10">P{idx} {direction}</text>'
                ),
                (
                    f'      <line x1="{offset_x:.1f}" y1="{offset_y + 112:.1f}" '
                    f'x2="{offset_x + 30:.1f}" y2="{offset_y + 112:.1f}" '
                    'stroke="#a33" data-notch="true" />'
                ),
                "    </g>",
            ]
        )
    lines.extend(["  </g>", "</svg>", ""])
    return "\n".join(lines)


def generate_manufacturing_artifacts(
    *,
    panel_receipt_path: Path,
    panel_uvs_path: Path,
    rom_fields_path: Path,
    output_dir: Path,
    manufacturing_method: str = "home_sewing",
    variable_allowance: bool = True,
    receipt_path: Path | None = None,
) -> ManufacturingReceipt:
    """Generate manufacturing artifacts and emit the final receipt."""

    if manufacturing_method not in ALLOWANCE_BY_METHOD:
        raise ValueError(
            "manufacturing_method must be one of "
            f"{', '.join(sorted(ALLOWANCE_BY_METHOD))}."
        )
    panel_receipt = load_panel_unwrap_receipt(panel_receipt_path)
    if not can_consume_panel_unwrap_receipt(panel_receipt, "manufacturing"):
        raise ValueError(
            f"PanelUnwrapReceipt not promoted ({panel_receipt.promotion}). "
            f"Blocked: {panel_receipt.blocked_consumers}"
        )
    if panel_receipt.uv_hash != _sha256_file(panel_uvs_path):
        raise ValueError("Panel UV hash does not match PanelUnwrapReceipt.uv_hash.")

    panels = _load_panel_uvs(panel_uvs_path, panel_receipt.panel_count)
    pressure_peak = _load_rom_field(rom_fields_path, "pressure_peak")
    shear_peak = _load_rom_field(rom_fields_path, "shear_peak")
    base_allowance = ALLOWANCE_BY_METHOD[manufacturing_method]
    allowance_field = _compute_allowance_field(
        pressure_peak=pressure_peak,
        shear_peak=shear_peak,
        base_allowance=base_allowance,
        variable_allowance=variable_allowance,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    allowance_path = output_dir / "seam_allowance.npz"
    np.savez_compressed(allowance_path, allowance=allowance_field)
    allowance_hash = _sha256_file(allowance_path)

    allowance_mean = float(allowance_field.mean())
    svg = _generate_cutting_layout_svg(
        panels=panels,
        grain_directions=panel_receipt.grain_directions,
        allowance_mean=allowance_mean,
        manufacturing_method=manufacturing_method,
    )
    cutting_path = output_dir / "cutting_layout.svg"
    cutting_path.write_text(svg, encoding="utf-8")
    cutting_hash = _sha256_file(cutting_path)

    allowance_varies = bool(float(allowance_field.std()) > 1e-4)
    notches_present = 'data-notch="true"' in svg
    labels_present = "<text" in svg
    promotion = 1 if allowance_varies and notches_present and labels_present else 0
    notes = ""
    if not allowance_varies:
        notes = "allowance_varies=False: check ROM field coverage"

    receipt = ManufacturingReceipt(
        panel_unwrap_receipt_hash=_sha256_file(panel_receipt_path),
        panel_count=panel_receipt.panel_count,
        manufacturing_method=manufacturing_method,
        accessibility_level=ACCESSIBILITY_BY_METHOD[manufacturing_method],
        seam_allowance_hash=allowance_hash,
        seam_allowance_mean=allowance_mean,
        seam_allowance_min=float(allowance_field.min()),
        seam_allowance_max=float(allowance_field.max()),
        allowance_varies=allowance_varies,
        grain_directions=panel_receipt.grain_directions,
        panel_hashes=[_sha256_array(panel) for panel in panels],
        cutting_artifacts_hash=cutting_hash,
        notches_present=notches_present,
        labels_present=labels_present,
        promotion=promotion,
        blocked_consumers=[],
        notes=notes,
    )
    target_receipt_path = receipt_path or (output_dir / "manufacturing_receipt.json")
    receipt.to_json(target_receipt_path)
    print(f"Wrote seam allowance to {allowance_path}")
    print(f"Wrote cutting layout to {cutting_path}")
    print(f"Wrote manufacturing receipt to {target_receipt_path}")
    return receipt


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-receipt", type=Path, required=True)
    parser.add_argument("--panel-uvs", type=Path, required=True)
    parser.add_argument("--rom-fields", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--out-manufacturing-receipt",
        type=Path,
        default=None,
        help="Output manufacturing receipt path (default: <out-dir>/manufacturing_receipt.json).",
    )
    parser.add_argument(
        "--manufacturing-method",
        choices=sorted(ALLOWANCE_BY_METHOD),
        default="home_sewing",
    )
    parser.add_argument(
        "--constant-allowance",
        action="store_true",
        help="Emit a constant allowance field; receipt will remain diagnostic-only.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    generate_manufacturing_artifacts(
        panel_receipt_path=args.panel_receipt,
        panel_uvs_path=args.panel_uvs,
        rom_fields_path=args.rom_fields,
        output_dir=args.out_dir,
        manufacturing_method=args.manufacturing_method,
        variable_allowance=not args.constant_allowance,
        receipt_path=args.out_manufacturing_receipt,
    )


if __name__ == "__main__":
    main()
