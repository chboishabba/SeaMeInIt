"""Generate hard-shell articulation panels from a fitted body record."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from suit_hard import HardShellSegmentationOptions, HardShellSegmenter

OUTPUT_ROOT = Path("outputs/hard_shell")

__all__ = ["generate_hard_shell", "load_body_record", "main"]


def load_body_record(path: Path) -> dict[str, Any]:
    """Load a body record containing mesh and joint data."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, Mapping):
            raise TypeError("Body record JSON must be a JSON object.")
    elif suffix == ".npz":
        data = np.load(path)
        payload = {key: data[key] for key in data.files}
    else:
        raise ValueError(f"Unsupported body record format: {path.suffix}")
    return dict(payload)


def _extract_joint_payload(record: Mapping[str, Any]) -> tuple[np.ndarray, Sequence[str] | None]:
    if "joint_positions" in record:
        positions = np.asarray(record["joint_positions"], dtype=float)
    elif "joints" in record:
        positions = np.asarray(record["joints"], dtype=float)
    else:
        raise KeyError("Body record must include `joint_positions` or `joints` for segmentation.")

    names: Sequence[str] | None = None
    if "joint_names" in record:
        raw_names = record["joint_names"]
        if not isinstance(raw_names, Sequence):
            raise TypeError("joint_names must be a sequence of strings.")
        names = [str(name) for name in raw_names]

    return positions, names


def _ensure_output_dir(base_dir: Path | None, body_path: Path) -> Path:
    if base_dir is None:
        target = OUTPUT_ROOT / body_path.stem
    else:
        target = base_dir
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_panel_npz(path: Path, panel_payload: Mapping[str, Any]) -> None:
    np.savez(
        path,
        cut_point=np.asarray(panel_payload["cut_point"], dtype=float),
        cut_normal=np.asarray(panel_payload["cut_normal"], dtype=float),
        hinge_line=np.asarray(panel_payload["hinge_line"], dtype=float),
        boundary=np.asarray(panel_payload["boundary"], dtype=float),
        allowance=float(panel_payload["allowance"]),
        motion_axis=np.asarray(panel_payload["motion_axis"], dtype=float),
        limb_length=float(panel_payload["limb_length"]),
    )


def _write_manifest(path: Path, segmentation_payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    manifest = {"panels": segmentation_payload["panels"], "metadata": dict(metadata)}
    with path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)


def generate_hard_shell(
    body_path: Path,
    *,
    output_dir: Path | None = None,
    options: HardShellSegmentationOptions | None = None,
) -> None:
    """Generate hard-shell panels and persist artefacts."""

    record = load_body_record(body_path)
    segmenter = HardShellSegmenter()
    joint_positions, joint_names = _extract_joint_payload(record)
    opts = options or HardShellSegmentationOptions()
    segmentation = segmenter.segment(
        joint_positions,
        options=opts,
        joint_names=joint_names,
    )

    target_dir = _ensure_output_dir(output_dir, body_path)

    payload = segmentation.as_dict()
    for panel in payload["panels"]:
        panel_name = panel["name"]
        panel_path = target_dir / f"{panel_name}.npz"
        _write_panel_npz(panel_path, panel)

    metadata = {
        "hinge_allowance": opts.hinge_allowance,
        "body_record": str(body_path),
        "panel_count": len(payload["panels"]),
    }
    _write_manifest(target_dir / "manifest.json", payload, metadata)


def _parse_options(args: argparse.Namespace) -> HardShellSegmentationOptions:
    return HardShellSegmentationOptions(
        hinge_allowance=args.hinge_allowance,
        panel_width_scale=args.panel_width_scale,
        panel_height_scale=args.panel_height_scale,
        hinge_extension_scale=args.hinge_extension_scale,
        boundary_points=args.boundary_points,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate motion-aware hard-shell panels from a body record.")
    parser.add_argument("body", type=Path, help="Path to the fitted body record containing joints.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Directory where panel artefacts should be written (default: outputs/hard_shell/<body name>).",
    )
    parser.add_argument("--hinge-allowance", type=float, default=0.004, help="Clearance offset applied to hinge seams in metres.")
    parser.add_argument(
        "--panel-width-scale",
        type=float,
        default=0.3,
        help="Multiplier applied to limb length to derive lateral panel span.",
    )
    parser.add_argument(
        "--panel-height-scale",
        type=float,
        default=0.22,
        help="Multiplier applied to limb length to derive circumferential panel span.",
    )
    parser.add_argument(
        "--hinge-extension-scale",
        type=float,
        default=0.08,
        help="Fraction of limb length extending the hinge beyond the ellipse bounds.",
    )
    parser.add_argument(
        "--boundary-points",
        type=int,
        default=24,
        help="Number of samples used to approximate each elliptical cut boundary.",
    )

    args = parser.parse_args(argv)

    options = _parse_options(args)
    generate_hard_shell(args.body, output_dir=args.output, options=options)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI helper
    raise SystemExit(main())
