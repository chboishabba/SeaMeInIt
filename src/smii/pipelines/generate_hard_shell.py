"""Generate hard-shell articulation panels from a fitted body record."""
"""Generate a rigid hard-shell from a fitted avatar mesh."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from suit_hard import HardShellSegmentationOptions, HardShellSegmenter

OUTPUT_ROOT = Path("outputs/hard_shell")
from suit_hard import ShellGenerator, ShellOptions

OUTPUT_ROOT = Path("outputs/hard_layer")

__all__ = ["generate_hard_shell", "load_body_record", "main"]


def load_body_record(path: Path) -> dict[str, Any]:
    """Load a body record containing mesh and joint data."""
def load_body_record(path: Path) -> dict[str, np.ndarray]:
    """Load a fitted body record stored as JSON or NPZ."""

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
            raise TypeError("Body record JSON must be an object containing vertices/faces arrays.")
        vertices = np.asarray(payload.get("vertices"), dtype=float)
        faces = np.asarray(payload.get("faces"), dtype=int)
    elif suffix == ".npz":
        archive = np.load(path)
        vertices = np.asarray(archive["vertices"], dtype=float)
        faces = np.asarray(archive["faces"], dtype=int)
    else:
        raise ValueError(f"Unsupported body record format: {path.suffix}")

    if vertices.ndim == 3:
        vertices = vertices[0]
    if faces.ndim == 3:
        faces = faces[0]

    return {"vertices": vertices, "faces": faces}


def _load_region_masks(path: Path | None, vertex_count: int) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("Region mask file must be a JSON object mapping names to vertex indices.")

    masks: dict[str, np.ndarray] = {}
    for name, indices in payload.items():
        mask = np.zeros(vertex_count, dtype=bool)
        if isinstance(indices, Sequence) and not isinstance(indices, (str, bytes)):
            for entry in indices:
                idx = int(entry)
                if idx < 0 or idx >= vertex_count:
                    raise ValueError(f"Region '{name}' index {idx} is outside the mesh range.")
                mask[idx] = True
        else:
            idx = int(indices)
            if idx < 0 or idx >= vertex_count:
                raise ValueError(f"Region '{name}' index {idx} is outside the mesh range.")
            mask[idx] = True
        masks[str(name)] = mask
    return masks


def _load_vertex_thickness(path: Path | None, vertex_count: int) -> np.ndarray | None:
    if path is None:
        return None
    suffix = path.suffix.lower()
    if suffix == ".npy":
        profile = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        if "thickness" in archive:
            profile = archive["thickness"]
        else:
            keys = list(archive.keys())
            if not keys:
                raise ValueError("Thickness archive contains no arrays.")
            profile = archive[keys[0]]
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        profile = np.asarray(payload, dtype=float)
    else:
        raise ValueError("Unsupported thickness profile format. Use .npy, .npz, or .json.")

    profile = np.asarray(profile, dtype=float)
    if profile.shape != (vertex_count,):
        raise ValueError("Vertex thickness array must match the number of vertices.")
    return profile


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
def _write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    serialisable = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in metadata.items()
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(serialisable, stream, indent=2)


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
    default_thickness: float = 0.004,
    region_thickness: Mapping[str, float] | None = None,
    vertex_thickness_path: Path | None = None,
    region_masks_path: Path | None = None,
    exclusions: Sequence[str] | None = None,
    allow_non_watertight: bool = False,
) -> None:
    """Run the hard-shell generation pipeline."""

    record = load_body_record(body_path)
    vertices = record["vertices"]
    region_masks = _load_region_masks(region_masks_path, vertices.shape[0])
    vertex_thickness = _load_vertex_thickness(vertex_thickness_path, vertices.shape[0])

    if vertex_thickness is not None and region_thickness:
        raise ValueError(
            "Specify either vertex thickness data or region thickness values, not both."
        )

    options = ShellOptions(
        default_thickness=default_thickness,
        region_masks=region_masks,
        enforce_watertight=not allow_non_watertight,
    )

    if vertex_thickness is not None:
        thickness_profile: Any = vertex_thickness
    elif region_thickness:
        thickness_profile = {name: float(value) for name, value in region_thickness.items()}
    else:
        thickness_profile = default_thickness

    generator = ShellGenerator()
    result = generator.generate(
        record,
        thickness_profile=thickness_profile,
        exclusions=list(exclusions) if exclusions else None,
        options=options,
    )

    target_dir = _ensure_output_dir(output_dir, body_path)
    shell_path = target_dir / "shell_layer.npz"
    np.savez(shell_path, vertices=result.vertices, faces=result.faces, thickness=result.thickness)

    metadata = dict(result.metadata)
    if vertex_thickness is not None:
        profile_source = "vertex"
    elif region_thickness:
        profile_source = "region"
    else:
        profile_source = "uniform"

    metadata.update(
        {
            "output_directory": str(target_dir),
            "body_record": str(body_path),
            "thickness_profile_source": profile_source,
            "exclusions": list(exclusions) if exclusions else [],
        }
    )
    _write_metadata(target_dir / "metadata.json", metadata)


def _parse_region_thickness(values: Sequence[str] | None) -> dict[str, float]:
    if not values:
        return {}
    result: dict[str, float] = {}
    for entry in values:
        if "=" not in entry:
            raise ValueError("Region thickness values must be formatted as NAME=VALUE.")
        name, value = entry.split("=", 1)
        result[name] = float(value)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a rigid shell over a fitted avatar.")
    parser.add_argument("body", type=Path, help="Path to the fitted body record (JSON or NPZ).")
    parser.add_argument(
        "--output",
        type=Path,
        help="Directory for shell artefacts (default: outputs/hard_layer/<body>).",
    )
    parser.add_argument(
        "--default-thickness", type=float, default=0.004, help="Baseline shell thickness in metres."
    )
    parser.add_argument(
        "--region-thickness",
        action="append",
        metavar="NAME=VALUE",
        help="Override thickness for a named region mask.",
    )
    parser.add_argument(
        "--vertex-thickness",
        type=Path,
        help="Path to a per-vertex thickness array (.npy, .npz, or JSON list).",
    )
    parser.add_argument(
        "--region-masks",
        type=Path,
        help="JSON file mapping region names to arrays of vertex indices.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        help="Region name to exclude from shell inflation (may be supplied multiple times).",
    )
    parser.add_argument(
        "--allow-non-watertight",
        action="store_true",
        help="Skip watertight validation (not recommended).",
    )
    args = parser.parse_args(argv)

    region_thickness = _parse_region_thickness(args.region_thickness)

    generate_hard_shell(
        args.body,
        output_dir=args.output,
        default_thickness=args.default_thickness,
        region_thickness=region_thickness,
        vertex_thickness_path=args.vertex_thickness,
        region_masks_path=args.region_masks,
        exclusions=args.exclude,
        allow_non_watertight=args.allow_non_watertight,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
