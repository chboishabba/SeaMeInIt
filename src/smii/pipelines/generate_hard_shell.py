"""Generate rigid shells or articulation-panel guides from fitted body records."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from suit_hard import (
    HardShellSegmentationOptions,
    HardShellSegmenter,
    ShellGenerator,
    ShellOptions,
)

SHELL_OUTPUT_ROOT = Path("outputs/hard_layer")
SEGMENTATION_OUTPUT_ROOT = Path("outputs/hard_shell")

__all__ = [
    "generate_hard_shell",
    "generate_hard_shell_panels",
    "load_body_record",
    "main",
]


def load_body_record(path: Path) -> dict[str, Any]:
    """Load a fitted body record stored as JSON or NPZ."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("Body record JSON must be an object.")
        return dict(payload)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    raise ValueError(f"Unsupported body record format: {path.suffix}")


def _mesh_payload(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if "vertices" not in record or "faces" not in record:
        raise KeyError("Body record must include vertices and faces for shell generation.")
    vertices = np.asarray(record["vertices"], dtype=float)
    faces = np.asarray(record["faces"], dtype=int)
    if vertices.ndim == 3:
        vertices = vertices[0]
    if faces.ndim == 3:
        faces = faces[0]
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Body vertices must be shaped (N, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Body faces must be shaped (M, 3).")
    if not np.isfinite(vertices).all():
        raise ValueError("Body vertices must be finite.")
    return {"vertices": vertices, "faces": faces}


def _extract_joint_payload(
    record: Mapping[str, Any],
) -> tuple[np.ndarray, Sequence[str] | None]:
    if "joint_positions" in record:
        positions = np.asarray(record["joint_positions"], dtype=float)
    elif "joints" in record:
        positions = np.asarray(record["joints"], dtype=float)
    else:
        raise KeyError(
            "Body record must include joint_positions or joints for articulation segmentation."
        )

    names: Sequence[str] | None = None
    if "joint_names" in record:
        raw_names = record["joint_names"]
        if isinstance(raw_names, np.ndarray):
            raw_names = raw_names.tolist()
        if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
            raise TypeError("joint_names must be a sequence of strings.")
        names = tuple(str(name) for name in raw_names)
    return positions, names


def _load_region_masks(path: Path | None, vertex_count: int) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Region mask file must map names to vertex indices.")

    masks: dict[str, np.ndarray] = {}
    for name, indices in payload.items():
        entries = (
            indices
            if isinstance(indices, Sequence) and not isinstance(indices, (str, bytes))
            else (indices,)
        )
        mask = np.zeros(vertex_count, dtype=bool)
        for entry in entries:
            index = int(entry)
            if index < 0 or index >= vertex_count:
                raise ValueError(f"Region {name!r} index {index} is outside the mesh range.")
            mask[index] = True
        masks[str(name)] = mask
    return masks


def _load_vertex_thickness(path: Path | None, vertex_count: int) -> np.ndarray | None:
    if path is None:
        return None
    suffix = path.suffix.lower()
    if suffix == ".npy":
        profile = np.load(path)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "thickness" in archive:
                profile = archive["thickness"]
            elif archive.files:
                profile = archive[archive.files[0]]
            else:
                raise ValueError("Thickness archive contains no arrays.")
    elif suffix == ".json":
        profile = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=float)
    else:
        raise ValueError("Unsupported thickness profile format. Use .npy, .npz, or .json.")

    result = np.asarray(profile, dtype=float)
    if result.shape != (vertex_count,):
        raise ValueError("Vertex thickness array must match the number of vertices.")
    if not np.isfinite(result).all():
        raise ValueError("Vertex thickness values must be finite.")
    return result


def _output_dir(base_dir: Path | None, body_path: Path, root: Path) -> Path:
    target = Path(base_dir) if base_dir is not None else root / body_path.stem
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    serialisable = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in metadata.items()
    }
    path.write_text(json.dumps(serialisable, indent=2) + "\n", encoding="utf-8")


def generate_hard_shell(
    body_path: Path,
    *,
    output_dir: Path | None = None,
    default_thickness: float = 0.004,
    region_thickness: Mapping[str, float] | None = None,
    vertex_thickness_path: Path | None = None,
    region_masks_path: Path | None = None,
    exclusions: Sequence[str] | None = None,
    allow_non_watertight: bool = False,
) -> Path:
    """Inflate a fitted body mesh and return the generated shell NPZ path."""

    body_path = Path(body_path)
    record = _mesh_payload(load_body_record(body_path))
    vertices = record["vertices"]
    region_masks = _load_region_masks(region_masks_path, vertices.shape[0])
    vertex_thickness = _load_vertex_thickness(vertex_thickness_path, vertices.shape[0])
    if vertex_thickness is not None and region_thickness:
        raise ValueError(
            "Specify either vertex thickness data or region thickness values, not both."
        )

    if vertex_thickness is not None:
        thickness_profile: Any = vertex_thickness
        profile_source = "vertex"
    elif region_thickness:
        thickness_profile = {name: float(value) for name, value in region_thickness.items()}
        profile_source = "region"
    else:
        thickness_profile = float(default_thickness)
        profile_source = "uniform"

    options = ShellOptions(
        default_thickness=float(default_thickness),
        region_masks=region_masks,
        enforce_watertight=not allow_non_watertight,
    )
    result = ShellGenerator().generate(
        record,
        thickness_profile=thickness_profile,
        exclusions=list(exclusions) if exclusions else None,
        options=options,
    )

    target_dir = _output_dir(output_dir, body_path, SHELL_OUTPUT_ROOT)
    shell_path = target_dir / "shell_layer.npz"
    np.savez(
        shell_path,
        vertices=result.vertices,
        faces=result.faces,
        thickness=result.thickness,
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "output_directory": str(target_dir),
            "body_record": str(body_path),
            "thickness_profile_source": profile_source,
            "exclusions": list(exclusions) if exclusions else [],
        }
    )
    _write_metadata(target_dir / "metadata.json", metadata)
    return shell_path


def generate_hard_shell_panels(
    body_path: Path,
    *,
    output_dir: Path | None = None,
    options: HardShellSegmentationOptions | None = None,
) -> Path:
    """Generate motion-aware articulation panel guides and return the manifest path."""

    body_path = Path(body_path)
    record = load_body_record(body_path)
    positions, names = _extract_joint_payload(record)
    settings = options or HardShellSegmentationOptions()
    segmentation = HardShellSegmenter().segment(
        positions,
        options=settings,
        joint_names=names,
    )
    target_dir = _output_dir(output_dir, body_path, SEGMENTATION_OUTPUT_ROOT)
    payload = segmentation.as_dict()
    panels = payload["panels"]
    if not isinstance(panels, list):  # pragma: no cover - defensive
        raise TypeError("Segmentation payload must contain a panel list.")
    for panel in panels:
        if not isinstance(panel, Mapping):  # pragma: no cover - defensive
            raise TypeError("Each segmentation panel must be an object.")
        np.savez(
            target_dir / f"{panel['name']}.npz",
            cut_point=np.asarray(panel["cut_point"], dtype=float),
            cut_normal=np.asarray(panel["cut_normal"], dtype=float),
            hinge_line=np.asarray(panel["hinge_line"], dtype=float),
            boundary=np.asarray(panel["boundary"], dtype=float),
            allowance=float(panel["allowance"]),
            motion_axis=np.asarray(panel["motion_axis"], dtype=float),
            limb_length=float(panel["limb_length"]),
        )
    manifest = {
        "panels": panels,
        "metadata": {
            "body_record": str(body_path),
            "panel_count": len(panels),
            "hinge_allowance": settings.hinge_allowance,
        },
    }
    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _parse_region_thickness(values: Sequence[str] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    for entry in values or ():
        if "=" not in entry:
            raise ValueError("Region thickness values must be formatted as NAME=VALUE.")
        name, value = entry.split("=", 1)
        result[name] = float(value)
    return result


def _segmentation_options(args: argparse.Namespace) -> HardShellSegmentationOptions:
    return HardShellSegmentationOptions(
        hinge_allowance=args.hinge_allowance,
        panel_width_scale=args.panel_width_scale,
        panel_height_scale=args.panel_height_scale,
        hinge_extension_scale=args.hinge_extension_scale,
        boundary_points=args.boundary_points,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("body", type=Path, help="Fitted body record in JSON or NPZ form.")
    parser.add_argument("--mode", choices=("auto", "shell", "segments"), default="auto")
    parser.add_argument("--output", type=Path, help="Output directory override.")

    parser.add_argument("--default-thickness", type=float, default=0.004)
    parser.add_argument("--region-thickness", action="append", metavar="NAME=VALUE")
    parser.add_argument("--vertex-thickness", type=Path)
    parser.add_argument("--region-masks", type=Path)
    parser.add_argument("--exclude", action="append")
    parser.add_argument("--allow-non-watertight", action="store_true")

    parser.add_argument("--hinge-allowance", type=float, default=0.004)
    parser.add_argument("--panel-width-scale", type=float, default=0.3)
    parser.add_argument("--panel-height-scale", type=float, default=0.22)
    parser.add_argument("--hinge-extension-scale", type=float, default=0.08)
    parser.add_argument("--boundary-points", type=int, default=24)
    args = parser.parse_args(argv)

    mode = args.mode
    if mode == "auto":
        record = load_body_record(args.body)
        mode = "shell" if "vertices" in record and "faces" in record else "segments"

    if mode == "segments":
        generate_hard_shell_panels(
            args.body,
            output_dir=args.output,
            options=_segmentation_options(args),
        )
    else:
        generate_hard_shell(
            args.body,
            output_dir=args.output,
            default_thickness=args.default_thickness,
            region_thickness=_parse_region_thickness(args.region_thickness),
            vertex_thickness_path=args.vertex_thickness,
            region_masks_path=args.region_masks,
            exclusions=args.exclude,
            allow_non_watertight=args.allow_non_watertight,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI helper
    raise SystemExit(main())
