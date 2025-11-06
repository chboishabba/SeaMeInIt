"""Generate a rigid hard-shell from a fitted avatar mesh."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from suit_hard import ShellGenerator, ShellOptions

OUTPUT_ROOT = Path("outputs/hard_layer")

__all__ = ["generate_hard_shell", "load_body_record", "main"]


def load_body_record(path: Path) -> dict[str, np.ndarray]:
    """Load a fitted body record stored as JSON or NPZ."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, Mapping):
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
