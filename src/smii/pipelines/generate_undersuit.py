"""Generate undersuit layers from a fitted SMPL-X body record."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from modules.cooling import plan_cooling_layout
from suit import UnderSuitGenerator, UnderSuitOptions
from suit.thermal_zones import DEFAULT_THERMAL_ZONE_SPEC

OUTPUT_ROOT = Path("outputs/suits")
COOLING_OUTPUT_ROOT = Path("outputs/modules/cooling")

__all__ = ["generate_undersuit", "load_body_record", "main"]


def load_body_record(path: Path) -> dict[str, np.ndarray]:
    """Load a fitted body mesh from JSON or NPZ disk representations."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, Mapping):
            raise TypeError("Body record JSON must be an object with vertices/faces arrays.")
        vertices = np.asarray(payload.get("vertices"), dtype=float)
        faces = np.asarray(payload.get("faces"), dtype=int)
    elif suffix == ".npz":
        data = np.load(path)
        vertices = np.asarray(data["vertices"], dtype=float)
        faces = np.asarray(data["faces"], dtype=int)
    else:
        raise ValueError(f"Unsupported body record format: {path.suffix}")

    if vertices.ndim == 3:
        vertices = vertices[0]
    if faces.ndim == 3:
        faces = faces[0]

    return {"vertices": vertices, "faces": faces}


def _load_measurements(path: Path | None) -> Mapping[str, float] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("Measurement payload must be a JSON object.")
    return {key: float(value) for key, value in payload.items()}


def _ensure_output_dir(base_dir: Path | None, body_path: Path) -> Path:
    if base_dir is None:
        target = OUTPUT_ROOT / body_path.stem
    else:
        target = base_dir
    target.mkdir(parents=True, exist_ok=True)
    return target


def _save_layer(layer_path: Path, layer_vertices: np.ndarray, layer_faces: np.ndarray) -> None:
    np.savez(layer_path, vertices=layer_vertices, faces=layer_faces)


def _write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    serialisable = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in metadata.items()
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(serialisable, stream, indent=2)


def generate_undersuit(
    body_path: Path,
    *,
    output_dir: Path | None = None,
    measurements: Mapping[str, float] | None = None,
    options: UnderSuitOptions | None = None,
    embed_cooling: bool = False,
    cooling_medium: str = "liquid",
) -> None:
    """Run the undersuit generation pipeline and persist all artefacts."""

    record = load_body_record(body_path)
    generator = UnderSuitGenerator()
    result = generator.generate(record, options=options, measurements=measurements)

    target_dir = _ensure_output_dir(output_dir, body_path)

    layer_paths = {
        "base": target_dir / "base_layer.npz",
        "insulation": target_dir / "insulation_layer.npz",
        "comfort": target_dir / "comfort_layer.npz",
    }

    _save_layer(layer_paths["base"], result.base_layer.vertices, result.base_layer.faces)
    if result.insulation_layer is not None:
        _save_layer(
            layer_paths["insulation"],
            result.insulation_layer.vertices,
            result.insulation_layer.faces,
        )
    if result.comfort_layer is not None:
        _save_layer(
            layer_paths["comfort"],
            result.comfort_layer.vertices,
            result.comfort_layer.faces,
        )

    metadata = dict(result.metadata)
    metadata["output_directory"] = str(target_dir)
    metadata["body_record"] = str(body_path)
    metadata["layers"] = {
        layer.name: {
            "thickness": layer.thickness,
            "surface_area": layer.surface_area,
        }
        for layer in result.layers()
    }
    if embed_cooling:
        _embed_cooling_manifest(
            body_path=body_path,
            target_dir=target_dir,
            base_vertices=result.base_layer.vertices,
            medium=cooling_medium,
            metadata=metadata,
        )

    _write_metadata(target_dir / "metadata.json", metadata)


def _embed_cooling_manifest(
    *,
    body_path: Path,
    target_dir: Path,
    base_vertices: np.ndarray,
    medium: str,
    metadata: dict[str, Any],
) -> None:
    medium = medium.lower()
    if medium not in {"liquid", "pcm"}:
        raise ValueError("Cooling medium must be either 'liquid' or 'pcm'.")

    plan = plan_cooling_layout(
        DEFAULT_THERMAL_ZONE_SPEC,
        base_vertices,
        medium=medium,
    )
    manifest = plan.to_manifest()
    manifest["body_record"] = str(body_path)
    manifest["undersuit_output"] = str(target_dir)

    cooling_dir = COOLING_OUTPUT_ROOT / target_dir.name
    cooling_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cooling_dir / f"{body_path.stem}_{medium}.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    cooling_meta = dict(metadata.get("cooling", {}))
    cooling_meta.update(
        {
            "medium": medium,
            "manifest": str(manifest_path),
            "zones": manifest["spec"]["zones"],
        }
    )
    metadata["cooling"] = cooling_meta


def _parse_options_from_args(args: argparse.Namespace) -> UnderSuitOptions:
    weights = None
    if args.weight:
        weights = {}
        for pair in args.weight:
            key, value = pair.split("=", 1)
            weights[key] = float(value)
    return UnderSuitOptions(
        base_thickness=args.base_thickness,
        insulation_thickness=args.insulation_thickness,
        comfort_liner_thickness=args.comfort_thickness,
        include_insulation=not args.no_insulation,
        include_comfort_liner=not args.no_comfort,
        ease_percent=args.ease_percent,
        measurement_weights=weights,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate undersuit layers from a fitted body mesh."
    )
    parser.add_argument("body", type=Path, help="Path to the fitted body record (JSON or NPZ).")
    parser.add_argument(
        "--output",
        type=Path,
        help="Directory where undersuit artefacts should be written (default: outputs/suits/<body name>).",
    )
    parser.add_argument(
        "--measurements",
        type=Path,
        help="Optional measurement JSON used to influence sizing ease.",
    )
    parser.add_argument(
        "--base-thickness", type=float, default=0.0015, help="Base layer shell thickness in metres."
    )
    parser.add_argument(
        "--insulation-thickness",
        type=float,
        default=0.003,
        help="Insulation layer thickness in metres.",
    )
    parser.add_argument(
        "--comfort-thickness", type=float, default=0.001, help="Comfort liner thickness in metres."
    )
    parser.add_argument(
        "--ease-percent",
        type=float,
        default=0.03,
        help="Fractional ease applied uniformly to the body.",
    )
    parser.add_argument(
        "--no-insulation",
        action="store_true",
        help="Disable generation of the insulation mid-layer.",
    )
    parser.add_argument(
        "--no-comfort",
        action="store_true",
        help="Disable generation of the comfort liner layer.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        metavar="MEASUREMENT=WEIGHT",
        help="Optional weighting overrides for measurement-driven scaling.",
    )
    parser.add_argument(
        "--embed-cooling",
        action="store_true",
        help="Generate a cooling routing manifest alongside undersuit layers.",
    )
    parser.add_argument(
        "--cooling-medium",
        choices=("liquid", "pcm"),
        default="liquid",
        help="Cooling medium used when embedding circuits.",
    )
    args = parser.parse_args(argv)

    measurements = _load_measurements(args.measurements)
    options = _parse_options_from_args(args)

    generate_undersuit(
        args.body,
        output_dir=args.output,
        measurements=measurements,
        options=options,
        embed_cooling=args.embed_cooling,
        cooling_medium=args.cooling_medium,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
