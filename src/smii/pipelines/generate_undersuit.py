"""Generate undersuit layers from a fitted SMPL-X body record."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from exporters.patterns import PatternExporter
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


def _load_panel_definitions(path: Path | None) -> tuple[dict[str, Any], ...]:
    if path is None:
        return ()
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    panels_iter: list[Any]
    if isinstance(payload, Mapping):
        raw_panels = payload.get("panels", payload)
        if isinstance(raw_panels, Mapping):
            panels_iter = [
                {"name": str(name), **(dict(definition) if isinstance(definition, Mapping) else {})}
                for name, definition in raw_panels.items()
            ]
        elif isinstance(raw_panels, Sequence) and not isinstance(raw_panels, (str, bytes)):
            panels_iter = [panel for panel in raw_panels]
        else:
            raise TypeError("Panel definition file must provide a 'panels' array or mapping.")
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        panels_iter = [panel for panel in payload]
    else:
        raise TypeError("Panel definition file must contain an object or array.")

    result: list[dict[str, Any]] = []
    for index, panel in enumerate(panels_iter):
        if not isinstance(panel, Mapping):
            raise TypeError("Each panel definition must be a JSON object.")
        panel_dict = dict(panel)
        name = str(panel_dict.get("name", f"panel_{index}"))
        indices = panel_dict.get("indices") or panel_dict.get("vertex_indices")
        if indices is None:
            raise KeyError(f"Panel '{name}' must define an 'indices' array referencing base vertices.")
        if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
            raise TypeError(f"Panel '{name}' indices must be an array of integers.")
        panel_dict["name"] = name
        result.append(panel_dict)
    return tuple(result)


def _load_seam_overrides(path: Path | None) -> dict[str, dict[str, Any]] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if not isinstance(payload, Mapping):
        raise TypeError("Seam override payload must be a JSON object.")

    seams_obj = payload.get("panels", payload)
    if not isinstance(seams_obj, Mapping):
        raise TypeError("Seam overrides must map panel names to configuration objects.")

    seams: dict[str, dict[str, Any]] = {}
    for name, overrides in seams_obj.items():
        if isinstance(overrides, Mapping):
            seams[str(name)] = {key: value for key, value in overrides.items()}
        else:
            raise TypeError("Each seam override entry must be an object with configuration values.")
    return seams


def _fan_triangulation(vertex_count: int) -> list[tuple[int, int, int]]:
    if vertex_count < 3:
        return []
    return [(0, i, i + 1) for i in range(1, vertex_count - 1)]


def _build_panel_payload(
    base_vertices: np.ndarray, definitions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    panels: list[dict[str, Any]] = []
    for index, definition in enumerate(definitions):
        name = str(definition.get("name", f"panel_{index}"))
        raw_indices = definition.get("indices") or definition.get("vertex_indices")
        if raw_indices is None:
            raise KeyError(f"Panel '{name}' must define an 'indices' array.")
        if not isinstance(raw_indices, Sequence) or isinstance(raw_indices, (str, bytes)):
            raise TypeError(f"Panel '{name}' indices must be a sequence of integers.")
        indices = [int(idx) for idx in raw_indices]
        try:
            panel_vertices = base_vertices[indices]
        except IndexError as exc:  # pragma: no cover - defensive guard
            raise IndexError(f"Panel '{name}' references an out-of-range vertex index.") from exc

        vertices = [list(map(float, vertex)) for vertex in panel_vertices]
        faces_payload = definition.get("faces") or definition.get("triangles")
        if faces_payload is None:
            faces = _fan_triangulation(len(indices))
        elif not isinstance(faces_payload, Sequence) or isinstance(faces_payload, (str, bytes)):
            raise TypeError(f"Panel '{name}' faces must be an array of index triples.")
        else:
            faces = [tuple(int(i) for i in face) for face in faces_payload]

        panels.append(
            {
                "name": name,
                "vertices": vertices,
                "faces": [list(face) for face in faces],
            }
        )
    return {"panels": panels}


def generate_undersuit(
    body_path: Path,
    *,
    output_dir: Path | None = None,
    measurements: Mapping[str, float] | None = None,
    options: UnderSuitOptions | None = None,
    embed_cooling: bool = False,
    cooling_medium: str = "liquid",
    panels_json: Path | None = None,
    seams_json: Path | None = None,
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

    panel_definitions = _load_panel_definitions(panels_json)
    if panel_definitions:
        seams = _load_seam_overrides(seams_json)
        mesh_payload = _build_panel_payload(result.base_layer.vertices, panel_definitions)
        exporter = PatternExporter()
        pattern_dir = target_dir / "patterns"
        exported = exporter.export(
            mesh_payload,
            seams,
            output_dir=pattern_dir,
            metadata={"body_record": str(body_path)},
        )
        pattern_metadata: dict[str, Any] = {
            "panel_source": str(panels_json),
            "panels": [panel["name"] for panel in mesh_payload["panels"]],
            "files": {fmt: str(path) for fmt, path in exported.items()},
            "panel_count": len(mesh_payload["panels"]),
        }
        if seams_json is not None:
            pattern_metadata["seam_source"] = str(seams_json)
        if seams is not None:
            pattern_metadata["seams"] = seams
        metadata["patterns"] = pattern_metadata

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
    parser.add_argument(
        "--panels-json",
        type=Path,
        help="Optional JSON file describing undersuit panel vertex indices.",
    )
    parser.add_argument(
        "--seams-json",
        type=Path,
        help="Optional JSON file mapping panel names to seam metadata overrides.",
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
        panels_json=args.panels_json,
        seams_json=args.seams_json,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
