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
from suit.seam_generator import SeamGenerator
from suit.thermal_zones import DEFAULT_THERMAL_ZONE_SPEC
from smii.meshing import load_body_record

OUTPUT_ROOT = Path("outputs/suits")
COOLING_OUTPUT_ROOT = Path("outputs/modules/cooling")

__all__ = ["generate_undersuit", "load_body_record", "main"]


def _load_measurements(path: Path | None) -> Mapping[str, float] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("Measurement payload must be a JSON object.")
    return {key: float(value) for key, value in payload.items()}


def _compose_measurement_map(
    record: Mapping[str, Any],
    overrides: Mapping[str, float] | None,
) -> dict[str, tuple[float, float]]:
    measurement_map: dict[str, tuple[float, float]] = {}

    report = record.get("measurement_report") if isinstance(record, Mapping) else None
    if isinstance(report, Mapping):
        values = report.get("values")
        if isinstance(values, Sequence):
            for entry in values:
                if not isinstance(entry, Mapping):
                    continue
                name = entry.get("name")
                value = entry.get("value")
                if name is None or value is None:
                    continue
                confidence = entry.get("confidence")
                source = entry.get("source")
                weight = 1.0 if source == "measured" else float(confidence) if confidence is not None else 1.0
                name_str = str(name)
                # Prefer measured entries when duplicates are encountered
                if name_str in measurement_map and measurement_map[name_str][1] >= 1.0 and weight < 1.0:
                    continue
                measurement_map[name_str] = (float(value), float(weight))

    if overrides:
        for name, value in overrides.items():
            measurement_map[str(name)] = (float(value), 1.0)

    return measurement_map


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


def _load_joint_map(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if isinstance(payload, Mapping):
        joint_payload = payload.get("joints", payload)
    else:
        raise TypeError("Joint map payload must be a JSON object.")

    if not isinstance(joint_payload, Mapping):
        raise TypeError("Joint map must provide a mapping of joint names to coordinates.")

    joints: dict[str, np.ndarray] = {}
    for name, coords in joint_payload.items():
        array = np.asarray(coords, dtype=float)
        if array.shape != (3,):
            raise ValueError(f"Joint '{name}' must be a length-3 coordinate array.")
        joints[str(name)] = array
    return joints or None


def _axis_vector(
    joint_map: Mapping[str, np.ndarray] | None,
    *,
    start_candidates: Sequence[str],
    end_candidates: Sequence[str],
) -> np.ndarray | None:
    if not joint_map:
        return None
    for start in start_candidates:
        if start not in joint_map:
            continue
        for end in end_candidates:
            if end not in joint_map:
                continue
            vector = joint_map[end] - joint_map[start]
            norm = np.linalg.norm(vector)
            if norm > 1e-6:
                return vector / norm
    return None


def _infer_axis_map(
    vertices: np.ndarray,
    joint_map: Mapping[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    longitudinal = _axis_vector(
        joint_map,
        start_candidates=("pelvis", "hips", "hip"),
        end_candidates=("neck", "spine2", "spine1", "withers", "head"),
    )
    lateral = _axis_vector(
        joint_map,
        start_candidates=(
            "left_hip",
            "left_knee",
            "left_shoulder",
            "left_front_shoulder",
        ),
        end_candidates=(
            "right_hip",
            "right_knee",
            "right_shoulder",
            "right_front_shoulder",
        ),
    )
    if longitudinal is None:
        # Use principal component along the greatest variance as a fallback
        centred = vertices - vertices.mean(axis=0)
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        longitudinal = vh[0]
    if lateral is None:
        if joint_map:
            lateral = _axis_vector(
                joint_map,
                start_candidates=("left_ankle", "left_foot", "left_front_paw"),
                end_candidates=("right_ankle", "right_foot", "right_front_paw"),
            )
        if lateral is None:
            centred = vertices - vertices.mean(axis=0)
            _, _, vh = np.linalg.svd(centred, full_matrices=False)
            lateral = vh[1]
    anterior = np.cross(longitudinal, lateral)
    if np.linalg.norm(anterior) < 1e-8:
        centred = vertices - vertices.mean(axis=0)
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        anterior = vh[2]
    # Re-orthonormalise the basis to ensure right-handedness
    longitudinal = longitudinal / np.linalg.norm(longitudinal)
    anterior = anterior / np.linalg.norm(anterior)
    lateral = np.cross(anterior, longitudinal)
    lateral = lateral / np.linalg.norm(lateral)
    anterior = np.cross(longitudinal, lateral)
    anterior = anterior / np.linalg.norm(anterior)
    return {
        "longitudinal": longitudinal,
        "lateral": lateral,
        "anterior": anterior,
    }


def generate_undersuit(
    body_path: Path,
    *,
    output_dir: Path | None = None,
    measurements: Mapping[str, float] | None = None,
    options: UnderSuitOptions | None = None,
    embed_cooling: bool = False,
    cooling_medium: str = "liquid",
    joint_map: Mapping[str, Sequence[float]] | None = None,
) -> None:
    """Run the undersuit generation pipeline and persist all artefacts."""

    record = load_body_record(body_path)
    measurement_pairs = _compose_measurement_map(record, measurements)
    measurement_values = {name: pair[0] for name, pair in measurement_pairs.items()}
    generator_measurements: Mapping[str, tuple[float, float]] | None = (
        measurement_pairs if measurement_pairs else None
    )

    generator = UnderSuitGenerator()
    result = generator.generate(record, options=options, measurements=generator_measurements)

    seam_generator = SeamGenerator()
    joint_lookup = (
        {name: np.asarray(value, dtype=float) for name, value in joint_map.items()}
        if joint_map
        else None
    )
    axis_map = _infer_axis_map(result.base_layer.vertices, joint_lookup)
    loops = seam_generator.derive_measurement_loops(
        result.base_layer.vertices,
        result.base_layer.faces,
        axis_map,
        measurements=measurement_values or None,
    )
    seam_graph = seam_generator.generate(
        result.base_layer.vertices,
        result.base_layer.faces,
        loops,
        axis_map,
        measurements=measurement_values or None,
    )

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

    pattern_dir = target_dir / "patterns"
    exporter = PatternExporter()
    mesh_payload = seam_graph.to_payload()
    exported = exporter.export(
        mesh_payload,
        seam_graph.seam_metadata,
        output_dir=pattern_dir,
        metadata={"body_record": str(body_path)},
    )
    metadata["patterns"] = {
        "panels": [panel.name for panel in seam_graph.panels],
        "files": {fmt: str(path) for fmt, path in exported.items()},
        "panel_count": len(seam_graph.panels),
        "measurement_loops": {
            loop.name: {
                "axis_coordinate": loop.axis_coordinate,
                "vertex_count": len(loop.vertex_indices),
            }
            for loop in seam_graph.measurement_loops
        },
        "seams": {
            name: dict(values) for name, values in seam_graph.seam_metadata.items()
        },
        "axis_map": {key: axis_map[key].tolist() for key in ("longitudinal", "lateral", "anterior")},
    }

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
        "--joint-map",
        type=Path,
        help="Optional JSON file describing joint coordinates used to orient seams.",
    )
    args = parser.parse_args(argv)

    measurements = _load_measurements(args.measurements)
    options = _parse_options_from_args(args)

    joint_map = _load_joint_map(args.joint_map)

    generate_undersuit(
        args.body,
        output_dir=args.output,
        measurements=measurements,
        options=options,
        embed_cooling=args.embed_cooling,
        cooling_medium=args.cooling_medium,
        joint_map=joint_map,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
