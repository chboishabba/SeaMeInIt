"""Generate undersuit layers from a fitted SMPL-X body record."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from exporters.patterns import PatternExporter
from modules.cooling import plan_cooling_layout
from suit import (
    Panel,
    PanelStatus,
    SurfacePatch,
    SuitMaterial,
    UnderSuitGenerator,
    UnderSuitOptions,
    combine_results,
    gate_panel_validation,
    panel_budgets_for,
    panel_to_payload,
    validate_panel_budgets,
    validate_panel_curvature,
)
from suit.panel_adapter import PanelPayloadSource
from suit.panel_payload import PanelPayload
from suit.seam_generator import SeamGenerator
from suit.seam_metadata import normalize_seam_metadata
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
                weight = (
                    1.0
                    if source == "measured"
                    else float(confidence)
                    if confidence is not None
                    else 1.0
                )
                name_str = str(name)
                # Prefer measured entries when duplicates are encountered
                if (
                    name_str in measurement_map
                    and measurement_map[name_str][1] >= 1.0
                    and weight < 1.0
                ):
                    continue
                measurement_map[name_str] = (float(value), float(weight))

    if overrides:
        for name, value in overrides.items():
            measurement_map[str(name)] = (float(value), 1.0)

    return measurement_map


def _resolve_body_path(path: Path) -> Path:
    """Return a concrete body record path, inferring file extensions when omitted."""

    path = Path(os.path.expandvars(str(path))).expanduser()
    if path.exists():
        return path
    if path.suffix:
        raise FileNotFoundError(f"Body record '{path}' does not exist.")

    parent = path.parent if path.parent != Path("") else Path(".")
    stem = path.name
    candidates = [parent / f"{stem}{ext}" for ext in (".npz", ".json")]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    missing = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Body record '{path}' not found. Provide a JSON or NPZ file (checked: {missing})."
    )


def _load_panel_definitions(path: Path | None) -> list[Mapping[str, Any]] | None:
    """Load optional panel definition overrides from JSON."""

    if path is None:
        return None
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, Mapping):
        panels = payload.get("panels")
        if panels is None:
            raise TypeError("Panel definition payload must include a 'panels' array.")
    else:
        panels = payload
    if not isinstance(panels, Sequence):
        raise TypeError("Panel definitions must be provided as a list.")

    result: list[Mapping[str, Any]] = []
    for entry in panels:
        if not isinstance(entry, Mapping):
            raise TypeError("Each panel definition must be a JSON object.")
        if "name" not in entry:
            raise KeyError("Panel definitions must include a 'name' field.")
        result.append(entry)
    return result or None


def _load_seam_overrides(path: Path | None) -> Mapping[str, Mapping[str, Any]] | None:
    """Load optional seam metadata overrides from JSON."""

    if path is None:
        return None
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("Seam override payload must be a JSON object.")

    overrides: dict[str, Mapping[str, Any]] = {}
    for name, value in payload.items():
        if not isinstance(value, Mapping):
            raise TypeError(f"Seam override for '{name}' must be a JSON object.")
        overrides[str(name)] = dict(value)
    return overrides


def _build_panel_payload(
    base_vertices: np.ndarray,
    definitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Construct a panel payload JSON structure from vertex indices or overrides."""

    if not definitions:
        return {"panels": []}

    vertices_arr = np.asarray(base_vertices, dtype=float)
    if vertices_arr.ndim != 2 or vertices_arr.shape[1] != 3:
        raise ValueError("Base layer vertices must be shaped (N, 3).")

    panels: list[PanelPayload] = []
    for definition in definitions:
        name = str(definition["name"])
        explicit_vertices = definition.get("vertices")
        if explicit_vertices is not None:
            panel_vertices = np.asarray(explicit_vertices, dtype=float)
        else:
            indices = definition.get("vertex_indices")
            if indices is None:
                raise KeyError(f"Panel '{name}' missing 'vertex_indices' or 'vertices'.")
            index_array = np.asarray(indices, dtype=int)
            if index_array.ndim != 1:
                raise ValueError(f"Panel '{name}' vertex_indices must be 1D.")
            if np.any(index_array < 0) or np.any(index_array >= vertices_arr.shape[0]):
                raise IndexError(f"Panel '{name}' vertex_indices reference out-of-range vertices.")
            panel_vertices = vertices_arr[index_array]

        faces = definition.get("faces")
        if faces is not None:
            panel_faces = [
                (int(face[0]), int(face[1]), int(face[2]))
                for face in faces  # type: ignore[index]
            ]
        else:
            count = panel_vertices.shape[0]
            if count < 3:
                raise ValueError(f"Panel '{name}' must include at least three vertices.")
            panel_faces = [(0, i, i + 1) for i in range(1, count - 1)]

        metadata = dict(definition.get("metadata", {}) or {})
        vertices = tuple(tuple(map(float, vertex)) for vertex in panel_vertices.tolist())
        faces = tuple(panel_faces)
        panels.append(
            PanelPayload(
                name=name,
                vertices=vertices,
                faces=faces,
                metadata=metadata,
            )
        )
    return {"panels": [panel.to_mapping() for panel in panels]}


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
    material: SuitMaterial = SuitMaterial.NEOPRENE,
    embed_cooling: bool = False,
    cooling_medium: str = "liquid",
    joint_map: Mapping[str, Sequence[float]] | None = None,
    panels_json: Path | None = None,
    seams_json: Path | None = None,
    pattern_backend: str = "simple",
    auto_split: bool = False,
    pdf_page_size: str = "a4",
) -> None:
    """Run the undersuit generation pipeline and persist all artefacts."""

    body_path = _resolve_body_path(body_path)
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
    budgets = panel_budgets_for(material)
    exporter = PatternExporter(budgets=budgets, pdf_page_size=pdf_page_size)
    panel_payloads: list[PanelPayload] = []
    panel_validation: dict[str, dict[str, object]] = {}
    for seam_panel in seam_graph.panels:
        vertices = tuple(tuple(map(float, vertex)) for vertex in seam_panel.vertices.tolist())
        faces = tuple(tuple(int(idx) for idx in face) for face in seam_panel.faces.tolist())
        source = PanelPayloadSource(vertices=vertices, faces=faces)
        surface_patch = SurfacePatch(
            vertex_indices=tuple(range(len(vertices))),
            face_indices=tuple(range(len(faces))),
            metadata={"panel_curvature": seam_panel.metadata.get("panel_curvature")},
        )
        panel = Panel(
            panel_id=seam_panel.name,
            surface_patch=surface_patch,
            budgets=budgets,
        )
        validation = combine_results(
            validate_panel_budgets(panel),
            validate_panel_curvature(panel),
        )
        if not validation.ok:
            panel = Panel(
                panel_id=panel.panel_id,
                surface_patch=panel.surface_patch,
                boundary_3d=panel.boundary_3d,
                boundary_2d=panel.boundary_2d,
                seams=panel.seams,
                grain=panel.grain,
                budgets=panel.budgets,
                status=PanelStatus(
                    sewable=False,
                    reason="; ".join(issue.code for issue in validation.issues),
                ),
            )
        payload = panel_to_payload(panel, source, include_surface_metadata=False)
        panel_payloads.append(payload)
        panel_validation[seam_panel.name] = {
            "ok": validation.ok,
            "issues": [issue.code for issue in validation.issues],
        }

    mesh_payload = {"panels": [panel.to_mapping() for panel in panel_payloads]}
    seam_metadata = normalize_seam_metadata(seam_graph.seam_metadata) or {}
    exported = exporter.export(
        mesh_payload,
        seam_metadata,
        output_dir=pattern_dir,
        metadata={"body_record": str(body_path)},
        auto_split=auto_split,
    )
    regularization_warnings = {}
    regularization_issues = {}
    regularization_summaries = {}
    if exporter.last_metadata:
        regularization_warnings = exporter.last_metadata.get("panel_warnings", {})
        regularization_issues = exporter.last_metadata.get("panel_issues", {})
        regularization_summaries = exporter.last_metadata.get("panel_issue_summaries", {})
    for panel_name, warnings in regularization_warnings.items():
        entry = panel_validation.setdefault(panel_name, {"ok": True, "issues": []})
        entry["regularization_issues"] = list(warnings)
    for panel_name, issues in regularization_issues.items():
        entry = panel_validation.setdefault(panel_name, {"ok": True, "issues": []})
        entry["regularization_issue_details"] = list(issues)
    for panel_name, summary in regularization_summaries.items():
        entry = panel_validation.setdefault(panel_name, {"ok": True, "issues": []})
        entry["regularization_status"] = dict(summary)
    for panel_name, entry in panel_validation.items():
        gate = gate_panel_validation(
            budget_issue_codes=entry.get("issues", []),
            regularization_issues=entry.get("regularization_issue_details"),
            material=material,
        )
        entry["gate"] = gate.to_mapping()
    metadata["patterns"] = {
        "panels": [panel.name for panel in seam_graph.panels],
        "files": {fmt: str(path) for fmt, path in exported.items()},
        "panel_count": len(seam_graph.panels),
        "material": material.value,
        "panel_validation": panel_validation,
        "measurement_loops": {
            loop.name: {
                "axis_coordinate": loop.axis_coordinate,
                "vertex_count": len(loop.vertex_indices),
            }
            for loop in seam_graph.measurement_loops
        },
        "seams": {name: dict(values) for name, values in seam_metadata.items()},
        "axis_map": {
            key: axis_map[key].tolist() for key in ("longitudinal", "lateral", "anterior")
        },
    }
    if auto_split and exporter.last_metadata:
        auto_split_meta = exporter.last_metadata.get("auto_split")
        if auto_split_meta:
            metadata["patterns"]["auto_split"] = dict(auto_split_meta)
        panel_count = exporter.last_metadata.get("panel_count")
        if panel_count:
            metadata["patterns"]["panel_count"] = int(panel_count)

    panel_definitions = _load_panel_definitions(panels_json)
    if panel_definitions:
        seams = normalize_seam_metadata(_load_seam_overrides(seams_json))
        mesh_payload = _build_panel_payload(result.base_layer.vertices, panel_definitions)
        exporter = PatternExporter(
            backend=pattern_backend,
            budgets=budgets,
            pdf_page_size=pdf_page_size,
        )
        pattern_dir = target_dir / "patterns"
        exported = exporter.export(
            mesh_payload,
            seams,
            output_dir=pattern_dir,
            metadata={"body_record": str(body_path)},
            auto_split=auto_split,
        )
        pattern_metadata: dict[str, Any] = {
            "panel_source": str(panels_json),
            "panels": [panel["name"] for panel in mesh_payload["panels"]],
            "files": {fmt: str(path) for fmt, path in exported.items()},
            "panel_count": len(mesh_payload["panels"]),
            "backend": pattern_backend,
        }
        if seams_json is not None:
            pattern_metadata["seam_source"] = str(seams_json)
        if seams is not None:
            pattern_metadata["seams"] = seams
        if auto_split and exporter.last_metadata:
            auto_split_meta = exporter.last_metadata.get("auto_split")
            if auto_split_meta:
                pattern_metadata["auto_split"] = dict(auto_split_meta)
            panel_count = exporter.last_metadata.get("panel_count")
            if panel_count:
                pattern_metadata["panel_count"] = int(panel_count)
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
        "--joint-map",
        type=Path,
        help="Optional JSON file describing joint coordinates used to orient seams.",
    )
    parser.add_argument(
        "--panels-json",
        type=Path,
        help="Optional panel payload JSON used to override automatically generated panels.",
    )
    parser.add_argument(
        "--seams-json",
        type=Path,
        help="Optional seam metadata JSON paired with --panels-json overrides.",
    )
    parser.add_argument(
        "--pattern-backend",
        choices=("simple", "lscm"),
        default="simple",
        help="Flattening backend used when exporting undersuit patterns.",
    )
    parser.add_argument(
        "--pdf-page-size",
        choices=("a4", "letter", "a0"),
        default="a4",
        help="PDF page size used for tiling (default: a4).",
    )
    parser.add_argument(
        "--auto-split",
        action="store_true",
        help="Automatically split panels when SUGGEST_SPLIT is emitted.",
    )
    parser.add_argument(
        "--material",
        choices=tuple(material.value for material in SuitMaterial),
        default=SuitMaterial.NEOPRENE.value,
        help="Material profile used to select panel budgets.",
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
        material=SuitMaterial(args.material),
        embed_cooling=args.embed_cooling,
        cooling_medium=args.cooling_medium,
        joint_map=joint_map,
        panels_json=args.panels_json,
        seams_json=args.seams_json,
        pattern_backend=args.pattern_backend,
        auto_split=args.auto_split,
        pdf_page_size=args.pdf_page_size,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
