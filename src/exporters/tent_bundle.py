"""Utilities for exporting combined suit and inflatable tent bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from modules.tent import DeploymentKinematics

from .patterns import PatternExporter
from .unity_unreal_export import ExportFormat, SMPLXTemplate, UnityUnrealExporter

__all__ = ["export_suit_tent_bundle"]


def export_suit_tent_bundle(
    *,
    suit_template: SMPLXTemplate,
    canopy_mesh: Mapping[str, Any],
    seam_plan: Mapping[str, Mapping[str, Any]],
    deployment: DeploymentKinematics,
    unity_exporter: UnityUnrealExporter,
    pattern_exporter: PatternExporter,
    output_dir: Path | str,
    instruction_formats: Sequence[str] = ("pdf",),
) -> Mapping[str, Any]:
    """Export a bundle containing suit GLB and tent deployment instructions."""

    deployment.validate()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    glb_path = unity_exporter.export(
        suit_template,
        out_dir / "suit_with_tent.glb",
        ExportFormat.GLB,
    )

    instructions_dir = out_dir / "instructions"
    instruction_paths = pattern_exporter.export(
        canopy_mesh,
        seam_plan,
        output_dir=instructions_dir,
        formats=instruction_formats,
        metadata={
            "deployment_sequence": deployment.sequence.path_names(),
            "anchor_count": len(deployment.anchors),
            "fold_path_count": len(deployment.fold_paths),
        },
    )

    metadata = {
        "anchors": {
            name: {
                "landmark": anchor.landmark,
                "position": list(anchor.position),
                "normal": list(anchor.normal),
                "description": anchor.description,
            }
            for name, anchor in deployment.anchors.items()
        },
        "fold_paths": {
            name: {
                "anchors": list(path.anchors),
                "points": [list(point) for point in path.points],
                "description": path.description,
                "order": path.order,
            }
            for name, path in deployment.fold_paths.items()
        },
        "sequence": [
            {
                "name": step.name,
                "path": step.path_name,
                "instruction": step.instruction,
                "dwell_time": step.dwell_time,
            }
            for step in deployment.sequence.steps
        ],
        "glb": str(glb_path.relative_to(out_dir)),
        "instructions": {
            fmt: str(path.relative_to(out_dir)) for fmt, path in instruction_paths.items()
        },
    }

    metadata_path = out_dir / "bundle.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "bundle_dir": out_dir,
        "glb_path": glb_path,
        "instructions": instruction_paths,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }
