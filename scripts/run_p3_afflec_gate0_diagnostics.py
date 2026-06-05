#!/usr/bin/env python3
"""Run and summarize P3 Afflec Gate 0 reference-quality diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

IMAGE_EXTENSIONS = {".avif", ".jpeg", ".jpg", ".png", ".webp"}
CURATED_EXCLUDES = {
    "1_PAY-EXCLUSIVE-Ben-Affleck-Defends-His-Massive-Back-Tattoo-After-Admitting-Sentiment-Ran-Against-It.avif",
    "Screenshot_20260604_135454.png",
    "images.jpg",
}
LANE_MODES = ("raw", "refined")


@dataclass(frozen=True, slots=True)
class LaneSpec:
    """One P3 Gate 0 diagnostic lane."""

    name: str
    reference_policy: str
    mode: str
    image_paths: tuple[Path, ...]
    output_dir: Path
    command: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_load(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected JSON object at {path}.")
    return payload


def _image_paths(reference_root: Path, *, curated: bool) -> tuple[Path, ...]:
    paths = tuple(
        sorted(
            path
            for path in reference_root.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and (not curated or path.name not in CURATED_EXCLUDES)
        )
    )
    if not paths:
        raise FileNotFoundError(f"No reference images found under {reference_root}.")
    return paths


def preflight_reference_images(reference_root: Path) -> list[dict[str, object]]:
    """Record image dimensions and whether MediaPipe can detect pose landmarks."""

    from PIL import Image
    from smii.pipelines.fit_from_images import _pose_landmarks_from_mediapipe

    rows: list[dict[str, object]] = []
    for path in _image_paths(reference_root, curated=False):
        row: dict[str, object] = {
            "path": str(path),
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "curated_excluded": path.name in CURATED_EXCLUDES,
        }
        try:
            with Image.open(path) as image:
                row["format"] = image.format
                row["width"] = int(image.size[0])
                row["height"] = int(image.size[1])
                row["mode"] = image.mode
        except Exception as exc:  # pragma: no cover - defensive corrupt-file path
            row["image_error"] = f"{type(exc).__name__}: {exc}"
        try:
            _pose_landmarks_from_mediapipe(path)
            row["mediapipe_pose"] = "detected"
        except Exception as exc:
            row["mediapipe_pose"] = "not_detected"
            row["mediapipe_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def build_lanes(
    *,
    reference_root: Path,
    output_root: Path,
    python: str,
) -> list[LaneSpec]:
    """Build the all-reference and curated diagnostic lanes."""

    lanes: list[LaneSpec] = []
    for reference_policy, curated in (("all_refs", False), ("curated_refs", True)):
        image_paths = _image_paths(reference_root, curated=curated)
        for mode in LANE_MODES:
            name = f"{reference_policy}_{mode}"
            output_dir = output_root / name
            command = [
                python,
                "-m",
                "smii.app",
                "afflec-demo",
                "--images",
                *(str(path) for path in image_paths),
                "--output",
                str(output_dir),
                "--detector",
                "mediapipe",
                "--fit-mode",
                "auto",
                "--require-high-trust-detector",
                "--clean-output",
            ]
            if mode == "raw":
                command.append("--skip-measurement-refinement")
            lanes.append(
                LaneSpec(
                    name=name,
                    reference_policy=reference_policy,
                    mode=mode,
                    image_paths=image_paths,
                    output_dir=output_dir,
                    command=tuple(command),
                )
            )
    return lanes


def _run_command(command: Sequence[str]) -> int:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not existing else f"src{os.pathsep}{existing}"
    result = subprocess.run(list(command), env=env, check=False)
    return int(result.returncode)


def _crown_eccentricity_residual(vertices: np.ndarray) -> float:
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] < 3:
        return 0.0
    y_values = points[:, 1]
    threshold = float(np.quantile(y_values, 0.95))
    crown = points[y_values >= threshold]
    if crown.shape[0] < 3:
        return 0.0
    xy = crown[:, [0, 2]]
    spread = np.ptp(xy, axis=0)
    major = float(np.max(spread))
    minor = float(np.min(spread))
    if major <= 1e-9:
        return 0.0
    return float((major - minor) / major)


def _mesh_summary(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with np.load(path) as payload:
        vertices = np.asarray(payload["vertices"], dtype=float)
        faces = np.asarray(payload["faces"], dtype=int)
    return {
        "path": str(path),
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "crown_eccentricity_residual": _crown_eccentricity_residual(vertices),
    }


def _beta_summary(values: object) -> dict[str, float] | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return {"l2_norm": 0.0, "max_abs": 0.0, "mean_abs": 0.0}
    return {
        "l2_norm": float(np.linalg.norm(array)),
        "max_abs": float(np.max(np.abs(array))),
        "mean_abs": float(np.mean(np.abs(array))),
    }


def _nested(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def summarize_lane(lane: LaneSpec, *, return_code: int | None = None) -> dict[str, object]:
    diagnostics = _json_load(lane.output_dir / "afflec_fit_diagnostics.json")
    raw_regression = _json_load(lane.output_dir / "afflec_raw_regression.json")
    measurement_fit = _json_load(lane.output_dir / "afflec_measurement_fit.json")
    params = _json_load(lane.output_dir / "afflec_smplx_params.json")
    receipt = _json_load(lane.output_dir / "body_carrier_receipt.json")

    mesh_paths = {
        "raw_reprojection": lane.output_dir / "afflec_body_raw_reprojection.npz",
        "refined_pre_repair": lane.output_dir / "afflec_body_refined_pre_repair.npz",
        "final_export": lane.output_dir / "afflec_body.npz",
    }

    return {
        "name": lane.name,
        "reference_policy": lane.reference_policy,
        "mode": lane.mode,
        "return_code": return_code,
        "output_dir": str(lane.output_dir),
        "image_count": len(lane.image_paths),
        "images": [str(path) for path in lane.image_paths],
        "command": list(lane.command),
        "diagnostics": {
            "status": _nested(diagnostics, "summary", "consistency_status"),
            "trust_level": _nested(diagnostics, "summary", "trust_level"),
            "flags": _nested(diagnostics, "summary", "consistency_flags") or [],
            "reference_quality": _nested(diagnostics, "summary", "reference_quality"),
            "raw_beta_summary": _nested(diagnostics, "raw_regression", "betas_summary"),
            "refined_beta_summary": _nested(diagnostics, "measurement_refinement", "betas_summary"),
        },
        "raw_regression_beta_summary": _beta_summary(
            raw_regression.get("betas") if raw_regression is not None else None
        ),
        "measurement_fit_beta_summary": _beta_summary(
            measurement_fit.get("betas") if measurement_fit is not None else None
        ),
        "parameter_beta_summary": _beta_summary(
            params.get("betas") if params is not None else None
        ),
        "body_receipt": dict(receipt) if receipt is not None else None,
        "meshes": {
            name: summary
            for name, path in mesh_paths.items()
            if (summary := _mesh_summary(path)) is not None
        },
    }


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "none"
    return str(value)


def _summary_table(lane_summaries: Iterable[Mapping[str, object]]) -> str:
    lines = [
        "| Lane | Images | Status | Flags | Receipt Promotion | Receipt Skull | Final Crown | Raw Crown | Refined Crown |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane in lane_summaries:
        diagnostics = lane.get("diagnostics")
        receipt = lane.get("body_receipt")
        meshes = lane.get("meshes")
        diagnostics_map = diagnostics if isinstance(diagnostics, Mapping) else {}
        receipt_map = receipt if isinstance(receipt, Mapping) else {}
        mesh_map = meshes if isinstance(meshes, Mapping) else {}
        final_mesh = mesh_map.get("final_export")
        raw_mesh = mesh_map.get("raw_reprojection")
        refined_mesh = mesh_map.get("refined_pre_repair")
        final_map = final_mesh if isinstance(final_mesh, Mapping) else {}
        raw_map = raw_mesh if isinstance(raw_mesh, Mapping) else {}
        refined_map = refined_mesh if isinstance(refined_mesh, Mapping) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(lane.get("name")),
                    str(lane.get("image_count")),
                    _fmt(diagnostics_map.get("status")),
                    _fmt(diagnostics_map.get("flags")),
                    _fmt(receipt_map.get("promotion")),
                    _fmt(receipt_map.get("skull_rigidity_residual")),
                    _fmt(final_map.get("crown_eccentricity_residual")),
                    _fmt(raw_map.get("crown_eccentricity_residual")),
                    _fmt(refined_map.get("crown_eccentricity_residual")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_reports(
    *,
    output_root: Path,
    reference_root: Path,
    lane_summaries: list[dict[str, object]],
    reference_preflight: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "p3_afflec_gate0_summary.json"
    markdown_path = output_root / "p3_afflec_gate0_summary.md"

    payload = {
        "generated_at": _utc_now(),
        "reference_root": str(reference_root),
        "policy": {
            "threshold_changes": "diagnostic_only_first",
            "threshold_tuning": "requires separate evidence-backed patch",
            "curated_excludes": sorted(CURATED_EXCLUDES),
        },
        "reference_preflight": reference_preflight or [],
        "lanes": lane_summaries,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    markdown = [
        "# P3 Afflec Gate 0 Diagnostic Evidence",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Reference root: `{reference_root}`",
        "",
        "Threshold policy: diagnostic-only first. Do not loosen the skull residual threshold from this run alone.",
        "",
        "## Lane Summary",
        "",
        _summary_table(lane_summaries),
        "",
        "## Curated Reference Policy",
        "",
        "The curated lane excludes:",
        "",
        *[f"- `{name}`" for name in sorted(CURATED_EXCLUDES)],
        "",
        "These files remain in the all-reference sensitivity lane.",
        "",
        "## Reference Preflight",
        "",
        "| File | Size | MediaPipe Pose | Curated Excluded |",
        "| --- | ---: | --- | --- |",
        *[
            "| "
            + " | ".join(
                [
                    f"`{row.get('file')}`",
                    f"{row.get('width', 'n/a')}x{row.get('height', 'n/a')}",
                    str(row.get("mediapipe_pose", "n/a")),
                    str(row.get("curated_excluded", False)),
                ]
            )
            + " |"
            for row in (reference_preflight or [])
        ],
        "",
        "## Next Decision Rule",
        "",
        "- If curated improves materially over all-refs, fix reference selection/provenance before thresholds.",
        "- If raw improves materially over refined, constrain measurement refinement before thresholds.",
        "- If raw/refined mesh checkpoints diverge from final export, localize mesh generation or repair/export.",
        "- Only tune thresholds in a separate patch if high-trust stable lanes are blocked only by a small skull-residual margin.",
        "",
    ]
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    return json_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("assets/reference_images/afflec"),
        help="Directory containing local Afflec reference images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/p3_afflec_gate0_20260604"),
        help="Output root for ignored diagnostic artifacts.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for Gate 0 subprocesses.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print lane commands without running Gate 0.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Summarize existing lane outputs without running Gate 0.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lanes = build_lanes(
        reference_root=args.reference_root,
        output_root=args.output_root,
        python=args.python,
    )

    if args.dry_run:
        for lane in lanes:
            print(f"[DRY-RUN] {lane.name}: {' '.join(lane.command)}")
        return 0

    reference_preflight = preflight_reference_images(args.reference_root)
    lane_summaries: list[dict[str, object]] = []
    for lane in lanes:
        print(f"[P3] lane={lane.name} images={len(lane.image_paths)} output={lane.output_dir}")
        return_code = 0 if args.summarize_only else _run_command(lane.command)
        lane_summaries.append(summarize_lane(lane, return_code=return_code))
        if return_code != 0:
            print(f"[P3] lane={lane.name} failed with return code {return_code}", file=sys.stderr)

    json_path, markdown_path = write_reports(
        output_root=args.output_root,
        reference_root=args.reference_root,
        lane_summaries=lane_summaries,
        reference_preflight=reference_preflight,
    )
    print(f"Wrote summary JSON to {json_path}")
    print(f"Wrote summary Markdown to {markdown_path}")
    promoted_or_diagnostic = [
        item for item in lane_summaries if str(item.get("reference_policy")) == "curated_refs"
    ]
    return 0 if all(item.get("return_code") == 0 for item in promoted_or_diagnostic) else 1


if __name__ == "__main__":
    raise SystemExit(main())
