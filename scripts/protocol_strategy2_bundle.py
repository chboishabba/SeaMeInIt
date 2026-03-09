#!/usr/bin/env python3
"""Strategy 2 bundle runner: render ROM native AND reprojected artifacts.

Outputs a timestamped, gitignored bundle under outputs/assets_bundles/.

This is intentionally a thin orchestrator that:
- creates a bundle
- audits topology/provenance (vertex/face divergence)
- renders both domains (native) and cross-projected seams (diagnostic)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _capture(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def _create_bundle(python: str, *, label: str) -> Path:
    out = _capture([python, "scripts/new_asset_bundle.py", "create", "--label", label])
    return Path(out).resolve()


def _mesh_counts(path: Path) -> tuple[int, int | None]:
    payload = np.load(path, allow_pickle=True)
    vertices = np.asarray(payload["vertices"])
    faces = np.asarray(payload["faces"]) if "faces" in payload else None
    return int(vertices.shape[0]), (int(faces.shape[0]) if faces is not None else None)


def _default_role(vertex_count: int) -> str:
    # Intentionally conservative default.
    # Role labels are provenance assertions, not geometry classifiers.
    return "unknown"


def _tag(role: str, vertex_count: int) -> str:
    safe_role = str(role).strip().replace(" ", "_")
    return f"{safe_role}_v{int(vertex_count)}"


def _artifact_entry(
    *,
    path: Path | None,
    artifact_level: str,
    role: str = "none",
    topology: str = "none",
    domain: str = "operator",
) -> dict[str, str] | None:
    if path is None:
        return None
    return {
        "path": str(path),
        "artifact_level": artifact_level,
        "role": role,
        "topology": topology,
        "domain": domain,
    }


def _reproject(
    python: str,
    *,
    seam_report: Path,
    source_mesh: Path,
    target_mesh: Path,
    vertex_map: Path | None,
    out_path: Path,
) -> Path:
    cmd = [
        python,
        "scripts/reproject_seam_report.py",
        "--report",
        str(seam_report),
        "--source-mesh",
        str(source_mesh),
        "--target-mesh",
        str(target_mesh),
        "--out",
        str(out_path),
    ]
    if vertex_map is not None:
        cmd += ["--vertex-map-file", str(vertex_map)]
    _run(cmd)
    return out_path


def _render(
    python: str,
    *,
    mesh: Path,
    align_to_mesh: Path | None,
    align_rotate: tuple[float, float, float],
    seam_report: Path | None,
    costs: Path | None,
    out_dir: Path,
    stem: str,
    timestamp: str,
    point_size: int,
    seam_width: int,
    body_alpha: int,
    axis_width: str | None,
    yaw_offset: float,
    rotate_x: float,
    rotate_y: float,
    rotate_z: float,
) -> None:
    cmd = [
        python,
        "scripts/render_seam_orbit.py",
        "--mesh",
        str(mesh),
        "--out-dir",
        str(out_dir),
        "--stem",
        stem,
        "--timestamp",
        timestamp,
        "--point-size",
        str(point_size),
        "--seam-width",
        str(seam_width),
        "--body-alpha",
        str(body_alpha),
        "--yaw-offset",
        str(float(yaw_offset)),
        "--rotate-x",
        str(float(rotate_x)),
        "--rotate-y",
        str(float(rotate_y)),
        "--rotate-z",
        str(float(rotate_z)),
        "--axis-up",
        "auto",
        "--canonicalize",
    ]
    if align_to_mesh is not None:
        cmd += ["--align-to-mesh", str(align_to_mesh)]
        cmd += ["--align-rotate-x", str(float(align_rotate[0]))]
        cmd += ["--align-rotate-y", str(float(align_rotate[1]))]
        cmd += ["--align-rotate-z", str(float(align_rotate[2]))]
    if axis_width is not None:
        cmd += ["--axis-width", str(axis_width)]
    if seam_report is not None:
        cmd += ["--seam-report", str(seam_report)]
    if costs is not None:
        cmd += ["--costs", str(costs)]
    _run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=str, default="python", help="Python interpreter to run scripts with.")
    parser.add_argument("--label", type=str, default="strategy2", help="Bundle label suffix.")

    parser.add_argument("--base-mesh", type=Path, required=True)
    parser.add_argument("--rom-mesh", type=Path, required=True)
    parser.add_argument("--base-seams", type=Path, required=True, help="Seam report solved on base mesh.")
    parser.add_argument("--rom-seams", type=Path, required=True, help="Seam report solved on ROM mesh.")
    parser.add_argument("--base-costs", type=Path, default=None)
    parser.add_argument("--rom-costs", type=Path, default=None)
    parser.add_argument("--rom-basis", type=Path, default=None)
    parser.add_argument("--rom-meta", type=Path, default=None)
    parser.add_argument("--rom-envelope", type=Path, default=None)
    parser.add_argument("--rom-certificate", type=Path, default=None)
    parser.add_argument("--rom-coeff-samples", type=Path, default=None)
    parser.add_argument("--vertex-map", type=Path, default=None, help="Optional NPZ with both map directions.")
    parser.add_argument(
        "--base-role",
        type=str,
        default=None,
        help="Required provenance role label for --base-mesh used in filenames (e.g. 'human').",
    )
    parser.add_argument(
        "--rom-role",
        type=str,
        default=None,
        help="Required provenance role label for --rom-mesh used in filenames (e.g. 'ogre').",
    )
    parser.add_argument(
        "--render-vertex-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also render correspondence orbits into bundle/maps/ when --vertex-map is provided.",
    )

    parser.add_argument("--point-size", type=int, default=6)
    parser.add_argument("--seam-width", type=int, default=6)
    parser.add_argument("--body-alpha", type=int, default=160)
    parser.add_argument(
        "--yaw-offset",
        type=float,
        default=90.0,
        help="Yaw offset (degrees) applied to all orbit renders; use to align 'front' (try 90).",
    )
    parser.add_argument("--base-rotate-x", type=float, default=0.0, help="Rigid X rotation for base mesh renders (deg).")
    parser.add_argument("--base-rotate-y", type=float, default=0.0, help="Rigid Y rotation for base mesh renders (deg).")
    parser.add_argument("--base-rotate-z", type=float, default=0.0, help="Rigid Z rotation for base mesh renders (deg).")
    parser.add_argument("--rom-rotate-x", type=float, default=0.0, help="Rigid X rotation for ROM mesh renders (deg).")
    parser.add_argument("--rom-rotate-y", type=float, default=0.0, help="Rigid Y rotation for ROM mesh renders (deg).")
    parser.add_argument("--rom-rotate-z", type=float, default=0.0, help="Rigid Z rotation for ROM mesh renders (deg).")
    parser.add_argument(
        "--axis-width",
        type=str,
        default="none",
        choices=("x", "y", "z", "none"),
        help="Force a shared render width-axis to align base/ROM orientations. Use 'none' for auto.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )
    args = parser.parse_args()

    python = str(args.python)
    ts = str(args.timestamp)
    axis_width = None if str(args.axis_width).lower() == "none" else str(args.axis_width).lower()
    yaw_offset = float(args.yaw_offset)
    base_rot = (float(args.base_rotate_x), float(args.base_rotate_y), float(args.base_rotate_z))
    rom_rot = (float(args.rom_rotate_x), float(args.rom_rotate_y), float(args.rom_rotate_z))

    base_v, base_f = _mesh_counts(args.base_mesh)
    rom_v, rom_f = _mesh_counts(args.rom_mesh)
    if args.base_role is None or args.rom_role is None:
        raise SystemExit(
            "Missing required provenance roles. Provide both --base-role and --rom-role "
            "(do not infer roles from vertex counts)."
        )
    base_role = str(args.base_role)
    rom_role = str(args.rom_role)
    base_tag = _tag(base_role, base_v)
    rom_tag = _tag(rom_role, rom_v)

    bundle = _create_bundle(python, label=str(args.label))
    renders = bundle / "renders"
    seams = bundle / "seams"
    manifests = bundle / "manifests"
    notes = bundle / "notes"
    maps = bundle / "maps"
    renders.mkdir(parents=True, exist_ok=True)
    seams.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    notes.mkdir(parents=True, exist_ok=True)
    maps.mkdir(parents=True, exist_ok=True)
    run_reference_dir = bundle / "run_reference"
    run_reference_index = run_reference_dir / "index.html"
    runs_index = bundle.parent / "index.html"
    outputs_index = bundle.parent.parent / "index.html"

    operator_report_dir: Path | None = None
    operator_report_index: Path | None = None
    if args.rom_basis is not None and args.rom_meta is not None:
        operator_report_dir = bundle / "rom_operator"
        operator_report_dir.mkdir(parents=True, exist_ok=True)

    # 1) Audit stages (mesh + seam provenance + vertex maps).
    audit_json = manifests / "pipeline_audit.json"
    audit_cmd = [
        python,
        "scripts/audit_pipeline_stages.py",
        "--out-json",
        str(audit_json),
        "--stage",
        f"base_mesh={args.base_mesh}",
        "--stage",
        f"rom_mesh={args.rom_mesh}",
        "--stage",
        f"base_seams={args.base_seams}",
        "--stage",
        f"rom_seams={args.rom_seams}",
        "--discover-from-seam-report",
        str(args.base_seams),
        "--discover-from-seam-report",
        str(args.rom_seams),
    ]
    if args.vertex_map is not None:
        audit_cmd += ["--stage", f"vertex_map={args.vertex_map}"]
    _run(audit_cmd)

    # 1b) Render vertex map correspondence orbits (high-collision maps should look obviously wrong).
    if args.vertex_map is not None and bool(args.render_vertex_map):
        for direction in ("source_to_target", "target_to_source"):
            map_name = (
                f"vertex_map__{rom_tag}__to__{base_tag}"
                if direction == "source_to_target"
                else f"vertex_map__{base_tag}__to__{rom_tag}"
            )
            out_dir = maps / f"{map_name}__{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    python,
                    "scripts/render_vertex_map_orbits.py",
                    "--source-mesh",
                    str(args.rom_mesh),
                    "--target-mesh",
                    str(args.base_mesh),
                    "--vertex-map",
                    str(args.vertex_map),
                    "--direction",
                    "source_to_target" if direction == "source_to_target" else "target_to_source",
                    "--out-dir",
                    str(out_dir),
                    "--timestamp",
                    ts,
                    "--point-size",
                    str(int(args.point_size)),
                    *(["--axis-width", axis_width] if axis_width is not None else []),
                    "--yaw-offset",
                    str(float(yaw_offset)),
                    "--source-rotate-x",
                    str(float(rom_rot[0])),
                    "--source-rotate-y",
                    str(float(rom_rot[1])),
                    "--source-rotate-z",
                    str(float(rom_rot[2])),
                    "--target-rotate-x",
                    str(float(base_rot[0])),
                    "--target-rotate-y",
                    str(float(base_rot[1])),
                    "--target-rotate-z",
                    str(float(base_rot[2])),
                    "--show-lines",
                    "--show-source",
                ]
            )

    # 2) Reproject seams both ways (diagnostic; non-bijective maps are expected to fail quality checks).
    base_with_rom = seams / f"{base_tag}__seams_from__{rom_tag}__reprojected.json"
    rom_with_base = seams / f"{rom_tag}__seams_from__{base_tag}__reprojected.json"
    _reproject(
        python,
        seam_report=args.rom_seams,
        source_mesh=args.rom_mesh,
        target_mesh=args.base_mesh,
        vertex_map=args.vertex_map,
        out_path=base_with_rom,
    )
    _reproject(
        python,
        seam_report=args.base_seams,
        source_mesh=args.base_mesh,
        target_mesh=args.rom_mesh,
        vertex_map=args.vertex_map,
        out_path=rom_with_base,
    )

    # 3) Render native + reprojected for both meshes, plus mesh-only.
    _render(
        python,
        mesh=args.base_mesh,
        align_to_mesh=None,
        align_rotate=(0.0, 0.0, 0.0),
        seam_report=args.base_seams,
        costs=args.base_costs,
        out_dir=renders,
        stem=f"{base_tag}__native_seams",
        timestamp=ts,
        point_size=int(args.point_size),
        seam_width=int(args.seam_width),
        body_alpha=int(args.body_alpha),
        axis_width=axis_width,
        yaw_offset=yaw_offset,
        rotate_x=base_rot[0],
        rotate_y=base_rot[1],
        rotate_z=base_rot[2],
    )
    _render(
        python,
        mesh=args.base_mesh,
        align_to_mesh=None,
        align_rotate=(0.0, 0.0, 0.0),
        seam_report=base_with_rom,
        costs=args.base_costs,
        out_dir=renders,
        stem=f"{base_tag}__reprojected_seams_from__{rom_tag}",
        timestamp=ts,
        point_size=int(args.point_size),
        seam_width=int(args.seam_width),
        body_alpha=int(args.body_alpha),
        axis_width=axis_width,
        yaw_offset=yaw_offset,
        rotate_x=base_rot[0],
        rotate_y=base_rot[1],
        rotate_z=base_rot[2],
    )
    _render(
        python,
        mesh=args.rom_mesh,
        align_to_mesh=args.base_mesh,
        align_rotate=base_rot,
        seam_report=args.rom_seams,
        costs=args.rom_costs,
        out_dir=renders,
        stem=f"{rom_tag}__native_seams",
        timestamp=ts,
        point_size=int(args.point_size),
        seam_width=int(args.seam_width),
        body_alpha=int(args.body_alpha),
        axis_width=axis_width,
        yaw_offset=yaw_offset,
        rotate_x=rom_rot[0],
        rotate_y=rom_rot[1],
        rotate_z=rom_rot[2],
    )
    _render(
        python,
        mesh=args.rom_mesh,
        align_to_mesh=args.base_mesh,
        align_rotate=base_rot,
        seam_report=rom_with_base,
        costs=args.rom_costs,
        out_dir=renders,
        stem=f"{rom_tag}__reprojected_seams_from__{base_tag}",
        timestamp=ts,
        point_size=int(args.point_size),
        seam_width=int(args.seam_width),
        body_alpha=int(args.body_alpha),
        axis_width=axis_width,
        yaw_offset=yaw_offset,
        rotate_x=rom_rot[0],
        rotate_y=rom_rot[1],
        rotate_z=rom_rot[2],
    )
    _render(
        python,
        mesh=args.base_mesh,
        align_to_mesh=None,
        align_rotate=(0.0, 0.0, 0.0),
        seam_report=None,
        costs=args.base_costs,
        out_dir=renders,
        stem=f"{base_tag}__mesh_only",
        timestamp=ts,
        point_size=int(args.point_size),
        seam_width=int(args.seam_width),
        body_alpha=int(args.body_alpha),
        axis_width=axis_width,
        yaw_offset=yaw_offset,
        rotate_x=base_rot[0],
        rotate_y=base_rot[1],
        rotate_z=base_rot[2],
    )
    _render(
        python,
        mesh=args.rom_mesh,
        align_to_mesh=args.base_mesh,
        align_rotate=base_rot,
        seam_report=None,
        costs=args.rom_costs,
        out_dir=renders,
        stem=f"{rom_tag}__mesh_only",
        timestamp=ts,
        point_size=int(args.point_size),
        seam_width=int(args.seam_width),
        body_alpha=int(args.body_alpha),
        axis_width=axis_width,
        yaw_offset=yaw_offset,
        rotate_x=rom_rot[0],
        rotate_y=rom_rot[1],
        rotate_z=rom_rot[2],
    )

    if operator_report_dir is not None:
        report_cmd = [
            python,
            "scripts/render_rom_operator_report.py",
            "--basis",
            str(args.rom_basis),
            "--rom-meta",
            str(args.rom_meta),
            "--out-dir",
            str(operator_report_dir),
            "--media-path",
            str(renders),
            "--media-path",
            str(maps),
            "--media-path",
            str(Path(args.base_seams).parent),
            "--media-path",
            str(Path(args.rom_seams).parent),
            "--media-path",
            str(Path(base_with_rom).parent),
            "--media-path",
            str(Path(rom_with_base).parent),
        ]
        if args.rom_coeff_samples is not None:
            report_cmd += ["--coeff-samples", str(args.rom_coeff_samples)]
        if args.rom_envelope is not None:
            report_cmd += ["--envelope", str(args.rom_envelope)]
        if args.rom_certificate is not None:
            report_cmd += ["--certificate", str(args.rom_certificate)]
        if args.rom_costs is not None:
            report_cmd += ["--costs", str(args.rom_costs)]
        if args.rom_mesh is not None:
            report_cmd += ["--body", str(args.rom_mesh)]
        _run(report_cmd)
        operator_report_index = operator_report_dir / "index.html"

    protocol_manifest = {
        "timestamp_utc": ts,
        "bundle": str(bundle),
        "inputs": {
            "base_mesh": str(args.base_mesh),
            "rom_mesh": str(args.rom_mesh),
            "base_role": base_role,
            "rom_role": rom_role,
            "base_tag": base_tag,
            "rom_tag": rom_tag,
            "base_vertex_count": base_v,
            "rom_vertex_count": rom_v,
            "base_face_count": base_f,
            "rom_face_count": rom_f,
            "base_seams": str(args.base_seams),
            "rom_seams": str(args.rom_seams),
            "vertex_map": str(args.vertex_map) if args.vertex_map is not None else None,
            "base_costs": str(args.base_costs) if args.base_costs is not None else None,
            "rom_costs": str(args.rom_costs) if args.rom_costs is not None else None,
            "rom_basis": str(args.rom_basis) if args.rom_basis is not None else None,
            "rom_meta": str(args.rom_meta) if args.rom_meta is not None else None,
            "rom_envelope": str(args.rom_envelope) if args.rom_envelope is not None else None,
            "rom_certificate": str(args.rom_certificate) if args.rom_certificate is not None else None,
            "rom_coeff_samples": str(args.rom_coeff_samples) if args.rom_coeff_samples is not None else None,
            "render_axis_width": axis_width,
            "render_yaw_offset_deg": float(yaw_offset),
            "base_render_rotate_deg": {"x": base_rot[0], "y": base_rot[1], "z": base_rot[2]},
            "rom_render_rotate_deg": {"x": rom_rot[0], "y": rom_rot[1], "z": rom_rot[2]},
        },
        "outputs": {
            "audit_json": str(audit_json),
            "base_with_rom": str(base_with_rom),
            "rom_with_base": str(rom_with_base),
            "renders_dir": str(renders),
            "maps_dir": str(maps),
            "rom_operator_report_dir": str(operator_report_dir) if operator_report_dir is not None else None,
            "rom_operator_report_index": str(operator_report_index) if operator_report_index is not None else None,
            "run_reference_dir": str(run_reference_dir),
            "run_reference_index": str(run_reference_index),
            "runs_index": str(runs_index),
            "outputs_index": str(outputs_index),
        },
        "artifacts": [
            entry
            for entry in [
                _artifact_entry(path=args.rom_basis, artifact_level="operator"),
                _artifact_entry(path=args.rom_meta, artifact_level="operator"),
                _artifact_entry(path=args.rom_envelope, artifact_level="operator"),
                _artifact_entry(path=args.rom_certificate, artifact_level="operator"),
                _artifact_entry(path=args.rom_coeff_samples, artifact_level="operator"),
                _artifact_entry(path=operator_report_index, artifact_level="operator"),
                _artifact_entry(
                    path=args.base_mesh,
                    artifact_level="topology",
                    role=base_role,
                    topology=f"v{base_v}",
                    domain="native",
                ),
                _artifact_entry(
                    path=args.rom_mesh,
                    artifact_level="topology",
                    role=rom_role,
                    topology=f"v{rom_v}",
                    domain="native",
                ),
                _artifact_entry(
                    path=args.base_seams,
                    artifact_level="topology",
                    role=base_role,
                    topology=f"v{base_v}",
                    domain="native",
                ),
                _artifact_entry(
                    path=args.rom_seams,
                    artifact_level="topology",
                    role=rom_role,
                    topology=f"v{rom_v}",
                    domain="native",
                ),
                _artifact_entry(
                    path=base_with_rom,
                    artifact_level="topology",
                    role=base_role,
                    topology=f"v{base_v}",
                    domain=f"reprojected_from_{rom_tag}",
                ),
                _artifact_entry(
                    path=rom_with_base,
                    artifact_level="topology",
                    role=rom_role,
                    topology=f"v{rom_v}",
                    domain=f"reprojected_from_{base_tag}",
                ),
                _artifact_entry(
                    path=args.base_costs,
                    artifact_level="topology",
                    role=base_role,
                    topology=f"v{base_v}",
                    domain="native",
                ),
                _artifact_entry(
                    path=args.rom_costs,
                    artifact_level="topology",
                    role=rom_role,
                    topology=f"v{rom_v}",
                    domain="native",
                ),
                _artifact_entry(
                    path=args.vertex_map,
                    artifact_level="topology",
                    role="none",
                    topology="none",
                    domain=f"map_{rom_tag}_to_{base_tag}",
                ),
            ]
            if entry is not None
        ],
        "notes": {
            "protocol": "Render ROM native + reprojected (Strategy 2). "
            "Do not assume vertex maps are invertible unless topology is identical.",
            "artifact_levels": "operator = basis/coefficients/envelopes/certificates/report; topology = mesh/cost/seam/render/reprojection artifacts.",
            "primary_view_surface": "run_reference/index.html is the canonical single-run page; rom_operator/index.html remains a specialized subpage.",
        },
    }
    (manifests / "protocol_strategy2.json").write_text(
        json.dumps(protocol_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (notes / "how_to_view.txt").write_text(
        "\n".join(
            [
                f"Bundle: {bundle}",
                "Key renders in renders/:",
                f"  {base_tag}__native_seams__orbit__{ts}.webm",
                f"  {base_tag}__reprojected_seams_from__{rom_tag}__orbit__{ts}.webm",
                f"  {rom_tag}__native_seams__orbit__{ts}.webm",
                f"  {rom_tag}__reprojected_seams_from__{base_tag}__orbit__{ts}.webm",
                f"  {base_tag}__mesh_only__orbit__{ts}.webm",
                f"  {rom_tag}__mesh_only__orbit__{ts}.webm",
                "",
                "Audit:",
                f"  manifests/pipeline_audit.json (and .csv alongside if requested)",
                "",
                "A/B comparison metrics:",
                "  manifests/seam_compare_metrics.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # 4) Emit A/B comparison metrics to make Strategy A vs B a controlled experiment.
    _run(
        [
            python,
            "scripts/seam_compare_metrics.py",
            "--bundle",
            str(bundle),
        ]
    )

    # 5) Build a canonical run-level reference page and refresh the runs index.
    _run(
        [
            python,
            "scripts/render_run_reference.py",
            "--run-root",
            str(bundle),
            "--out-dir",
            str(run_reference_dir),
            "--title",
            f"Strategy 2 Bundle: {bundle.name}",
        ]
    )
    _run(
        [
            python,
            "scripts/render_run_index.py",
            "--runs-root",
            str(bundle.parent),
            "--out",
            str(runs_index),
        ]
    )
    comparisons_root = bundle.parent.parent / "comparisons"
    seams_root = bundle.parent.parent / "seams_run"
    unified_index_cmd = [
        python,
        "scripts/render_run_index.py",
        "--runs-root",
        str(bundle.parent),
    ]
    if comparisons_root.exists():
        unified_index_cmd += ["--runs-root", str(comparisons_root)]
    if seams_root.exists():
        unified_index_cmd += ["--runs-root", str(seams_root)]
    unified_index_cmd += ["--out", str(outputs_index)]
    _run(unified_index_cmd)

    print(bundle)


if __name__ == "__main__":
    main()
