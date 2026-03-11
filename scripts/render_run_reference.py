#!/usr/bin/env python3
"""Render a canonical HTML reference page for one run root."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MEDIA_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".webm",
    ".mp4",
    ".mov",
    ".m4v",
}
BODY_EXTS = {".npz"}
IGNORE_DIR_PARTS = ("frames", "orbit_frames", "map_orbit_frames")
IGNORE_FILES = {"coeff_top_variance.png", "coeff_sample_norms.png"}
MORPHOLOGY_OVERRIDE_FILES = ("morphology_observations.json", "morphology_audit_overrides.json")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _should_ignore(path: Path, *, out_dir: Path) -> bool:
    try:
        path.relative_to(out_dir)
        return True
    except ValueError:
        pass
    parts_lower = tuple(part.lower() for part in path.parts)
    if any(token in part for part in parts_lower for token in IGNORE_DIR_PARTS):
        return True
    if path.name in IGNORE_FILES:
        return True
    if path.name.startswith("frame_") and path.suffix.lower() == ".png":
        return True
    return False


def _discover_files(run_root: Path, *, out_dir: Path) -> list[Path]:
    discovered: list[Path] = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file():
            continue
        if _should_ignore(path, out_dir=out_dir):
            continue
        discovered.append(path)
    return discovered


def _relative(path: Path, *, out_dir: Path) -> str:
    try:
        return os.path.relpath(path, out_dir)
    except ValueError:
        return str(path)


def _media_entry(path: Path, *, out_dir: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    kind = "video" if suffix in {".webm", ".mp4", ".mov", ".m4v"} else "image"
    mime_type = mimetypes.guess_type(str(path))[0] or ("video/webm" if kind == "video" else "image/png")
    return {
        "path": str(path),
        "relative_path": _relative(path, out_dir=out_dir),
        "kind": kind,
        "mime_type": mime_type,
        "parent": path.parent.name,
        "name": path.name,
        "sha256": _sha256_path(path),
    }


def _subpage_entry(path: Path, *, out_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": _relative(path, out_dir=out_dir),
        "name": path.name,
        "parent": path.parent.name,
    }


def _artifact_table(paths: Iterable[Path], *, out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.append(
            {
                "path": str(path),
                "relative_path": _relative(path, out_dir=out_dir),
                "parent": path.parent.name,
                "name": path.name,
                "sha256": _sha256_path(path),
            }
        )
    return rows


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _npz_mesh_summary(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        vertices = np.asarray(data["vertices"], dtype=float)
        faces = np.asarray(data["faces"], dtype=int) if "faces" in data else np.zeros((0, 3), dtype=int)
    vertex_digest = hashlib.sha256(vertices.tobytes()).hexdigest()
    return {
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "geometry_sha256": vertex_digest,
    }


def _load_morphology_overrides(run_root: Path) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for filename in MORPHOLOGY_OVERRIDE_FILES:
        payload = _load_json(run_root / filename)
        if not payload:
            continue
        entries = payload.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and isinstance(entry.get("artifact"), str):
                    overrides[entry["artifact"]] = entry
        if isinstance(payload.get("artifacts"), dict):
            for artifact, entry in payload["artifacts"].items():
                if isinstance(artifact, str) and isinstance(entry, dict):
                    overrides[artifact] = entry
    return overrides


def _lookup_override(overrides: dict[str, dict[str, Any]], artifact: str) -> dict[str, Any] | None:
    direct = overrides.get(artifact)
    if direct is not None:
        return direct
    normalized = artifact
    while normalized.startswith("../"):
        normalized = normalized[3:]
    direct = overrides.get(normalized)
    if direct is not None:
        return direct
    for key, value in overrides.items():
        if artifact.endswith(key) or normalized.endswith(key):
            return value
    return None


def _expected_morphology(stage: str, *, geometry_changed: bool, artifact_kind: str) -> str:
    if stage == "body_fit":
        return "base_or_expected"
    if stage == "rom_sample_pose":
        return "flailing_or_pose_deformed"
    if stage == "rom_operator":
        return "base_geometry_unchanged"
    if stage == "seam_reprojection":
        return "base_geometry_unchanged"
    if artifact_kind == "render_view" and geometry_changed:
        return "same_as_source_geometry"
    return "base_geometry_unchanged"


def _stage_sort_key(stage: str) -> int:
    order = {
        "body_fit": 0,
        "rom_sample_pose": 1,
        "rom_operator": 2,
        "seam_solution": 3,
        "seam_reprojection": 4,
        "render_view": 5,
        "manifest": 6,
    }
    return order.get(stage, 99)


def _render_manifest_entries(files: list[Path], *, out_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in files:
        if not (path.parent.name == "renders" and path.name.endswith(".json") and "__manifest__" in path.stem):
            continue
        payload = _load_json(path)
        if not payload:
            continue
        mesh_info = payload.get("mesh") if isinstance(payload.get("mesh"), dict) else {}
        seam_info = payload.get("seam_report") if isinstance(payload.get("seam_report"), dict) else {}
        rel = _relative(path, out_dir=out_dir)
        name = path.name
        if "__mesh_only__" in name:
            stage = "render_view"
            note = "Render of body geometry without seams; use mesh hash to trace morphology source."
        elif "__native_seams__" in name:
            stage = "render_view"
            note = "Render of native seam solution on source topology; seam changes without changing geometry."
        elif "__reprojected_seams_from__" in name:
            stage = "render_view"
            note = "Render of reprojected seam artifact; apparent morphology can come from target geometry plus transfer."
        else:
            stage = "render_view"
            note = "Render artifact."
        entries.append(
            {
                "artifact": rel,
                "artifact_kind": "render_view",
                "stage": stage,
                "topology": mesh_info.get("vertex_count"),
                "geometry_sha256": mesh_info.get("sha256"),
                "geometry_changed": False,
                "observed_morphology": "unclassified",
                "expected_morphology": _expected_morphology(stage, geometry_changed=False, artifact_kind="render_view"),
                "source_mesh": mesh_info.get("path"),
                "related_artifact": seam_info.get("path"),
                "diagnostic_note": note,
            }
        )
    return entries


def _morphology_audit(files: list[Path], *, run_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    overrides = _load_morphology_overrides(run_root)
    entries: list[dict[str, Any]] = []
    matched_override_keys: set[str] = set()

    body_paths = sorted(
        path
        for path in files
        if path.suffix.lower() in BODY_EXTS and ("body" in path.stem or path.parent.name.startswith("body"))
    )
    previous_geometry_hash: str | None = None
    for path in body_paths:
        rel = _relative(path, out_dir=out_dir)
        mesh = _npz_mesh_summary(path)
        geometry_changed = previous_geometry_hash is None or mesh["geometry_sha256"] != previous_geometry_hash
        entries.append(
            {
                "artifact": rel,
                "artifact_kind": "body_mesh",
                "stage": "body_fit",
                "topology": mesh["vertex_count"],
                "geometry_sha256": mesh["geometry_sha256"],
                "geometry_changed": geometry_changed,
                "observed_morphology": "unclassified",
                "expected_morphology": _expected_morphology("body_fit", geometry_changed=geometry_changed, artifact_kind="body_mesh"),
                "source_mesh": None,
                "related_artifact": None,
                "diagnostic_note": "Body geometry artifact; use this as the authoritative morphology source for downstream stages on the same topology.",
            }
        )
        previous_geometry_hash = str(mesh["geometry_sha256"])

    for path in files:
        rel = _relative(path, out_dir=out_dir)
        if path.name in {"seam_report.json"} or path.name.endswith("_reprojected.json"):
            is_reprojected = path.name.endswith("_reprojected.json")
            entries.append(
                {
                    "artifact": rel,
                    "artifact_kind": "seam_report",
                    "stage": "seam_reprojection" if is_reprojected else "seam_solution",
                    "topology": None,
                    "geometry_sha256": None,
                    "geometry_changed": False,
                    "observed_morphology": "inherits_source_geometry",
                    "expected_morphology": _expected_morphology(
                        "seam_reprojection" if is_reprojected else "seam_solution",
                        geometry_changed=False,
                        artifact_kind="seam_report",
                    ),
                    "source_mesh": None,
                    "related_artifact": None,
                    "diagnostic_note": "Seam solve changes edge/path structure but not body geometry."
                    if not is_reprojected
                    else "Reprojection changes seam placement across topologies; apparent morphology may reflect transfer artifacts rather than geometry change.",
                }
            )
        elif path.suffix.lower() in BODY_EXTS and path.parent.name == "rom_samples" and path.name.startswith("sample_"):
            mesh = _npz_mesh_summary(path)
            entries.append(
                {
                    "artifact": rel,
                    "artifact_kind": "rom_sample_mesh",
                    "stage": "rom_sample_pose",
                    "topology": mesh["vertex_count"],
                    "geometry_sha256": mesh["geometry_sha256"],
                    "geometry_changed": True,
                    "observed_morphology": "unclassified",
                    "expected_morphology": _expected_morphology(
                        "rom_sample_pose",
                        geometry_changed=True,
                        artifact_kind="rom_sample_mesh",
                    ),
                    "source_mesh": None,
                    "related_artifact": None,
                    "diagnostic_note": "Representative ROM-native posed/deformed sample mesh; inspect this stage directly when judging flailing or other morphology changes.",
                }
            )
        elif path.name in {"rom_run.json", "afflec_coeff_samples.json", "coeff_summary.json", "report_manifest.json"} or path.name == "seam_costs.npz":
            parent = path.parent.name
            if parent == "rom_operator" or "operator" in parent or path.name in {"coeff_summary.json", "report_manifest.json"}:
                note = "Operator-level ROM artifact; should not change geometry by itself."
            elif "sample" in path.name:
                note = "ROM sample/coefficient artifact; inspect alongside posed samples to understand flailing."
            else:
                note = "ROM-derived field artifact projected onto an existing topology."
            entries.append(
                {
                    "artifact": rel,
                    "artifact_kind": "rom_artifact",
                    "stage": "rom_operator",
                    "topology": None,
                    "geometry_sha256": None,
                    "geometry_changed": False,
                    "observed_morphology": "field_only",
                    "expected_morphology": _expected_morphology("rom_operator", geometry_changed=False, artifact_kind="rom_artifact"),
                    "source_mesh": None,
                    "related_artifact": None,
                    "diagnostic_note": note,
                }
            )

    entries.extend(_render_manifest_entries(files, out_dir=out_dir))

    for entry in entries:
        override = _lookup_override(overrides, str(entry["artifact"]))
        if not override:
            continue
        for key, value in overrides.items():
            artifact = str(entry["artifact"])
            normalized = artifact
            while normalized.startswith("../"):
                normalized = normalized[3:]
            if artifact == key or normalized == key or artifact.endswith(key) or normalized.endswith(key):
                matched_override_keys.add(key)
        for key in ("observed_morphology", "expected_morphology", "diagnostic_note", "source_mesh", "related_artifact"):
            if key in override:
                entry[key] = override[key]
        if "geometry_changed" in override:
            entry["geometry_changed"] = bool(override["geometry_changed"])

    for key, override in overrides.items():
        if key in matched_override_keys:
            continue
        stage = str(override.get("stage", "manifest"))
        artifact_kind = str(override.get("artifact_kind", "manual_observation"))
        geometry_changed = bool(override.get("geometry_changed", False))
        entries.append(
            {
                "artifact": key,
                "artifact_kind": artifact_kind,
                "stage": stage,
                "topology": override.get("topology"),
                "geometry_sha256": override.get("geometry_sha256"),
                "geometry_changed": geometry_changed,
                "observed_morphology": str(override.get("observed_morphology", "unclassified")),
                "expected_morphology": str(
                    override.get(
                        "expected_morphology",
                        _expected_morphology(stage, geometry_changed=geometry_changed, artifact_kind=artifact_kind),
                    )
                ),
                "source_mesh": override.get("source_mesh"),
                "related_artifact": override.get("related_artifact"),
                "diagnostic_note": str(override.get("diagnostic_note", "Manual morphology observation.")),
            }
        )
    entries.sort(key=lambda entry: (_stage_sort_key(str(entry["stage"])), str(entry["artifact"])))
    return entries


def _morphology_audit_section(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No morphology audit entries detected.</p>"
    return (
        "<table><tr><th>Stage</th><th>Artifact</th><th>Geometry changed</th><th>Observed</th><th>Expected</th><th>Note</th></tr>"
        + "".join(
            f'<tr><td>{html.escape(str(row["stage"]))}</td>'
            f'<td><a href="{html.escape(str(row["artifact"]))}"><code>{html.escape(Path(str(row["artifact"])).name)}</code></a></td>'
            f'<td>{html.escape("yes" if bool(row["geometry_changed"]) else "no")}</td>'
            f'<td>{html.escape(str(row["observed_morphology"]))}</td>'
            f'<td>{html.escape(str(row["expected_morphology"]))}</td>'
            f'<td>{html.escape(str(row["diagnostic_note"]))}</td></tr>'
            for row in rows
        )
        + "</table>"
    )


def _group_files(files: list[Path], *, run_root: Path, out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    media = [_media_entry(path, out_dir=out_dir) for path in files if path.suffix.lower() in MEDIA_EXTS]
    subpages = [
        _subpage_entry(path, out_dir=out_dir)
        for path in files
        if path.name == "index.html" and path.parent != out_dir
    ]
    bodies = _artifact_table(
        [
            path
            for path in files
            if path.suffix.lower() in BODY_EXTS
            and ("body" in path.stem or path.parent.name.startswith("body"))
        ],
        out_dir=out_dir,
    )
    fit_json = _artifact_table(
        [path for path in files if path.name.endswith("_fit_diagnostics.json") or path.name == "afflec_fit_diagnostics.json"],
        out_dir=out_dir,
    )
    rom_artifacts = _artifact_table(
        [
            path
            for path in files
            if path.name in {"rom_run.json", "afflec_basis.npz", "afflec_coeff_samples.json", "seam_costs.npz", "rom_sample_manifest.json"}
            or path.parent.name.startswith("rom_")
            or path.parent.name == "rom_samples"
        ],
        out_dir=out_dir,
    )
    seam_reports = _artifact_table(
        [path for path in files if path.name == "seam_report.json" or path.name.endswith("_reprojected.json")],
        out_dir=out_dir,
    )
    manifests = _artifact_table(
        [
            path
            for path in files
            if path.suffix.lower() == ".json"
            and (
                path.parent.name in {"manifests", "notes"}
                or path.name.endswith("_manifest.json")
                or path.name == "report_manifest.json"
                or path.name == "pipeline_audit.json"
                or path.name == "seam_compare_metrics.json"
            )
        ],
        out_dir=out_dir,
    )
    return {
        "media": media,
        "subpages": subpages,
        "bodies": bodies,
        "fit": fit_json,
        "rom": rom_artifacts,
        "seams": seam_reports,
        "manifests": manifests,
    }


def _media_html(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "<p>No media artifacts found for this run.</p>"
    cards: list[str] = []
    for entry in entries:
        rel = html.escape(str(entry["relative_path"]))
        name = html.escape(str(entry["name"]))
        parent = html.escape(str(entry["parent"]))
        if entry["kind"] == "video":
            tag = (
                '<video controls preload="metadata" style="width:100%;background:#111;border-radius:10px;">'
                f'<source src="{rel}" type="{html.escape(str(entry["mime_type"]))}"></video>'
            )
        else:
            tag = (
                f'<img src="{rel}" alt="{name}" '
                'style="width:100%;border-radius:10px;border:1px solid #d8d1c7;background:#fff;" />'
            )
        cards.append(
            "<figure class=\"media-card\">"
            f"<div class=\"media-group\">{parent}</div>"
            f"{tag}"
            f"<figcaption><code>{name}</code></figcaption>"
            "</figure>"
        )
    return f'<div class="media-grid">{"".join(cards)}</div>'


def _link_list(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "<p>None detected.</p>"
    items = "".join(
        f'<li><a href="{html.escape(str(entry["relative_path"]))}"><code>{html.escape(str(entry["parent"]))}/{html.escape(str(entry["name"]))}</code></a></li>'
        for entry in entries
    )
    return f"<ul>{items}</ul>"


def _artifact_section(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>None detected.</p>"
    return (
        "<table><tr><th>Parent</th><th>Artifact</th><th>SHA256</th></tr>"
        + "".join(
            f'<tr><td>{html.escape(str(row["parent"]))}</td>'
            f'<td><a href="{html.escape(str(row["relative_path"]))}"><code>{html.escape(str(row["name"]))}</code></a></td>'
            f'<td><code>{html.escape(str(row["sha256"]))}</code></td></tr>'
            for row in rows
        )
        + "</table>"
    )


def render_run_reference(run_root: Path, *, out_dir: Path, title: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _discover_files(run_root, out_dir=out_dir)
    grouped = _group_files(files, run_root=run_root, out_dir=out_dir)
    morphology_audit = _morphology_audit(files, run_root=run_root, out_dir=out_dir)
    manifest = {
        "run_root": str(run_root),
        "summary": {
            "file_count": len(files),
            "media_count": len(grouped["media"]),
            "subpage_count": len(grouped["subpages"]),
            "body_count": len(grouped["bodies"]),
            "rom_artifact_count": len(grouped["rom"]),
            "seam_report_count": len(grouped["seams"]),
            "manifest_count": len(grouped["manifests"]),
            "morphology_audit_count": len(morphology_audit),
        },
        "sections": grouped,
        "morphology_audit": morphology_audit,
    }
    manifest_path = out_dir / "run_report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    heading = title or f"Run Reference: {run_root.name}"
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(heading)}</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; color: #222; background: #f4efe8; }}
    .callout {{ border-left: 4px solid #2f6db3; padding: 0.75rem 1rem; background: #f6fbff; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.45rem 0.55rem; text-align: left; vertical-align: top; }}
    th {{ background: #f5f5f5; width: 16rem; }}
    code {{ background: #f4f4f4; padding: 0.08rem 0.25rem; }}
    .media-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }}
    .media-card {{ margin: 0; padding: 0.75rem; background: #fffaf2; border: 1px solid #d7cec1; border-radius: 14px; }}
    .media-group {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #7a6448; margin-bottom: 0.5rem; }}
  </style>
</head>
<body>
  <h1>{html.escape(heading)}</h1>
  <p class="callout">This page is the canonical run-level viewing surface for completed artifacts under <code>{html.escape(str(run_root))}</code>. It intentionally excludes transient frame PNG directories and legacy operator-chart PNGs.</p>
  <h2>Specialized Pages</h2>
  {_link_list(grouped["subpages"])}
  <h2>Body Fit</h2>
  {_artifact_section(grouped["bodies"] + grouped["fit"])}
  <h2>Morphology Audit</h2>
  <p class="callout">This ledger distinguishes where geometry actually changes from stages that only change fields, seams, or rendering. Use it to track whether a run stays base/expected or starts looking flailing or ogre-like.</p>
  {_morphology_audit_section(morphology_audit)}
  <h2>ROM Operator</h2>
  {_artifact_section(grouped["rom"])}
  <h2>Seam Reports</h2>
  {_artifact_section(grouped["seams"])}
  <h2>Media Gallery</h2>
  {_media_html(grouped["media"])}
  <h2>Provenance and Manifests</h2>
  {_artifact_section(grouped["manifests"])}
</body>
</html>
"""
    html_path = out_dir / "index.html"
    html_path.write_text(html_text, encoding="utf-8")
    return html_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Run root containing body/ROM/seam/media artifacts.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for the run reference page.")
    parser.add_argument("--title", type=str, default=None, help="Optional page title override.")
    args = parser.parse_args(argv)

    run_root = args.run_root.resolve()
    out_dir = (args.out_dir or (run_root / "run_reference")).resolve()
    html_path = render_run_reference(run_root, out_dir=out_dir, title=args.title)
    print(html_path)


if __name__ == "__main__":
    main()
