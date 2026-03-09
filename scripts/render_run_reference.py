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
            if path.name in {"rom_run.json", "afflec_basis.npz", "afflec_coeff_samples.json", "seam_costs.npz"}
            or path.parent.name.startswith("rom_")
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
        },
        "sections": grouped,
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
