#!/usr/bin/env python3
"""Render an index page for run reference pages across one or more roots."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from datetime import datetime
from pathlib import Path


_TIMESTAMP_PREFIX_RE = re.compile(r"^(?P<stamp>\d{8}_\d{6})(?:__.*)?$")


def _infer_timestamp(run_dir: Path) -> tuple[str, str]:
    match = _TIMESTAMP_PREFIX_RE.match(run_dir.name)
    if match:
        stamp = match.group("stamp")
        parsed = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
        return stamp, parsed.strftime("%Y-%m-%d %H:%M:%S")
    modified = datetime.fromtimestamp(run_dir.stat().st_mtime)
    return "mtime", modified.strftime("%Y-%m-%d %H:%M:%S")


def _run_type_label(runs_root: Path) -> str:
    name = runs_root.name.lower()
    if name == "comparisons":
        return "comparison"
    if name == "assets_bundles":
        return "bundle"
    if name == "seams_run":
        return "seam_run"
    if name == "bodies":
        return "body_fit"
    if name == "rom":
        return "rom"
    return runs_root.name


def _run_entry(run_dir: Path, *, runs_root: Path) -> dict[str, object]:
    report_dir = run_dir / "run_reference"
    report_index = report_dir / "index.html"
    manifest_path = report_dir / "run_report_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    summary = manifest.get("summary", {}) if isinstance(manifest, dict) else {}
    timestamp_key, timestamp_display = _infer_timestamp(run_dir)
    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "run_type": _run_type_label(runs_root),
        "source_root": str(runs_root),
        "timestamp_key": timestamp_key,
        "timestamp_display": timestamp_display,
        "has_report": report_index.exists(),
        "report_path": str(report_index) if report_index.exists() else None,
        "summary": summary,
    }


def _report_href(report_path: str | None, *, out_path: Path) -> str | None:
    if report_path is None:
        return None
    return os.path.relpath(report_path, out_path.parent)


def render_run_index(runs_roots: list[Path], *, out_path: Path) -> Path:
    runs: list[dict[str, object]] = []
    for runs_root in runs_roots:
        runs.extend(_run_entry(path, runs_root=runs_root) for path in sorted(runs_root.iterdir()) if path.is_dir())
    runs.sort(key=lambda run: (str(run["timestamp_key"]), str(run["name"])), reverse=True)
    payload = {
        "runs_roots": [str(root) for root in runs_roots],
        "run_count": len(runs),
        "runs": runs,
    }
    manifest_path = out_path.with_suffix(".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    row_parts: list[str] = []
    for run in runs:
        report_href = _report_href(str(run["report_path"]) if run["report_path"] is not None else None, out_path=out_path)
        row_parts.append(
            "<tr>"
            f"<td>{html.escape(str(run['timestamp_display']))}</td>"
            f"<td>{html.escape(str(run['run_type']))}</td>"
            f"<td><code>{html.escape(str(run['name']))}</code></td>"
            f"<td>{'yes' if run['has_report'] else 'no'}</td>"
            f"<td>{html.escape(str((run['summary'] or {}).get('media_count', 'n/a')))}</td>"
            f"<td>{html.escape(str((run['summary'] or {}).get('body_count', 'n/a')))}</td>"
            f"<td>{html.escape(str((run['summary'] or {}).get('rom_artifact_count', 'n/a')))}</td>"
            f"<td>{html.escape(str((run['summary'] or {}).get('seam_report_count', 'n/a')))}</td>"
            + (
                f'<td><a href="{html.escape(report_href)}">open</a></td>'
                if run["has_report"] and report_href is not None
                else "<td>missing</td>"
            )
            + "</tr>"
        )
    rows = "".join(row_parts)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Run Index</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; color: #222; background: #f4efe8; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 0.45rem 0.55rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    code {{ background: #f4f4f4; padding: 0.08rem 0.25rem; }}
  </style>
</head>
<body>
  <h1>Run Index</h1>
  <p>Catalog of run reference pages across <code>{html.escape(", ".join(str(root) for root in runs_roots))}</code>.</p>
  <table>
    <tr><th>Timestamp</th><th>Type</th><th>Run</th><th>Has page</th><th>Media</th><th>Bodies</th><th>ROM</th><th>Seams</th><th>Reference</th></tr>
    {rows or '<tr><td colspan="9">No runs found.</td></tr>'}
  </table>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        action="append",
        required=True,
        help="Directory containing run subdirectories. Repeat to aggregate multiple run roots.",
    )
    parser.add_argument("--out", type=Path, required=True, help="HTML output path for the index.")
    args = parser.parse_args(argv)
    html_path = render_run_index([path.resolve() for path in args.runs_root], out_path=args.out.resolve())
    print(html_path)


if __name__ == "__main__":
    main()
