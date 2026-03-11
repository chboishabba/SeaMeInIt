#!/usr/bin/env python3
"""Render a static HTML report for operator-level ROM artifacts."""

from __future__ import annotations

import argparse
import html
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from smii.rom import load_basis, load_seam_cost_field


def _sha256_path(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path | None) -> Mapping[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected JSON object at {path}.")
    return payload


def _load_coeff_samples(path: Path | None) -> Mapping[str, Any] | None:
    payload = _load_json(path)
    if payload is None:
        return None
    samples = payload.get("samples")
    if not isinstance(samples, Sequence):
        raise TypeError("Coefficient samples payload must contain a 'samples' list.")
    return payload


def _load_sample_manifest(path: Path | None) -> Mapping[str, Any] | None:
    payload = _load_json(path)
    if payload is None:
        return None
    samples = payload.get("samples")
    if not isinstance(samples, Sequence):
        raise TypeError("ROM sample manifest must contain a 'samples' list.")
    return payload


def _load_body_vertex_count(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.lib.npyio.NpzFile):
        if "vertices" in payload:
            vertices = payload["vertices"]
        elif "v" in payload:
            vertices = payload["v"]
        else:
            return None
    else:
        vertices = payload
    arr = np.asarray(vertices, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return int(arr.shape[0])


def _safe_topology_label(vertex_count: int | None) -> str:
    return f"v{int(vertex_count)}" if vertex_count is not None else "none"


def _inline_bar_chart_svg(values: Sequence[float], labels: Sequence[str], title: str) -> str:
    width = 920
    height = 320
    margin_left = 48
    margin_bottom = 54
    margin_top = 26
    count = max(1, len(values))
    usable_width = width - margin_left - 24
    usable_height = height - margin_top - margin_bottom
    max_value = max((float(v) for v in values), default=0.0)
    gap = 8
    bar_width = max(12, int((usable_width - gap * max(count - 1, 0)) / count))
    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html.escape(title)}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#fbfaf5" rx="12" />',
        f'<text x="{margin_left}" y="18" fill="#222" font-size="15" font-weight="700">{html.escape(title)}</text>',
    ]
    if not values:
        svg_parts.append(
            f'<text x="{margin_left}" y="{height // 2}" fill="#666" font-size="14">No data available</text>'
        )
        svg_parts.append("</svg>")
        return "".join(svg_parts)

    base_y = height - margin_bottom
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{base_y}" x2="{width - 16}" y2="{base_y}" stroke="#7d7d7d" stroke-width="1.5" />'
    )
    for idx, raw in enumerate(values):
        value = float(raw)
        norm = 0.0 if max_value <= 1e-12 else value / max_value
        x = margin_left + idx * (bar_width + gap)
        bar_height = max(2.0, norm * usable_height)
        y = base_y - bar_height
        label = html.escape(str(labels[idx]))
        value_label = html.escape(f"{value:.3g}")
        svg_parts.append(
            f'<rect x="{x}" y="{y:.2f}" width="{bar_width}" height="{bar_height:.2f}" '
            'fill="#2f6db3" stroke="#1a3f6b" stroke-width="1" rx="3" />'
        )
        svg_parts.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{max(32.0, y - 6):.2f}" text-anchor="middle" '
            f'fill="#333" font-size="11">{value_label}</text>'
        )
        svg_parts.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{base_y + 16}" text-anchor="middle" '
            f'fill="#444" font-size="10">{label}</text>'
        )
    svg_parts.append("</svg>")
    return "".join(svg_parts)


def _iter_media_paths(paths: Sequence[Path]) -> list[Path]:
    supported = {
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
    seen: set[Path] = set()
    media: list[Path] = []
    for candidate in paths:
        path = Path(candidate)
        if not path.exists():
            continue
        if path.is_dir():
            iterator = sorted(
                item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in supported
            )
        elif path.suffix.lower() in supported:
            iterator = [path]
        else:
            iterator = []
        for item in iterator:
            resolved = item.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            media.append(item)
    return media


def _media_manifest_entry(path: Path, *, out_dir: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    kind = "video" if suffix in {".webm", ".mp4", ".mov", ".m4v"} else "image"
    mime_type = mimetypes.guess_type(str(path))[0] or ("video/webm" if kind == "video" else "image/png")
    try:
        relative_path = os.path.relpath(path, out_dir)
    except ValueError:
        relative_path = str(path)
    return {
        "path": str(path),
        "relative_path": relative_path,
        "kind": kind,
        "mime_type": mime_type,
        "group": path.parent.name,
        "name": path.name,
        "sha256": _sha256_path(path),
    }


def _embedded_media_html(entries: Sequence[Mapping[str, Any]]) -> str:
    if not entries:
        return "<p>No external media embedded.</p>"
    cards: list[str] = []
    for entry in entries:
        rel = html.escape(str(entry["relative_path"]))
        name = html.escape(str(entry["name"]))
        group = html.escape(str(entry["group"]))
        if entry["kind"] == "video":
            media_tag = (
                f'<video controls preload="metadata" style="width:100%;background:#111;border-radius:10px;">'
                f'<source src="{rel}" type="{html.escape(str(entry["mime_type"]))}"></video>'
            )
        else:
            media_tag = (
                f'<img src="{rel}" alt="{name}" '
                'style="width:100%;border-radius:10px;border:1px solid #d8d2c8;background:#fff;" />'
            )
        cards.append(
            "<figure class=\"media-card\">"
            f"<div class=\"media-group\">{group}</div>"
            f"{media_tag}"
            f"<figcaption><code>{name}</code></figcaption>"
            "</figure>"
        )
    return f'<div class="media-grid">{"".join(cards)}</div>'


def _build_coeff_summary(coeff_payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if coeff_payload is None:
        return {
            "available": False,
            "field_name": None,
            "sample_count": 0,
            "component_count": 0,
            "top_variance_components": [],
            "sample_norms": [],
        }

    samples = coeff_payload["samples"]
    if not samples:
        return {
            "available": True,
            "field_name": "seam_sensitivity",
            "sample_count": 0,
            "component_count": 0,
            "top_variance_components": [],
            "sample_norms": [],
        }

    field_name = next(iter(samples[0]["coeffs"].keys()))
    matrix = np.asarray(
        [entry["coeffs"][field_name] for entry in samples],
        dtype=float,
    )
    variance = np.var(matrix, axis=0) if matrix.size else np.zeros(0, dtype=float)
    mean_abs = np.mean(np.abs(matrix), axis=0) if matrix.size else np.zeros(0, dtype=float)
    order = np.argsort(variance)[::-1]
    top = [
        {
            "index": int(idx),
            "variance": float(variance[idx]),
            "mean_abs": float(mean_abs[idx]),
        }
        for idx in order[: min(8, len(order))]
    ]
    sample_norms = [
        {
            "pose_id": str(entry.get("pose_id", f"sample_{idx}")),
            "weight": float(entry.get("weight", 1.0)),
            "norm": float(np.linalg.norm(matrix[idx])),
        }
        for idx, entry in enumerate(samples)
    ]
    return {
        "available": True,
        "field_name": field_name,
        "sample_count": int(matrix.shape[0]),
        "component_count": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "top_variance_components": top,
        "sample_norms": sample_norms,
    }


def _artifact_entry(
    *,
    path: Path | None,
    artifact_level: str,
    role: str = "none",
    topology: str = "none",
    domain: str = "operator",
) -> Mapping[str, Any] | None:
    if path is None:
        return None
    return {
        "path": str(path),
        "sha256": _sha256_path(path),
        "artifact_level": artifact_level,
        "role": role,
        "topology": topology,
        "domain": domain,
    }


def _html_table(rows: Sequence[tuple[str, str]]) -> str:
    return "".join(
        f"<tr><th>{key}</th><td>{value}</td></tr>"
        for key, value in rows
    )


def _consistency_summary(
    *,
    basis_vertex_count: int,
    basis_component_count: int,
    meta_payload: Mapping[str, Any],
    coeff_summary: Mapping[str, Any],
    body_vertex_count: int | None,
    cost_vertex_count: int | None,
) -> tuple[str, list[str]]:
    flags: list[str] = []
    meta = meta_payload.get("meta")
    meta_vertex_count = None
    if isinstance(meta, Mapping) and meta.get("vertex_count") is not None:
        meta_vertex_count = int(meta["vertex_count"])
        if meta_vertex_count != basis_vertex_count:
            flags.append(
                f"ROM meta vertex count ({meta_vertex_count}) does not match basis vertex count ({basis_vertex_count})."
            )
    if body_vertex_count is not None and body_vertex_count != basis_vertex_count:
        flags.append(
            f"Body vertex count ({body_vertex_count}) does not match basis vertex count ({basis_vertex_count})."
        )
    if cost_vertex_count is not None and cost_vertex_count != basis_vertex_count:
        flags.append(
            f"Cost-field vertex count ({cost_vertex_count}) does not match basis vertex count ({basis_vertex_count})."
        )
    coeff_components = int(coeff_summary.get("component_count", 0))
    if coeff_summary.get("available") and coeff_components not in {0, basis_component_count}:
        flags.append(
            f"Coefficient component count ({coeff_components}) does not match basis component count ({basis_component_count})."
        )
    return ("WARN" if flags else "PASS"), flags


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--rom-meta", type=Path, required=True)
    parser.add_argument("--coeff-samples", type=Path, default=None)
    parser.add_argument("--sample-manifest", type=Path, default=None)
    parser.add_argument("--envelope", type=Path, default=None)
    parser.add_argument("--certificate", type=Path, default=None)
    parser.add_argument("--costs", type=Path, default=None)
    parser.add_argument("--body", type=Path, default=None)
    parser.add_argument(
        "--media-path",
        dest="media_paths",
        type=Path,
        action="append",
        default=[],
        help="Optional file or directory of existing image/video artifacts to embed in the report.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    basis = load_basis(args.basis)
    meta_payload = _load_json(args.rom_meta) or {}
    coeff_payload = _load_coeff_samples(args.coeff_samples)
    sample_manifest = _load_sample_manifest(args.sample_manifest)
    envelope_payload = _load_json(args.envelope)
    certificate_payload = _load_json(args.certificate)
    costs_payload = load_seam_cost_field(args.costs) if args.costs is not None and args.costs.exists() else None
    body_vertex_count = _load_body_vertex_count(args.body)

    vertex_count = int(basis.metadata.vertex_count)
    topology = _safe_topology_label(vertex_count)
    coeff_summary = _build_coeff_summary(coeff_payload)
    cost_vertex_count = int(len(costs_payload.vertex_costs)) if costs_payload is not None else None
    embedded_media = [
        _media_manifest_entry(path, out_dir=out_dir)
        for path in _iter_media_paths(args.media_paths)
    ]
    consistency_status, consistency_flags = _consistency_summary(
        basis_vertex_count=vertex_count,
        basis_component_count=int(basis.metadata.component_count),
        meta_payload=meta_payload,
        coeff_summary=coeff_summary,
        body_vertex_count=body_vertex_count,
        cost_vertex_count=cost_vertex_count,
    )
    coeff_summary_path = out_dir / "coeff_summary.json"
    coeff_summary_path.write_text(json.dumps(coeff_summary, indent=2), encoding="utf-8")

    generated_files: dict[str, str] = {}

    warnings_list = [
        "Operator-level ROM artifacts describe basis/coefficient/schedule behavior.",
        "Mesh renders and seam reports are topology-level projections, not the pure ROM operator.",
    ]
    if coeff_payload is None:
        warnings_list.append("Coefficient samples were not provided; coefficient-space sections are partial.")
    warnings_list.extend(consistency_flags)

    artifact_entries = [
        _artifact_entry(path=args.basis, artifact_level="operator"),
        _artifact_entry(path=args.rom_meta, artifact_level="operator"),
        _artifact_entry(path=args.coeff_samples, artifact_level="operator"),
        _artifact_entry(
            path=args.sample_manifest,
            artifact_level="topology",
            role="none",
            topology=_safe_topology_label(
                int(sample_manifest.get("meta", {}).get("source_vertex_count"))
                if sample_manifest
                and isinstance(sample_manifest.get("meta"), Mapping)
                and sample_manifest.get("meta", {}).get("source_vertex_count") is not None
                else None
            ),
            domain="operator_native_pose",
        ),
        _artifact_entry(path=args.envelope, artifact_level="operator"),
        _artifact_entry(path=args.certificate, artifact_level="operator"),
        _artifact_entry(
            path=args.costs,
            artifact_level="topology",
            role="none",
            topology=_safe_topology_label(cost_vertex_count),
            domain="native",
        ),
        _artifact_entry(
            path=args.body,
            artifact_level="topology",
            role="none",
            topology=_safe_topology_label(body_vertex_count),
            domain="native",
        ),
    ]
    artifact_entries = [entry for entry in artifact_entries if entry is not None]

    report_manifest = {
        "summary": {
            "consistency_status": consistency_status,
            "consistency_flags": consistency_flags,
            "basis_vertex_count": vertex_count,
            "basis_component_count": int(basis.metadata.component_count),
            "basis_normalization": basis.metadata.normalization,
            "coefficient_samples_available": bool(coeff_summary["available"]),
            "coefficient_sample_count": int(coeff_summary["sample_count"]),
            "certificate_status": certificate_payload.get("status") if certificate_payload else None,
            "rank_correlation": certificate_payload.get("rank_correlation") if certificate_payload else None,
            "meta_vertex_count": int(meta_payload.get("meta", {}).get("vertex_count")) if isinstance(meta_payload.get("meta"), Mapping) and meta_payload.get("meta", {}).get("vertex_count") is not None else None,
            "cost_vertex_count": cost_vertex_count,
            "body_vertex_count": body_vertex_count,
        },
        "artifacts": artifact_entries,
        "embedded_media": embedded_media,
        "generated_files": generated_files,
        "warnings": warnings_list,
        "sample_morphology": sample_manifest,
    }
    report_manifest_path = out_dir / "report_manifest.json"
    report_manifest_path.write_text(json.dumps(report_manifest, indent=2), encoding="utf-8")

    operator_rows = [
        ("Consistency status", consistency_status),
        ("Basis vertices", str(vertex_count)),
        ("Basis components", str(basis.metadata.component_count)),
        ("Basis normalization", str(basis.metadata.normalization or "unknown")),
        ("Coeff samples", str(coeff_summary["sample_count"])),
        ("Certificate status", str(certificate_payload.get("status")) if certificate_payload else "n/a"),
        ("Rank correlation", f"{certificate_payload.get('rank_correlation'):.4f}" if certificate_payload and certificate_payload.get("rank_correlation") is not None else "n/a"),
    ]
    topology_rows = [
        ("Body path", str(args.body) if args.body else "n/a"),
        ("Costs path", str(args.costs) if args.costs else "n/a"),
        ("Basis topology", topology),
        ("Body topology", _safe_topology_label(body_vertex_count)),
        ("Cost topology", _safe_topology_label(cost_vertex_count)),
        ("ROM meta vertex count", str(meta_payload.get("meta", {}).get("vertex_count")) if isinstance(meta_payload.get("meta"), Mapping) and meta_payload.get("meta", {}).get("vertex_count") is not None else "n/a"),
    ]
    top_components_html = "".join(
        f"<tr><td>k{entry['index']}</td><td>{entry['variance']:.6g}</td><td>{entry['mean_abs']:.6g}</td></tr>"
        for entry in coeff_summary["top_variance_components"]
    )
    warning_html = "".join(f"<li>{message}</li>" for message in warnings_list)
    top_chart_html = _inline_bar_chart_svg(
        [entry["variance"] for entry in coeff_summary["top_variance_components"]],
        [f"k{entry['index']}" for entry in coeff_summary["top_variance_components"]],
        "Top coefficient variance components",
    )
    norm_chart_html = _inline_bar_chart_svg(
        [entry["norm"] for entry in coeff_summary["sample_norms"][: min(12, len(coeff_summary["sample_norms"]))]],
        [entry["pose_id"] for entry in coeff_summary["sample_norms"][: min(12, len(coeff_summary["sample_norms"]))]],
        "Sample coefficient norms",
    )
    media_html = _embedded_media_html(embedded_media)
    sample_rows = ""
    if sample_manifest is not None:
        sample_rows = "".join(
            "<tr>"
            f"<td><code>{html.escape(str(entry.get('pose_id', 'unknown')))}</code></td>"
            f"<td>{html.escape(', '.join(str(item) for item in entry.get('selection_reasons', [])))}</td>"
            f"<td>{float(entry.get('field_l2_norm', 0.0)):.6g}</td>"
            f"<td>{float(entry.get('displacement_mean_norm', 0.0)):.6g}</td>"
            f"<td><a href=\"{html.escape(os.path.relpath(str(entry.get('mesh_path')), out_dir)) if entry.get('mesh_path') else ''}\"><code>{html.escape(str(entry.get('mesh_name', 'mesh')))}</code></a></td>"
            "</tr>"
            for entry in sample_manifest.get("samples", [])
            if isinstance(entry, Mapping)
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ROM Operator Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; color: #222; background: #f3efe7; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
    th {{ background: #f5f5f5; width: 18rem; }}
    code {{ background: #f5f5f5; padding: 0.1rem 0.25rem; }}
    .callout {{ border-left: 4px solid #4b7bec; padding: 0.75rem 1rem; background: #f7fbff; }}
    .chart {{ width: 100%; height: auto; margin-bottom: 1rem; }}
    .media-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }}
    .media-card {{ margin: 0; padding: 0.75rem; background: #fffaf1; border: 1px solid #d9cfbe; border-radius: 14px; }}
    .media-group {{ font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; color: #74624d; margin-bottom: 0.5rem; }}
  </style>
</head>
<body>
  <h1>ROM Operator Report</h1>
  <p class="callout">This report is the operator-level inspection surface for ROM artifacts. It separates basis/coefficient/certificate outputs from topology-level mesh and seam projections.</p>
  <h2>Operator Summary</h2>
  <table>{_html_table(operator_rows)}</table>
  <h2>Topology Projection Summary</h2>
  <table>{_html_table(topology_rows)}</table>
  <h2>Warnings</h2>
  <ul>{warning_html}</ul>
  <h2>Top Coefficient Components</h2>
  <table><tr><th>Component</th><th>Variance</th><th>Mean abs</th></tr>{top_components_html or '<tr><td colspan="3">No coefficient data available.</td></tr>'}</table>
  <h2>Operator Charts</h2>
  {top_chart_html}
  {norm_chart_html}
  <h2>Representative ROM Sample Morphologies</h2>
  <p class="callout">These are sampler-native posed/deformed meshes chosen to show where morphology changes, including flailing, actually appear. They are not inverse-mapped back to the fitted body topology.</p>
  <table><tr><th>Pose</th><th>Selection reasons</th><th>Field L2</th><th>Disp mean</th><th>Mesh</th></tr>{sample_rows or '<tr><td colspan="5">No representative ROM sample manifest provided.</td></tr>'}</table>
  <h2>Embedded Media</h2>
  {media_html}
  <h2>Source Artifacts</h2>
  <table>
    <tr><th>Path</th><th>Level</th><th>Topology</th><th>Domain</th></tr>
    {''.join(f"<tr><td><code>{entry['path']}</code></td><td>{entry['artifact_level']}</td><td>{entry['topology']}</td><td>{entry['domain']}</td></tr>" for entry in artifact_entries)}
  </table>
</body>
</html>
"""
    html_path = out_dir / "index.html"
    html_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote ROM operator report to {html_path}")


if __name__ == "__main__":
    main()
