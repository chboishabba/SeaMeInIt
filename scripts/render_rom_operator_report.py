#!/usr/bin/env python3
"""Render a static HTML report for operator-level ROM artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

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


def _render_bar_chart(values: Sequence[float], labels: Sequence[str], title: str, output: Path) -> Path:
    width = 960
    height = 420
    margin = 50
    img = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(img)
    draw.text((margin, 12), title, fill=(20, 20, 20))
    if not values:
        draw.text((margin, 70), "No data available", fill=(90, 90, 90))
        img.save(output)
        return output

    max_value = max(float(v) for v in values)
    usable_width = width - 2 * margin
    usable_height = height - 2 * margin - 40
    count = max(1, len(values))
    bar_gap = 8
    bar_width = max(8, int((usable_width - bar_gap * (count - 1)) / count))
    for idx, raw in enumerate(values):
        value = float(raw)
        norm = 0.0 if max_value <= 1e-12 else value / max_value
        x0 = margin + idx * (bar_width + bar_gap)
        x1 = x0 + bar_width
        y1 = height - margin - 20
        y0 = y1 - int(norm * usable_height)
        draw.rectangle((x0, y0, x1, y1), fill=(66, 135, 245), outline=(40, 70, 120))
        draw.text((x0, y1 + 4), str(labels[idx]), fill=(40, 40, 40))
        draw.text((x0, max(24, y0 - 16)), f"{value:.3g}", fill=(40, 40, 40))
    img.save(output)
    return output


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
    parser.add_argument("--envelope", type=Path, default=None)
    parser.add_argument("--certificate", type=Path, default=None)
    parser.add_argument("--costs", type=Path, default=None)
    parser.add_argument("--body", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    basis = load_basis(args.basis)
    meta_payload = _load_json(args.rom_meta) or {}
    coeff_payload = _load_coeff_samples(args.coeff_samples)
    envelope_payload = _load_json(args.envelope)
    certificate_payload = _load_json(args.certificate)
    costs_payload = load_seam_cost_field(args.costs) if args.costs is not None and args.costs.exists() else None
    body_vertex_count = _load_body_vertex_count(args.body)

    vertex_count = int(basis.metadata.vertex_count)
    topology = _safe_topology_label(vertex_count)
    coeff_summary = _build_coeff_summary(coeff_payload)
    cost_vertex_count = int(len(costs_payload.vertex_costs)) if costs_payload is not None else None
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
    if coeff_summary["available"]:
        top = coeff_summary["top_variance_components"]
        top_plot = out_dir / "coeff_top_variance.png"
        _render_bar_chart(
            [entry["variance"] for entry in top],
            [f"k{entry['index']}" for entry in top],
            "Top coefficient variance components",
            top_plot,
        )
        generated_files["coeff_top_variance_png"] = str(top_plot)

        norms = coeff_summary["sample_norms"][: min(12, len(coeff_summary["sample_norms"]))]
        norm_plot = out_dir / "coeff_sample_norms.png"
        _render_bar_chart(
            [entry["norm"] for entry in norms],
            [entry["pose_id"] for entry in norms],
            "Sample coefficient norms",
            norm_plot,
        )
        generated_files["coeff_sample_norms_png"] = str(norm_plot)

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
        "generated_files": generated_files,
        "warnings": warnings_list,
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
    image_html = "".join(
        f"<figure><img src=\"{Path(path).name}\" alt=\"{name}\" style=\"max-width:100%;border:1px solid #ccc;\"><figcaption>{name}</figcaption></figure>"
        for name, path in generated_files.items()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ROM Operator Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
    th {{ background: #f5f5f5; width: 18rem; }}
    code {{ background: #f5f5f5; padding: 0.1rem 0.25rem; }}
    .callout {{ border-left: 4px solid #4b7bec; padding: 0.75rem 1rem; background: #f7fbff; }}
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
  <h2>Generated Visuals</h2>
  {image_html or '<p>No visuals generated.</p>'}
  <h2>Source Artifacts</h2>
  <table>
    <tr><th>Path</th><th>Level</th><th>Topology</th><th>Domain</th></tr>
    {''.join(f"<tr><td><code>{entry['path']}</code></td><td>{entry['artifact_level']}</td><td>{entry['topology']}</td><td>{entry['domain']}</td></tr>" for entry in artifact_entries)}
  </table>
</body>
</html>
"""
    html_path = out_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"Wrote ROM operator report to {html_path}")


if __name__ == "__main__":
    main()
