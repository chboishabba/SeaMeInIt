#!/usr/bin/env python3
"""Audit mesh/cost/seam lineage and topology compatibility across pipeline stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _collect_body(path: Path) -> dict[str, Any]:
    payload = np.load(path)
    vertices = np.asarray(payload["vertices"], dtype=float)
    faces = np.asarray(payload["faces"], dtype=int) if "faces" in payload else None
    record: dict[str, Any] = {
        "stage": "base_body",
        "path": str(path),
        "file_sha256": _sha256_path(path),
        "vertex_count": int(vertices.shape[0]),
        "vertex_sha256": _sha256_array(vertices),
    }
    if faces is not None:
        record["face_count"] = int(faces.shape[0])
        record["face_sha256"] = _sha256_array(faces)
    return record


def _collect_basis(path: Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    basis = np.asarray(payload["basis"], dtype=float)
    vertices = np.asarray(payload["vertices"], dtype=float) if "vertices" in payload else None
    meta = payload["meta"].item() if "meta" in payload else {}
    return {
        "stage": "canonical_basis",
        "path": str(path),
        "file_sha256": _sha256_path(path),
        "basis_shape": [int(v) for v in basis.shape],
        "vertex_count": int(vertices.shape[0]) if vertices is not None else None,
        "meta": meta,
    }


def _collect_rom_costs(path: Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    vertex_costs = np.asarray(payload["vertex_costs"], dtype=float)
    edges = np.asarray(payload["edges"], dtype=int) if "edges" in payload else np.empty((0, 2), dtype=int)
    metadata: dict[str, Any] = {}
    if "metadata" in payload:
        try:
            raw = payload["metadata"].item()
            if isinstance(raw, dict):
                metadata = raw
        except Exception:
            metadata = {}
    return {
        "stage": "rom_costs",
        "path": str(path),
        "file_sha256": _sha256_path(path),
        "vertex_cost_count": int(vertex_costs.shape[0]),
        "edge_count": int(edges.shape[0]) if edges.ndim == 2 else 0,
        "metadata": metadata,
    }


def _collect_rom_meta(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    return {
        "stage": "rom_meta",
        "path": str(path),
        "file_sha256": _sha256_path(path),
        "meta": meta,
    }


def _collect_seam_report(path: Path, *, stage: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    edges: list[tuple[int, int]] = []
    panels = report.get("panels", {})
    if isinstance(panels, dict):
        for panel in panels.values():
            if not isinstance(panel, dict):
                continue
            raw_edges = panel.get("edges", [])
            if not isinstance(raw_edges, list):
                continue
            for entry in raw_edges:
                if not isinstance(entry, dict):
                    continue
                edge = entry.get("edge")
                if not isinstance(edge, list) or len(edge) != 2:
                    continue
                edges.append((int(edge[0]), int(edge[1])))
    max_index = max(max(a, b) for a, b in edges) if edges else None
    min_index = min(min(a, b) for a, b in edges) if edges else None
    return {
        "stage": stage,
        "path": str(path),
        "file_sha256": _sha256_path(path),
        "solver": report.get("solver"),
        "total_cost": report.get("total_cost"),
        "edge_count": int(len(edges)),
        "edge_index_min": int(min_index) if min_index is not None else None,
        "edge_index_max": int(max_index) if max_index is not None else None,
        "warning_count": int(len(report.get("warnings", []))) if isinstance(report.get("warnings"), list) else 0,
        "provenance": report.get("provenance"),
        "reprojection": report.get("reprojection"),
    }


def _build_checks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_stage = {record["stage"]: record for record in records}
    checks: list[dict[str, Any]] = []

    body = by_stage.get("base_body")
    rom_costs = by_stage.get("rom_costs")
    if body and rom_costs:
        body_vertices = int(body["vertex_count"])
        cost_vertices = int(rom_costs["vertex_cost_count"])
        checks.append(
            {
                "name": "body_vs_rom_vertex_count",
                "ok": body_vertices == cost_vertices,
                "body_vertex_count": body_vertices,
                "rom_vertex_count": cost_vertices,
            }
        )

    if body:
        body_vertices = int(body["vertex_count"])
        for key in ("seam_report_ogre", "seam_report_reprojected"):
            seam = by_stage.get(key)
            if not seam:
                continue
            max_idx = seam.get("edge_index_max")
            checks.append(
                {
                    "name": f"{key}_indices_within_body",
                    "ok": max_idx is None or int(max_idx) < body_vertices,
                    "body_vertex_count": body_vertices,
                    "edge_index_max": max_idx,
                }
            )
    return checks


def _write_csv(path: Path, records: list[dict[str, Any]], checks: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "kind": "record",
                "name": record.get("stage"),
                "path": record.get("path"),
                "vertex_count": record.get("vertex_count") or record.get("vertex_cost_count"),
                "edge_count": record.get("edge_count"),
                "edge_index_max": record.get("edge_index_max"),
                "ok": "",
                "details": "",
            }
        )
    for check in checks:
        rows.append(
            {
                "kind": "check",
                "name": check.get("name"),
                "path": "",
                "vertex_count": "",
                "edge_count": "",
                "edge_index_max": check.get("edge_index_max", ""),
                "ok": check.get("ok"),
                "details": json.dumps(check, sort_keys=True),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["kind", "name", "path", "vertex_count", "edge_count", "edge_index_max", "ok", "details"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", type=Path, required=True, help="Body mesh NPZ path (vertices/faces).")
    parser.add_argument("--basis", type=Path, default=None, help="Optional canonical basis NPZ.")
    parser.add_argument("--rom-costs", type=Path, required=True, help="ROM seam cost NPZ.")
    parser.add_argument("--rom-meta", type=Path, default=None, help="Optional ROM metadata JSON.")
    parser.add_argument("--seam-ogre", type=Path, default=None, help="Optional seam report from high-topology run.")
    parser.add_argument(
        "--seam-reprojected",
        type=Path,
        default=None,
        help="Optional seam report reprojected to base body topology.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to outputs/seams_run/lineage_audit_<timestamp>.json.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults alongside out-json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = args.out_json or Path("outputs/seams_run") / f"lineage_audit_{timestamp}.json"
    out_csv = args.out_csv or out_json.with_suffix(".csv")

    records: list[dict[str, Any]] = [_collect_body(args.body), _collect_rom_costs(args.rom_costs)]
    if args.basis is not None:
        records.append(_collect_basis(args.basis))
    if args.rom_meta is not None:
        records.append(_collect_rom_meta(args.rom_meta))
    if args.seam_ogre is not None:
        records.append(_collect_seam_report(args.seam_ogre, stage="seam_report_ogre"))
    if args.seam_reprojected is not None:
        records.append(_collect_seam_report(args.seam_reprojected, stage="seam_report_reprojected"))

    checks = _build_checks(records)
    payload = {
        "timestamp_utc": timestamp,
        "records": records,
        "checks": checks,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(out_csv, records, checks)
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
