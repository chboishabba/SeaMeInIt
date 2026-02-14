#!/usr/bin/env python3
"""Audit topology/provenance across explicit pipeline stages.

Goal: make vertex/face count divergence and seam nonconformance obvious and
machine-readable so we stop arguing from filenames and screenshots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
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


def _sha256_array(arr: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(arr).view(np.uint8))
    return digest.hexdigest()


def _mesh_edge_set(faces: np.ndarray) -> set[tuple[int, int]]:
    edge_set: set[tuple[int, int]] = set()
    for a, b, c in np.asarray(faces, dtype=int):
        for u, v in ((a, b), (b, c), (c, a)):
            uu, vv = int(u), int(v)
            if uu == vv:
                continue
            edge_set.add((uu, vv) if uu < vv else (vv, uu))
    return edge_set


def _collect_seam_edges(report: dict[str, Any]) -> list[tuple[int, int]]:
    edge_set: set[tuple[int, int]] = set()
    panels = report.get("panels", {})
    if not isinstance(panels, dict):
        return []
    for panel in panels.values():
        if not isinstance(panel, dict):
            continue
        entries = panel.get("edges", [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            edge = entry.get("edge")
            if not isinstance(edge, list) or len(edge) != 2:
                continue
            a, b = int(edge[0]), int(edge[1])
            if a == b:
                continue
            edge_set.add((a, b) if a < b else (b, a))
    return sorted(edge_set)


def _edge_conformance_metrics(
    seam_edges: list[tuple[int, int]], mesh_edges: set[tuple[int, int]]
) -> dict[str, object]:
    unique = {(a, b) if a < b else (b, a) for a, b in seam_edges}
    if not unique:
        return {
            "seam_unique_edge_count": 0,
            "mesh_edge_valid_count": 0,
            "mesh_edge_invalid_count": 0,
            "mesh_edge_valid_ratio": 1.0,
        }
    valid = sum(1 for edge in unique if edge in mesh_edges)
    invalid = int(len(unique) - valid)
    return {
        "seam_unique_edge_count": int(len(unique)),
        "mesh_edge_valid_count": int(valid),
        "mesh_edge_invalid_count": int(invalid),
        "mesh_edge_valid_ratio": float(valid / max(1, len(unique))),
    }


def _load_mesh_npz(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    payload = np.load(path)
    vertices = np.asarray(payload["vertices"], dtype=float)
    faces = np.asarray(payload["faces"], dtype=int) if "faces" in payload else None
    return vertices, faces


def _load_vertex_map_meta(path: Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    meta: dict[str, Any] = {}
    if "meta" in payload:
        try:
            raw = payload["meta"].item()
            if isinstance(raw, dict):
                meta = raw
        except Exception:
            meta = {}
    # Also include quick stats if arrays exist.
    for prefix in ("source_to_target", "target_to_source"):
        idx_key = f"{prefix}_indices"
        dist_key = f"{prefix}_distances"
        if idx_key in payload and dist_key in payload:
            indices = np.asarray(payload[idx_key])
            distances = np.asarray(payload[dist_key])
            if indices.ndim == 1 and distances.ndim == 1 and len(indices) == len(distances):
                meta.setdefault(prefix, {})
                meta[prefix].update(
                    {
                        "count": int(len(indices)),
                        "unique_targets": int(len(np.unique(indices))),
                        "collision_ratio": float(1.0 - (len(np.unique(indices)) / max(1, len(indices)))),
                        "max_distance": float(np.max(distances)) if len(distances) else 0.0,
                        "mean_distance": float(np.mean(distances)) if len(distances) else 0.0,
                    }
                )
    return meta


@dataclass(frozen=True)
class StageSpec:
    name: str
    path: Path


def _parse_stage(arg: str) -> StageSpec:
    if "=" not in arg:
        raise ValueError("stage must be NAME=PATH")
    name, path = arg.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("stage NAME must be non-empty")
    return StageSpec(name=name, path=Path(path))


def _collect_from_seam_report(path: Path) -> list[StageSpec]:
    report = json.loads(path.read_text(encoding="utf-8"))
    found: list[StageSpec] = []
    prov = report.get("provenance", {})
    if isinstance(prov, dict):
        body = prov.get("body_path")
        rom = prov.get("rom_costs_path")
        ref = prov.get("reference_body_path")
        if isinstance(body, str):
            found.append(StageSpec("provenance_body", Path(body)))
        if isinstance(rom, str):
            found.append(StageSpec("provenance_rom_costs", Path(rom)))
        if isinstance(ref, str) and ref:
            found.append(StageSpec("provenance_reference_body", Path(ref)))
    rep = report.get("reprojection", {})
    if isinstance(rep, dict):
        for key in ("source_mesh", "target_mesh", "vertex_map_file"):
            value = rep.get(key)
            if isinstance(value, str) and value:
                found.append(StageSpec(f"reprojection_{key}", Path(value)))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Explicit stage as NAME=PATH (repeatable).",
    )
    parser.add_argument(
        "--discover-from-seam-report",
        action="append",
        default=[],
        type=Path,
        help="Add implied stages from seam_report.json provenance/reprojection blocks.",
    )
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = args.out_json or Path("outputs/seams_run") / f"pipeline_audit_{ts}.json"
    out_csv = args.out_csv or out_json.with_suffix(".csv")

    stages: list[StageSpec] = []
    for raw in args.stage:
        stages.append(_parse_stage(raw))
    for seam_report in args.discover_from_seam_report:
        for spec in _collect_from_seam_report(seam_report):
            stages.append(spec)

    # Dedup by (name,path) preserving order.
    seen: set[tuple[str, str]] = set()
    dedup: list[StageSpec] = []
    for spec in stages:
        key = (spec.name, str(spec.path))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(spec)
    stages = dedup

    records: list[dict[str, Any]] = []
    meshes: dict[str, dict[str, Any]] = {}

    for spec in stages:
        path = spec.path
        suffix = path.suffix.lower()
        record: dict[str, Any] = {"stage": spec.name, "path": str(path), "exists": path.exists()}
        if not path.exists():
            records.append(record)
            continue

        record["sha256"] = _sha256_path(path)
        if suffix == ".npz":
            payload = np.load(path, allow_pickle=True)
            if "vertices" in payload:
                vertices = np.asarray(payload["vertices"], dtype=float)
                faces = np.asarray(payload["faces"], dtype=int) if "faces" in payload else None
                record.update(
                    {
                        "kind": "mesh_npz",
                        "vertex_count": int(vertices.shape[0]),
                        "face_count": int(faces.shape[0]) if faces is not None else None,
                        "vertices_sha256": _sha256_array(vertices),
                        "faces_sha256": _sha256_array(faces) if faces is not None else None,
                    }
                )
                meshes[spec.name] = {"vertices": vertices, "faces": faces, "record": record}
            elif (
                "source_to_target_indices" in payload
                or "target_to_source_indices" in payload
                or "source_to_target_distances" in payload
                or "target_to_source_distances" in payload
            ):
                record.update({"kind": "vertex_map_npz", "meta": _load_vertex_map_meta(path)})
            elif "vertex_costs" in payload or "costs" in payload:
                key = "vertex_costs" if "vertex_costs" in payload else "costs"
                costs = np.asarray(payload[key], dtype=float).reshape(-1)
                record.update(
                    {
                        "kind": "rom_costs_npz",
                        "vertex_cost_count": int(costs.shape[0]),
                    }
                )
            else:
                record.update({"kind": "npz_unknown"})
        elif suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and "panels" in obj and "solver" in obj:
                record["kind"] = "seam_report"
                seam_edges = _collect_seam_edges(obj)
                record["seam_unique_edge_count"] = int(
                    len({(a, b) if a < b else (b, a) for a, b in seam_edges})
                )
                record["seam_edge_index_max"] = (
                    int(max(max(a, b) for a, b in seam_edges)) if seam_edges else None
                )

                prov = obj.get("provenance", {})
                body_path = None
                if isinstance(prov, dict) and isinstance(prov.get("body_path"), str):
                    body_path = Path(prov["body_path"])
                # If we already saw the body mesh stage, validate conformance.
                mesh_edges = None
                mesh_name = None
                if body_path is not None:
                    for name, entry in meshes.items():
                        if Path(entry["record"]["path"]) == body_path:
                            faces = entry["faces"]
                            if faces is not None:
                                mesh_edges = _mesh_edge_set(faces)
                                mesh_name = name
                            break
                if mesh_edges is not None:
                    record["edge_conformance"] = _edge_conformance_metrics(seam_edges, mesh_edges)
                    record["edge_conformance_mesh_stage"] = mesh_name
            else:
                record["kind"] = "json_unknown"
        else:
            record["kind"] = "unknown"
        records.append(record)

    checks: list[dict[str, Any]] = []
    mesh_names = [name for name, entry in meshes.items() if entry["record"].get("kind") == "mesh_npz"]
    for i, a in enumerate(mesh_names):
        for b in mesh_names[i + 1 :]:
            ra = meshes[a]["record"]
            rb = meshes[b]["record"]
            checks.append(
                {
                    "name": f"mesh_pair::{a}__vs__{b}",
                    "a": a,
                    "b": b,
                    "vertex_count_equal": ra.get("vertex_count") == rb.get("vertex_count"),
                    "face_count_equal": ra.get("face_count") == rb.get("face_count"),
                    "faces_sha256_equal": (ra.get("faces_sha256") is not None and ra.get("faces_sha256") == rb.get("faces_sha256")),
                    "vertices_sha256_equal": ra.get("vertices_sha256") == rb.get("vertices_sha256"),
                }
            )

    payload = {"timestamp_utc": ts, "records": records, "checks": checks}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Flat CSV for quick diffing.
    rows: list[dict[str, Any]] = []
    for r in records:
        rows.append(
            {
                "kind": r.get("kind", ""),
                "stage": r.get("stage", ""),
                "path": r.get("path", ""),
                "exists": r.get("exists", ""),
                "vertex_count": r.get("vertex_count", r.get("vertex_cost_count", "")),
                "face_count": r.get("face_count", ""),
                "seam_edges": r.get("seam_unique_edge_count", ""),
                "mesh_edge_valid_ratio": (
                    r.get("edge_conformance", {}).get("mesh_edge_valid_ratio")
                    if isinstance(r.get("edge_conformance"), dict)
                    else ""
                ),
                "sha256": r.get("sha256", ""),
            }
        )
    for c in checks:
        rows.append(
            {
                "kind": "check",
                "stage": c.get("name", ""),
                "path": "",
                "exists": "",
                "vertex_count": "",
                "face_count": "",
                "seam_edges": "",
                "mesh_edge_valid_ratio": "",
                "sha256": json.dumps(c, sort_keys=True),
            }
        )

    with out_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "kind",
                "stage",
                "path",
                "exists",
                "vertex_count",
                "face_count",
                "seam_edges",
                "mesh_edge_valid_ratio",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(out_json)


if __name__ == "__main__":
    main()
