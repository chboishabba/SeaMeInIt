#!/usr/bin/env python3
"""Compute A/B seam-transfer comparison metrics from a Strategy 2 bundle.

Given a bundle produced by `scripts/protocol_strategy2_bundle.py`, this script:
- loads native and reprojected seam reports for base+ROM domains,
- computes graph/length/conformance stats per report,
- surfaces reprojection quality fields when present,
- writes a deterministic JSON summary into bundle/manifests/.

This is intended to make "Strategy A vs Strategy B" a controlled experiment
instead of a visual/qualitative debate.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mesh_edge_set(faces: np.ndarray) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for tri in np.asarray(faces, dtype=int):
        if len(tri) != 3:
            continue
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        for u, v in ((a, b), (b, c), (c, a)):
            if u == v:
                continue
            if u < v:
                edges.add((u, v))
            else:
                edges.add((v, u))
    return edges


def _collect_edges(report: dict[str, Any]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    panels = report.get("panels") or {}
    if isinstance(panels, dict):
        for panel in panels.values():
            if not isinstance(panel, dict):
                continue
            for entry in panel.get("edges", []) or []:
                if isinstance(entry, dict) and "edge" in entry:
                    e = entry["edge"]
                else:
                    e = entry
                if (
                    isinstance(e, (list, tuple))
                    and len(e) == 2
                    and isinstance(e[0], (int, np.integer))
                    and isinstance(e[1], (int, np.integer))
                ):
                    a, b = int(e[0]), int(e[1])
                    if a == b:
                        continue
                    out.append((a, b))
    return out


def _unique_edges(edges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    uniq: set[tuple[int, int]] = set()
    for a, b in edges:
        uniq.add((a, b) if a < b else (b, a))
    return sorted(uniq)


@dataclass(frozen=True)
class GraphStats:
    unique_edge_count: int
    unique_vertex_count: int
    component_count: int
    cycle_component_count: int
    open_component_count: int
    max_degree: int
    junction_vertex_count: int


def _graph_stats(edges: list[tuple[int, int]]) -> GraphStats:
    if not edges:
        return GraphStats(
            unique_edge_count=0,
            unique_vertex_count=0,
            component_count=0,
            cycle_component_count=0,
            open_component_count=0,
            max_degree=0,
            junction_vertex_count=0,
        )

    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    degrees = {v: len(nbrs) for v, nbrs in adj.items()}
    max_degree = max(degrees.values()) if degrees else 0
    junction_vertex_count = sum(1 for d in degrees.values() if d > 2)

    visited: set[int] = set()
    component_count = 0
    cycle_component_count = 0
    open_component_count = 0

    for start in adj:
        if start in visited:
            continue
        component_count += 1
        stack = [start]
        visited.add(start)
        vertices: list[int] = []
        while stack:
            v = stack.pop()
            vertices.append(v)
            for n in adj.get(v, ()):
                if n not in visited:
                    visited.add(n)
                    stack.append(n)

        # Simple cycle in graph sense: all degrees == 2 in this component.
        if vertices and all(degrees.get(v, 0) == 2 for v in vertices):
            cycle_component_count += 1
        else:
            open_component_count += 1

    return GraphStats(
        unique_edge_count=len(edges),
        unique_vertex_count=len(adj),
        component_count=component_count,
        cycle_component_count=cycle_component_count,
        open_component_count=open_component_count,
        max_degree=max_degree,
        junction_vertex_count=junction_vertex_count,
    )


def _seam_length(edges: list[tuple[int, int]], vertices: np.ndarray) -> float:
    if not edges:
        return 0.0
    v = np.asarray(vertices, dtype=float)
    total = 0.0
    for a, b in edges:
        if a < 0 or b < 0 or a >= v.shape[0] or b >= v.shape[0]:
            continue
        total += float(np.linalg.norm(v[a] - v[b]))
    return total


def _edge_conformance(edges: list[tuple[int, int]], mesh_edges: set[tuple[int, int]]) -> dict[str, Any]:
    if not edges:
        return {"mesh_edge_valid_ratio": None, "mesh_edge_valid_edges": 0, "mesh_edge_total_edges": 0}
    if not mesh_edges:
        return {"mesh_edge_valid_ratio": None, "mesh_edge_valid_edges": None, "mesh_edge_total_edges": len(edges)}
    valid = 0
    for a, b in edges:
        key = (a, b) if a < b else (b, a)
        if key in mesh_edges:
            valid += 1
    ratio = valid / max(1, len(edges))
    return {"mesh_edge_valid_ratio": float(ratio), "mesh_edge_valid_edges": int(valid), "mesh_edge_total_edges": int(len(edges))}


def _report_summary(*, report_path: Path, mesh_path: Path) -> dict[str, Any]:
    report = _load_json(report_path)
    mesh_npz = np.load(mesh_path)
    vertices = np.asarray(mesh_npz["vertices"], dtype=float)
    faces = np.asarray(mesh_npz["faces"], dtype=int) if "faces" in mesh_npz else None
    mesh_edges = _mesh_edge_set(faces) if faces is not None else set()

    raw_edges = _collect_edges(report)
    uniq = _unique_edges(raw_edges)
    stats = _graph_stats(uniq)
    length = _seam_length(uniq, vertices)
    conformance = _edge_conformance(uniq, mesh_edges)

    reproj = report.get("reprojection")
    return {
        "report_path": str(report_path),
        "mesh_path": str(mesh_path),
        "solver": report.get("solver"),
        "total_cost": report.get("total_cost"),
        "mdl_cost": report.get("mdl_cost"),
        "warnings_count": len(report.get("warnings") or []),
        "unique_edges": stats.unique_edge_count,
        "unique_vertices": stats.unique_vertex_count,
        "components": stats.component_count,
        "cycle_components": stats.cycle_component_count,
        "open_components": stats.open_component_count,
        "max_degree": stats.max_degree,
        "junction_vertices": stats.junction_vertex_count,
        "seam_length_l2": float(length),
        "edge_conformance": conformance,
        "reprojection": reproj if isinstance(reproj, dict) else None,
        "provenance": report.get("provenance") if isinstance(report.get("provenance"), dict) else None,
    }


def _safe_ratio(numer: float, denom: float) -> float | None:
    if denom == 0.0 or not math.isfinite(denom):
        return None
    return float(numer / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Bundle root created by protocol_strategy2_bundle.py")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional explicit output path (defaults to bundle/manifests/seam_compare_metrics.json).",
    )
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    protocol_path = bundle / "manifests" / "protocol_strategy2.json"
    if not protocol_path.exists():
        raise SystemExit(f"Missing protocol manifest: {protocol_path}")

    protocol = _load_json(protocol_path)
    inputs = protocol.get("inputs") or {}
    outputs = protocol.get("outputs") or {}

    base_mesh = Path(inputs["base_mesh"])
    rom_mesh = Path(inputs["rom_mesh"])
    base_native = Path(inputs["base_seams"])
    rom_native = Path(inputs["rom_seams"])
    base_from_rom = Path(outputs["base_with_rom"])
    rom_from_base = Path(outputs["rom_with_base"])

    summary = {
        "bundle": str(bundle),
        "timestamp_utc": protocol.get("timestamp_utc"),
        "inputs": {
            "base_tag": inputs.get("base_tag"),
            "rom_tag": inputs.get("rom_tag"),
            "base_mesh": str(base_mesh),
            "rom_mesh": str(rom_mesh),
            "base_native_seams": str(base_native),
            "rom_native_seams": str(rom_native),
            "base_seams_from_rom_reprojected": str(base_from_rom),
            "rom_seams_from_base_reprojected": str(rom_from_base),
        },
        "reports": {
            "base_native": _report_summary(report_path=base_native, mesh_path=base_mesh),
            "rom_native": _report_summary(report_path=rom_native, mesh_path=rom_mesh),
            "base_from_rom": _report_summary(report_path=base_from_rom, mesh_path=base_mesh),
            "rom_from_base": _report_summary(report_path=rom_from_base, mesh_path=rom_mesh),
        },
        "comparisons": {},
    }

    base_native_len = float(summary["reports"]["base_native"]["seam_length_l2"])
    rom_from_base_len = float(summary["reports"]["rom_from_base"]["seam_length_l2"])
    rom_native_len = float(summary["reports"]["rom_native"]["seam_length_l2"])
    base_from_rom_len = float(summary["reports"]["base_from_rom"]["seam_length_l2"])

    # These "length deltas" are not a geodesic distortion measure; they are a simple
    # sanity metric to spot egregious collapse after reprojection.
    summary["comparisons"] = {
        "A_base_to_rom": {
            "source_native_length": base_native_len,
            "target_reprojected_length": rom_from_base_len,
            "abs_delta": float(abs(rom_from_base_len - base_native_len)),
            "rel_delta": _safe_ratio(abs(rom_from_base_len - base_native_len), base_native_len),
            "reprojection": summary["reports"]["rom_from_base"].get("reprojection"),
        },
        "B_rom_to_base": {
            "source_native_length": rom_native_len,
            "target_reprojected_length": base_from_rom_len,
            "abs_delta": float(abs(base_from_rom_len - rom_native_len)),
            "rel_delta": _safe_ratio(abs(base_from_rom_len - rom_native_len), rom_native_len),
            "reprojection": summary["reports"]["base_from_rom"].get("reprojection"),
        },
    }

    out_json = args.out_json or (bundle / "manifests" / "seam_compare_metrics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_json)


if __name__ == "__main__":
    main()

