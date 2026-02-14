#!/usr/bin/env python3
"""Reproject seam-report edge indices from one mesh topology to another."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_vertices(path: Path) -> np.ndarray:
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        if "vertices" in payload:
            vertices = payload["vertices"]
        elif "v" in payload:
            vertices = payload["v"]
        else:
            raise KeyError(f"{path} does not contain 'vertices' or 'v'.")
    else:
        vertices = payload
    arr = np.asarray(vertices, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path} vertices must be shaped (N, 3).")
    return arr


def _load_faces(path: Path) -> np.ndarray | None:
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        if "faces" not in payload:
            return None
        faces = payload["faces"]
    else:
        return None
    arr = np.asarray(faces, dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path} faces must be shaped (M, 3).")
    return arr


def _mesh_edge_set(faces: np.ndarray) -> set[tuple[int, int]]:
    faces_arr = np.asarray(faces, dtype=int)
    edge_set: set[tuple[int, int]] = set()
    for a, b, c in faces_arr:
        for u, v in ((a, b), (b, c), (c, a)):
            uu = int(u)
            vv = int(v)
            if uu == vv:
                continue
            edge_set.add((uu, vv) if uu < vv else (vv, uu))
    return edge_set


def _collect_unique_edges(report: dict[str, Any]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    panels = report.get("panels", {})
    if not isinstance(panels, dict):
        return edges
    for panel in panels.values():
        if not isinstance(panel, dict):
            continue
        entries = panel.get("edges", [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            edge = _extract_edge(entry)
            if edge is None:
                continue
            a, b = int(edge[0]), int(edge[1])
            if a == b:
                continue
            edges.add((a, b) if a < b else (b, a))
    return edges


def _edge_conformance_metrics(
    seam_edges: set[tuple[int, int]], mesh_edges: set[tuple[int, int]]
) -> dict[str, object]:
    if not seam_edges:
        return {
            "seam_unique_edge_count": 0,
            "mesh_edge_valid_count": 0,
            "mesh_edge_invalid_count": 0,
            "mesh_edge_valid_ratio": 1.0,
        }
    valid = sum(1 for edge in seam_edges if edge in mesh_edges)
    invalid = int(len(seam_edges) - valid)
    return {
        "seam_unique_edge_count": int(len(seam_edges)),
        "mesh_edge_valid_count": int(valid),
        "mesh_edge_invalid_count": int(invalid),
        "mesh_edge_valid_ratio": float(valid / max(1, len(seam_edges))),
    }


def _nearest_map(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    source_indices: list[int],
    *,
    batch_size: int = 512,
) -> tuple[dict[int, int], dict[int, float]]:
    mapping: dict[int, int] = {}
    distances: dict[int, float] = {}
    if not source_indices:
        return mapping, distances

    source_coords = source_vertices[np.asarray(source_indices, dtype=int)]
    target = np.asarray(target_vertices, dtype=float)

    for start in range(0, len(source_indices), max(1, int(batch_size))):
        end = min(len(source_indices), start + max(1, int(batch_size)))
        block = source_coords[start:end]  # (B, 3)
        # Squared Euclidean distance matrix (B, T)
        diff = block[:, None, :] - target[None, :, :]
        dist2 = np.einsum("bij,bij->bi", diff, diff)
        nearest = np.argmin(dist2, axis=1)
        nearest_dist = np.sqrt(np.min(dist2, axis=1))
        for local_idx, src_idx in enumerate(source_indices[start:end]):
            tgt_idx = int(nearest[local_idx])
            mapping[int(src_idx)] = tgt_idx
            distances[int(src_idx)] = float(nearest_dist[local_idx])
    return mapping, distances


def _load_vertex_map_artifact(
    map_path: Path,
    *,
    source_vertex_count: int,
    target_vertex_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str]:
    payload = np.load(map_path, allow_pickle=True)
    candidates: list[tuple[np.ndarray, np.ndarray, str]] = []
    if "source_to_target_indices" in payload and "source_to_target_distances" in payload:
        candidates.append(
            (
                np.asarray(payload["source_to_target_indices"], dtype=np.int64),
                np.asarray(payload["source_to_target_distances"], dtype=np.float64),
                "source_to_target",
            )
        )
    if "target_to_source_indices" in payload and "target_to_source_distances" in payload:
        candidates.append(
            (
                np.asarray(payload["target_to_source_indices"], dtype=np.int64),
                np.asarray(payload["target_to_source_distances"], dtype=np.float64),
                "target_to_source",
            )
        )
    if not candidates:
        raise KeyError(
            f"{map_path} missing required mapping arrays "
            "('source_to_target_*' and/or 'target_to_source_*')."
        )

    selected: tuple[np.ndarray, np.ndarray, str] | None = None
    for indices, distances, direction in candidates:
        if indices.ndim != 1 or distances.ndim != 1 or len(indices) != len(distances):
            continue
        if len(indices) != source_vertex_count:
            continue
        if np.any(indices < 0) or np.any(indices >= target_vertex_count):
            continue
        selected = (indices, distances, direction)
        break
    if selected is None:
        lengths = {direction: int(len(indices)) for indices, _, direction in candidates}
        raise ValueError(
            f"{map_path} has no compatible mapping array for source={source_vertex_count}, target={target_vertex_count}. "
            f"available_lengths={lengths}"
        )
    indices, distances, direction = selected

    meta: dict[str, Any] = {}
    if "meta" in payload:
        try:
            raw = payload["meta"].item()
            if isinstance(raw, dict):
                meta = raw
        except Exception:
            meta = {}
    return indices, distances, meta, direction


def _extract_edge(entry: Any) -> tuple[int, int] | None:
    if isinstance(entry, dict):
        raw = entry.get("edge")
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return int(raw[0]), int(raw[1])
        return None
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        return int(entry[0]), int(entry[1])
    return None


def _reproject_report(
    report: dict[str, Any],
    *,
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    source_path: Path,
    target_path: Path,
    vertex_map_file: Path | None = None,
) -> dict[str, Any]:
    panels = report.get("panels", {})
    if not isinstance(panels, dict):
        raise TypeError("seam_report.json must contain a 'panels' object.")

    used_source_vertices: set[int] = set()
    total_edges_in = 0
    for panel in panels.values():
        if not isinstance(panel, dict):
            continue
        entries = panel.get("edges", [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            edge = _extract_edge(entry)
            if edge is None:
                continue
            a, b = edge
            if a < 0 or b < 0 or a >= len(source_vertices) or b >= len(source_vertices):
                continue
            used_source_vertices.add(a)
            used_source_vertices.add(b)
            total_edges_in += 1

    source_index_list = sorted(used_source_vertices)
    mapping_mode = "nearest_subset"
    map_meta: dict[str, Any] = {}
    if vertex_map_file is not None:
        full_indices, full_distances, map_meta, map_direction = _load_vertex_map_artifact(
            vertex_map_file,
            source_vertex_count=len(source_vertices),
            target_vertex_count=len(target_vertices),
        )
        vertex_map = {idx: int(full_indices[idx]) for idx in source_index_list}
        distances = {idx: float(full_distances[idx]) for idx in source_index_list}
        mapping_mode = f"explicit_map:{map_direction}"
    else:
        vertex_map, distances = _nearest_map(
            source_vertices,
            target_vertices,
            source_index_list,
        )

    total_edges_out = 0
    collapsed_edges = 0
    duplicate_edges = 0
    for panel_name, panel in panels.items():
        if not isinstance(panel, dict):
            continue
        entries = panel.get("edges", [])
        if not isinstance(entries, list):
            continue

        seen: set[tuple[int, int]] = set()
        remapped_entries: list[dict[str, Any]] = []
        for entry in entries:
            edge = _extract_edge(entry)
            if edge is None:
                continue
            src_a, src_b = edge
            if src_a not in vertex_map or src_b not in vertex_map:
                continue
            dst_a = int(vertex_map[src_a])
            dst_b = int(vertex_map[src_b])
            if dst_a == dst_b:
                collapsed_edges += 1
                continue
            normalized = (dst_a, dst_b) if dst_a < dst_b else (dst_b, dst_a)
            if normalized in seen:
                duplicate_edges += 1
                continue
            seen.add(normalized)

            if isinstance(entry, dict):
                updated = dict(entry)
            else:
                updated = {}
            updated["edge"] = [normalized[0], normalized[1]]
            remapped_entries.append(updated)
            total_edges_out += 1

        panel["edges"] = remapped_entries
        panel_warnings = panel.get("warnings", [])
        if not isinstance(panel_warnings, list):
            panel_warnings = []
        panel_warnings.append(
            f"reprojected from {source_path.name} to {target_path.name}; "
            f"edges now on target topology"
        )
        panel["warnings"] = panel_warnings
        panels[panel_name] = panel

    max_distance = max(distances.values()) if distances else 0.0
    mean_distance = float(np.mean(list(distances.values()))) if distances else 0.0

    report["panels"] = panels
    warnings = report.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = []
    warnings.append(
        "seam report reprojected to target topology "
        f"({source_path.name} -> {target_path.name})"
    )
    report["warnings"] = warnings
    edge_retention_ratio = (float(total_edges_out) / float(total_edges_in)) if total_edges_in > 0 else 0.0
    unique_target_vertices = len(set(vertex_map.values())) if vertex_map else 0
    target_vertex_collision_ratio = (
        float(max(0, len(source_index_list) - unique_target_vertices)) / float(len(source_index_list))
        if source_index_list
        else 0.0
    )
    report["reprojection"] = {
        "source_mesh": str(source_path),
        "target_mesh": str(target_path),
        "source_vertex_count": int(len(source_vertices)),
        "target_vertex_count": int(len(target_vertices)),
        "mapped_vertex_count": int(len(source_index_list)),
        "unique_target_vertices": int(unique_target_vertices),
        "target_vertex_collision_ratio": float(target_vertex_collision_ratio),
        "mapping_mode": mapping_mode,
        "vertex_map_file": str(vertex_map_file) if vertex_map_file is not None else None,
        "vertex_map_meta": map_meta,
        "max_distance": float(max_distance),
        "mean_distance": float(mean_distance),
        "edge_count_in": int(total_edges_in),
        "edge_count_out": int(total_edges_out),
        "edge_retention_ratio": float(edge_retention_ratio),
        "collapsed_edges_dropped": int(collapsed_edges),
        "duplicate_edges_dropped": int(duplicate_edges),
    }
    return report


def _quality_violations(
    reprojection: dict[str, Any],
    *,
    max_mean_distance: float,
    max_distance: float,
    min_edge_retention: float,
    max_target_collision_ratio: float,
) -> list[str]:
    violations: list[str] = []
    mean_distance = float(reprojection.get("mean_distance", 0.0))
    max_dist = float(reprojection.get("max_distance", 0.0))
    edge_retention = float(reprojection.get("edge_retention_ratio", 0.0))
    collision_ratio = float(reprojection.get("target_vertex_collision_ratio", 0.0))

    if mean_distance > max_mean_distance:
        violations.append(
            f"mean_distance {mean_distance:.6g} exceeds threshold {max_mean_distance:.6g}"
        )
    if max_dist > max_distance:
        violations.append(
            f"max_distance {max_dist:.6g} exceeds threshold {max_distance:.6g}"
        )
    if edge_retention < min_edge_retention:
        violations.append(
            f"edge_retention_ratio {edge_retention:.6g} below threshold {min_edge_retention:.6g}"
        )
    if collision_ratio > max_target_collision_ratio:
        violations.append(
            "target_vertex_collision_ratio "
            f"{collision_ratio:.6g} exceeds threshold {max_target_collision_ratio:.6g}"
        )
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="Input seam_report.json.")
    parser.add_argument("--source-mesh", type=Path, required=True, help="Mesh used by the input seam report.")
    parser.add_argument("--target-mesh", type=Path, required=True, help="Mesh to reproject seam edges onto.")
    parser.add_argument("--out", type=Path, required=True, help="Output seam_report.json path.")
    parser.add_argument(
        "--vertex-map-file",
        type=Path,
        default=None,
        help=(
            "Optional full-mesh correspondence map (.npz) from source->target created by "
            "scripts/build_mesh_vertex_map.py. When provided, reprojection uses this map instead of seam-point NN."
        ),
    )
    parser.add_argument(
        "--max-mean-distance",
        type=float,
        default=0.05,
        help="Quality threshold in meters for mean source->target NN distance (default: 0.05).",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.15,
        help="Quality threshold in meters for max source->target NN distance (default: 0.15).",
    )
    parser.add_argument(
        "--min-edge-retention",
        type=float,
        default=0.5,
        help="Minimum retained edge ratio edge_count_out/edge_count_in (default: 0.5).",
    )
    parser.add_argument(
        "--strict-quality",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, exit non-zero when reprojection quality thresholds are violated.",
    )
    parser.add_argument(
        "--max-target-collision-ratio",
        type=float,
        default=0.2,
        help="Maximum allowed many-to-one source->target vertex collision ratio (default: 0.2).",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    source_vertices = _load_vertices(args.source_mesh)
    target_vertices = _load_vertices(args.target_mesh)
    target_faces = _load_faces(args.target_mesh)
    out_report = _reproject_report(
        report,
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        source_path=args.source_mesh,
        target_path=args.target_mesh,
        vertex_map_file=args.vertex_map_file,
    )
    repro = out_report.get("reprojection", {})
    violations = _quality_violations(
        repro,
        max_mean_distance=float(args.max_mean_distance),
        max_distance=float(args.max_distance),
        min_edge_retention=float(args.min_edge_retention),
        max_target_collision_ratio=float(args.max_target_collision_ratio),
    )
    if violations:
        warnings = out_report.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = []
        warnings.extend([f"reprojection quality warning: {item}" for item in violations])
        out_report["warnings"] = warnings
        repro["quality_violations"] = violations
        repro["quality_ok"] = False
    else:
        repro["quality_violations"] = []
        repro["quality_ok"] = True
    out_report["reprojection"] = repro

    # Always include mesh-edge conformance metrics when faces are available.
    if target_faces is not None:
        seam_edges = _collect_unique_edges(out_report)
        mesh_edges = _mesh_edge_set(target_faces)
        metrics = _edge_conformance_metrics(seam_edges, mesh_edges)
        metrics["mesh_vertex_count"] = int(len(target_vertices))
        metrics["mesh_face_count"] = int(len(target_faces))
        out_report["metrics"] = dict(metrics)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_report, indent=2), encoding="utf-8")
    print(
        "Reprojected seam report: "
        f"mapped {repro.get('mapped_vertex_count', 0)} vertices, "
        f"unique_target={repro.get('unique_target_vertices', 0)}, "
        f"collision_ratio={repro.get('target_vertex_collision_ratio', 0.0):.6g}, "
        f"edges {repro.get('edge_count_in', 0)} -> {repro.get('edge_count_out', 0)}, "
        f"max_dist={repro.get('max_distance', 0.0):.6g}, "
        f"mean_dist={repro.get('mean_distance', 0.0):.6g}, "
        f"edge_retention={repro.get('edge_retention_ratio', 0.0):.6g}"
    )
    if violations:
        print("Quality violations:")
        for violation in violations:
            print(f"  - {violation}")
    print(f"Wrote {args.out}")
    if violations and args.strict_quality:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
