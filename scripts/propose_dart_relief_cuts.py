#!/usr/bin/env python3
"""Propose diagnostic dart and relief-cut candidates from mesh developability."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

Edge = tuple[int, int]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    if "vertices" not in payload or "faces" not in payload:
        raise KeyError("Mesh NPZ must contain 'vertices' and 'faces' arrays.")
    vertices = np.asarray(payload["vertices"], dtype=float)
    faces = np.asarray(payload["faces"], dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Mesh vertices must be shaped (N, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Mesh faces must be shaped (M, 3).")
    return vertices, faces


def _load_seam_edges(path: Path | None) -> tuple[Edge, ...]:
    if path is None:
        return ()
    payload = np.load(path, allow_pickle=False)
    if "seam_edges" not in payload:
        raise KeyError("Seam edges NPZ must contain 'seam_edges'.")
    raw_edges = np.asarray(payload["seam_edges"], dtype=int)
    if raw_edges.ndim != 2 or raw_edges.shape[1] != 2:
        raise ValueError("seam_edges must be shaped (N, 2).")
    return tuple(sorted({_normalize_edge(int(a), int(b)) for a, b in raw_edges}))


def _normalize_edge(a: int, b: int) -> Edge:
    aa = int(a)
    bb = int(b)
    return (aa, bb) if aa <= bb else (bb, aa)


def _face_edges(face: Sequence[int]) -> tuple[Edge, Edge, Edge]:
    a, b, c = (int(face[0]), int(face[1]), int(face[2]))
    return (_normalize_edge(a, b), _normalize_edge(b, c), _normalize_edge(c, a))


def _mesh_edges(faces: np.ndarray) -> tuple[Edge, ...]:
    return tuple(
        sorted({edge for face in np.asarray(faces, dtype=int) for edge in _face_edges(face)})
    )


def _adjacency(edges: Sequence[Edge]) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        graph[int(a)].add(int(b))
        graph[int(b)].add(int(a))
    return graph


def _boundary_vertices(faces: np.ndarray) -> set[int]:
    edge_counts: dict[Edge, int] = defaultdict(int)
    for face in np.asarray(faces, dtype=int):
        for edge in _face_edges(face):
            edge_counts[edge] += 1
    return {vertex for edge, count in edge_counts.items() if count == 1 for vertex in edge}


def _angle_deficit(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    angles = np.zeros(int(vertices.shape[0]), dtype=float)
    for face in np.asarray(faces, dtype=int):
        pts = vertices[face]
        for local_idx, vertex_idx in enumerate(face):
            origin = pts[local_idx]
            prev_point = pts[(local_idx - 1) % 3]
            next_point = pts[(local_idx + 1) % 3]
            a = prev_point - origin
            b = next_point - origin
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom <= 1e-12:
                continue
            cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
            angles[int(vertex_idx)] += float(np.arccos(cosine))
    return np.abs((2.0 * np.pi) - angles)


def _shortest_path_to_targets(
    graph: dict[int, set[int]],
    source: int,
    targets: set[int],
) -> list[int]:
    if source in targets:
        return [int(source)]
    queue: deque[int] = deque([int(source)])
    previous: dict[int, int | None] = {int(source): None}
    while queue:
        node = queue.popleft()
        for neighbor in sorted(graph.get(node, ())):
            if neighbor in previous:
                continue
            previous[neighbor] = node
            if neighbor in targets:
                path = [neighbor]
                while path[-1] != int(source):
                    parent = previous[path[-1]]
                    if parent is None:
                        break
                    path.append(parent)
                path.reverse()
                return path
            queue.append(neighbor)
    return [int(source)]


def _path_edges(path_vertices: Sequence[int]) -> list[list[int]]:
    return [[int(a), int(b)] for a, b in zip(path_vertices, path_vertices[1:], strict=False)]


def propose_candidates(
    *,
    mesh_path: Path,
    out_json: Path,
    seam_edges_path: Path | None = None,
    max_candidates: int = 12,
    percentile: float = 95.0,
) -> dict[str, object]:
    """Write diagnostic dart/relief candidates and return the payload."""

    vertices, faces = _load_mesh(mesh_path)
    seam_edges = _load_seam_edges(seam_edges_path)
    mesh_edges = _mesh_edges(faces)
    graph = _adjacency(mesh_edges)
    boundary_vertices = _boundary_vertices(faces)
    seam_vertices = {vertex for edge in seam_edges for vertex in edge}
    targets = seam_vertices or boundary_vertices
    if not targets:
        targets = set(range(int(vertices.shape[0])))

    deficit = _angle_deficit(vertices, faces)
    threshold = float(np.percentile(deficit, float(percentile))) if deficit.size else 0.0
    ordered = sorted(
        (idx for idx, value in enumerate(deficit) if float(value) >= threshold),
        key=lambda idx: (-float(deficit[idx]), int(idx)),
    )
    candidates: list[dict[str, object]] = []
    used: set[int] = set()
    for vertex_idx in ordered:
        if len(candidates) >= max(0, int(max_candidates)):
            break
        if vertex_idx in used:
            continue
        path_vertices = _shortest_path_to_targets(graph, int(vertex_idx), targets)
        endpoint = int(path_vertices[-1])
        endpoint_class = (
            "existing_seam"
            if endpoint in seam_vertices
            else "mesh_boundary"
            if endpoint in boundary_vertices
            else "diagnostic_local"
        )
        candidate_type = "dart" if endpoint_class == "existing_seam" else "relief_cut"
        path_length = 0.0
        for a, b in zip(path_vertices, path_vertices[1:], strict=False):
            path_length += float(np.linalg.norm(vertices[int(a)] - vertices[int(b)]))
        candidates.append(
            {
                "candidate_id": f"{candidate_type}_{len(candidates):03d}",
                "candidate_type": candidate_type,
                "apex_vertex": int(vertex_idx),
                "curvature_score": float(deficit[int(vertex_idx)]),
                "developability_threshold": threshold,
                "endpoint_vertex": endpoint,
                "endpoint_class": endpoint_class,
                "path_vertices": [int(value) for value in path_vertices],
                "path_edges": _path_edges(path_vertices),
                "path_length": path_length,
                "typed_topology_effect": (
                    "diagnostic_wedge_take-up_candidate"
                    if candidate_type == "dart"
                    else "diagnostic_boundary_relief_candidate"
                ),
            }
        )
        used.update(path_vertices[:2])

    payload: dict[str, object] = {
        "diagnostic_only": True,
        "mesh_path": str(mesh_path),
        "mesh_hash": _sha256_file(mesh_path),
        "seam_edges_path": str(seam_edges_path) if seam_edges_path is not None else None,
        "seam_edges_hash": _sha256_file(seam_edges_path) if seam_edges_path is not None else None,
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "curvature_metric": "absolute_angle_deficit",
        "candidate_threshold_percentile": float(percentile),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote dart/relief candidates to {out_json}")
    return payload


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--seam-edges", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--percentile", type=float, default=95.0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    propose_candidates(
        mesh_path=args.mesh,
        seam_edges_path=args.seam_edges,
        out_json=args.out_json,
        max_candidates=args.max_candidates,
        percentile=args.percentile,
    )


if __name__ == "__main__":
    main()
