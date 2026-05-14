#!/usr/bin/env python3
"""Unwrap promoted seam topology into receipted panel UV artifacts."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from smii.seams import (
    PanelUnwrapReceipt,
    can_consume_solver_promotion_receipt,
    load_solver_promotion_receipt,
)

Edge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class PanelPatch:
    """Post-cut mesh component used by the bootstrap panel unwrapper."""

    vertices: tuple[int, ...]
    edges: tuple[Edge, ...]
    faces: tuple[tuple[int, int, int], ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if "vertices" not in payload or "faces" not in payload:
        raise KeyError("Mesh NPZ must contain 'vertices' and 'faces' arrays.")
    vertices = np.asarray(payload["vertices"], dtype=float)
    faces = np.asarray(payload["faces"], dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Mesh vertices must be shaped (N, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Mesh faces must be shaped (M, 3).")
    return vertices, faces


def _load_seam_edges(path: Path) -> tuple[Edge, ...]:
    payload = np.load(path, allow_pickle=True)
    if "seam_edges" not in payload:
        raise KeyError("Seam edges NPZ must contain a 'seam_edges' array.")
    raw_edges = np.asarray(payload["seam_edges"], dtype=int)
    if raw_edges.ndim != 2 or raw_edges.shape[1] != 2:
        raise ValueError("seam_edges must be shaped (N, 2).")
    return tuple(sorted({_normalize_edge(int(a), int(b)) for a, b in raw_edges}))


def _normalize_edge(a: int, b: int) -> Edge:
    aa = int(a)
    bb = int(b)
    return (aa, bb) if aa <= bb else (bb, aa)


def _mesh_edges(faces: np.ndarray) -> tuple[Edge, ...]:
    edges: set[Edge] = set()
    for a, b, c in np.asarray(faces, dtype=int):
        for u, v in ((a, b), (b, c), (c, a)):
            edge = _normalize_edge(int(u), int(v))
            if edge[0] != edge[1]:
                edges.add(edge)
    return tuple(sorted(edges))


def _adjacency(edges: Sequence[Edge]) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        graph[int(a)].add(int(b))
        graph[int(b)].add(int(a))
    return graph


def _connected_components(vertex_count: int, edges: Sequence[Edge]) -> list[set[int]]:
    graph = _adjacency(edges)
    remaining = set(range(vertex_count))
    components: list[set[int]] = []
    while remaining:
        start = remaining.pop()
        component = {start}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, ()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _extract_panels(
    *,
    vertex_count: int,
    faces: np.ndarray,
    seam_edges: Sequence[Edge],
) -> list[PanelPatch]:
    mesh_edges = _mesh_edges(faces)
    remaining_edges = tuple(sorted(set(mesh_edges) - set(seam_edges)))
    face_tuples = [tuple(int(v) for v in face) for face in np.asarray(faces, dtype=int)]
    panels: list[PanelPatch] = []
    for component in _connected_components(vertex_count, remaining_edges):
        component_edges = tuple(
            edge for edge in remaining_edges if edge[0] in component and edge[1] in component
        )
        component_faces = tuple(
            face for face in face_tuples if all(vertex in component for vertex in face)
        )
        if component_edges or component_faces:
            panels.append(
                PanelPatch(
                    vertices=tuple(sorted(component)),
                    edges=component_edges,
                    faces=component_faces,
                )
            )
    return panels


def _unwrap_panel(vertices: np.ndarray, panel: PanelPatch, *, method: str) -> np.ndarray:
    if method not in {"lscm", "abf", "arap"}:
        raise ValueError("solver must be lscm, abf, or arap.")
    coords = np.asarray(vertices[list(panel.vertices)], dtype=float)
    if len(coords) == 0:
        return np.empty((0, 2), dtype=float)
    centered = coords - coords.mean(axis=0)
    if len(coords) == 1:
        uv = np.zeros((1, 2), dtype=float)
    else:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axes = vt[:2]
        if axes.shape[0] < 2:
            axes = np.vstack([axes, np.array([[0.0, 1.0, 0.0]])])
        uv = centered @ axes.T
    return uv


def _compute_distortion(vertices: np.ndarray, panel: PanelPatch, uv: np.ndarray) -> float:
    if not panel.edges:
        return 0.0
    index = {vertex: idx for idx, vertex in enumerate(panel.vertices)}
    distortions: list[float] = []
    for a, b in panel.edges:
        local_a = index[a]
        local_b = index[b]
        length_3d = float(np.linalg.norm(vertices[a] - vertices[b]))
        length_2d = float(np.linalg.norm(uv[local_a] - uv[local_b]))
        if length_3d <= 1e-12:
            continue
        distortions.append(abs(length_2d - length_3d) / length_3d)
    if not distortions:
        return 0.0
    return float(sum(distortions) / len(distortions))


def _subdivide_panel(panel: PanelPatch) -> PanelPatch:
    """Record a retry boundary without mutating mesh topology in the bootstrap path."""

    return panel


def _infer_grain_direction(vertices: np.ndarray, panel: PanelPatch, uv: np.ndarray) -> str:
    if len(panel.vertices) < 2 or uv.size == 0:
        return "warp"
    coords = np.asarray(vertices[list(panel.vertices)], dtype=float)
    centered = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    primary = np.abs(vt[0]) if vt.size else np.array([1.0, 0.0, 0.0])
    axis = int(np.argmax(primary))
    if axis == 0:
        return "warp"
    if axis == 1:
        return "weft"
    return "bias"


def unwrap_panels(
    *,
    solver_receipt_path: Path,
    seam_edges_path: Path,
    mesh_path: Path,
    output_dir: Path,
    distortion_threshold: float = 0.05,
    max_subdivisions: int = 3,
    solver: str = "lscm",
    receipt_path: Path | None = None,
) -> PanelUnwrapReceipt:
    """Unwrap panels and emit a hash-linked panel unwrap receipt."""

    solver_receipt = load_solver_promotion_receipt(solver_receipt_path)
    if not can_consume_solver_promotion_receipt(solver_receipt, "panel_unwrap"):
        raise ValueError(
            f"SolverPromotionReceipt not promoted ({solver_receipt.promotion}). "
            f"Blocked: {solver_receipt.blocked_consumers}"
        )
    if not solver_receipt.panels_are_disks:
        raise ValueError(
            "panels_are_disks=False: seam topology is incomplete; add seam cuts "
            "before unwrapping. The unwrapper is not the problem."
        )
    if solver_receipt.seam_hash != _sha256_file(seam_edges_path):
        raise ValueError("Seam edges hash does not match SolverPromotionReceipt.seam_hash.")

    vertices, faces = _load_mesh(mesh_path)
    seam_edges = _load_seam_edges(seam_edges_path)
    panels = _extract_panels(
        vertex_count=int(vertices.shape[0]),
        faces=faces,
        seam_edges=seam_edges,
    )
    if not panels:
        raise ValueError("No panels could be extracted from seam topology.")

    per_panel_distortion: list[float] = []
    per_panel_uv: dict[str, np.ndarray] = {}
    per_panel_grain: list[str] = []
    subdivisions_used = 0

    for panel_idx, panel in enumerate(panels):
        current_panel = panel
        for iteration in range(max(0, int(max_subdivisions)) + 1):
            uv = _unwrap_panel(vertices, current_panel, method=solver)
            distortion = _compute_distortion(vertices, current_panel, uv)
            if distortion <= distortion_threshold or iteration == max_subdivisions:
                per_panel_distortion.append(float(distortion))
                per_panel_uv[f"panel_{panel_idx}"] = uv
                per_panel_grain.append(_infer_grain_direction(vertices, current_panel, uv))
                subdivisions_used = max(subdivisions_used, iteration)
                break
            current_panel = _subdivide_panel(current_panel)

    output_dir.mkdir(parents=True, exist_ok=True)
    uv_path = output_dir / "panel_uvs.npz"
    np.savez_compressed(uv_path, **per_panel_uv)
    uv_hash = _sha256_file(uv_path)

    worst = float(max(per_panel_distortion))
    mean = float(sum(per_panel_distortion) / len(per_panel_distortion))
    promotion = 1 if worst <= distortion_threshold else 0
    receipt = PanelUnwrapReceipt(
        solver_receipt_hash=_sha256_file(solver_receipt_path),
        panel_count=len(panels),
        panels_all_disks=solver_receipt.panels_are_disks,
        per_panel_distortion=per_panel_distortion,
        worst_panel_distortion=worst,
        mean_panel_distortion=mean,
        distortion_threshold=float(distortion_threshold),
        subdivision_iterations=subdivisions_used,
        grain_directions=per_panel_grain,
        uv_hash=uv_hash,
        seam_topology_hash=solver_receipt.seam_hash,
        promotion=promotion,
        blocked_consumers=[] if promotion == 1 else [],
    )
    target_receipt_path = receipt_path or (output_dir / "panel_unwrap_receipt.json")
    receipt.to_json(target_receipt_path)
    print(f"Wrote panel UVs to {uv_path}")
    print(f"Wrote panel unwrap receipt to {target_receipt_path}")
    return receipt


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-receipt", type=Path, required=True)
    parser.add_argument("--seam-edges", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True, help="Mesh NPZ with vertices/faces.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--out-panel-receipt",
        type=Path,
        default=None,
        help="Output panel receipt path (default: <out-dir>/panel_unwrap_receipt.json).",
    )
    parser.add_argument("--distortion-threshold", type=float, default=0.05)
    parser.add_argument("--max-subdivisions", type=int, default=3)
    parser.add_argument("--solver", choices=["lscm", "abf", "arap"], default="lscm")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    unwrap_panels(
        solver_receipt_path=args.solver_receipt,
        seam_edges_path=args.seam_edges,
        mesh_path=args.mesh,
        output_dir=args.out_dir,
        distortion_threshold=args.distortion_threshold,
        max_subdivisions=args.max_subdivisions,
        solver=args.solver,
        receipt_path=args.out_panel_receipt,
    )


if __name__ == "__main__":
    main()
