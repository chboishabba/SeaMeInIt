#!/usr/bin/env python3
"""Render diagnostic panel UVs and blocked 2D pattern previews."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from smii.seams import load_panel_unwrap_receipt

Edge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class PanelTopology:
    vertices: tuple[int, ...]
    faces: tuple[tuple[int, int, int], ...]
    boundary_edges: tuple[Edge, ...]
    seam_edges: tuple[Edge, ...]


@dataclass(frozen=True, slots=True)
class SeamGraphSummary:
    edge_segment_count: int
    vertex_count: int
    connected_component_count: int
    endpoint_count: int
    branch_vertex_count: int
    largest_component_edge_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_panel_uvs(path: Path, panel_count: int) -> list[np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    panels: list[np.ndarray] = []
    for idx in range(panel_count):
        key = f"panel_{idx}"
        if key not in payload:
            raise KeyError(f"Panel UV artifact is missing '{key}'.")
        panel = np.asarray(payload[key], dtype=float)
        if panel.ndim != 2 or panel.shape[1] != 2:
            raise ValueError(f"{key} must be shaped (N, 2).")
        if not np.isfinite(panel).all():
            raise ValueError(f"{key} must contain finite UV coordinates.")
        panels.append(panel)
    return panels


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _convex_hull(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return np.asarray(points, dtype=float)
    unique = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(unique) <= 2:
        return unique
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]
    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _panel_color(distortion: float, threshold: float) -> str:
    if distortion <= threshold:
        return "#1f7a4d"
    if distortion <= threshold * 2.0:
        return "#c47a00"
    return "#b42318"


def _bounds(panels: Sequence[np.ndarray]) -> tuple[float, float, float, float]:
    stacked = np.vstack([panel for panel in panels if panel.size])
    min_x = float(stacked[:, 0].min())
    max_x = float(stacked[:, 0].max())
    min_y = float(stacked[:, 1].min())
    max_y = float(stacked[:, 1].max())
    if abs(max_x - min_x) <= 1e-12:
        max_x = min_x + 1.0
    if abs(max_y - min_y) <= 1e-12:
        max_y = min_y + 1.0
    return min_x, max_x, min_y, max_y


def _transform_points(
    points: np.ndarray,
    *,
    min_x: float,
    max_y: float,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> np.ndarray:
    transformed = np.asarray(points, dtype=float).copy()
    transformed[:, 0] = (transformed[:, 0] - min_x) * scale + offset_x
    transformed[:, 1] = (max_y - transformed[:, 1]) * scale + offset_y
    return transformed


def _points_attr(points: np.ndarray) -> str:
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in np.asarray(points, dtype=float))


def _normalize_edge(a: int, b: int) -> Edge:
    aa = int(a)
    bb = int(b)
    return (aa, bb) if aa <= bb else (bb, aa)


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
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(graph.get(node, ())):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _seam_graph_summary(seam_edges: Sequence[Edge]) -> SeamGraphSummary:
    normalized_edges = tuple(sorted({_normalize_edge(a, b) for a, b in seam_edges}))
    if not normalized_edges:
        return SeamGraphSummary(
            edge_segment_count=0,
            vertex_count=0,
            connected_component_count=0,
            endpoint_count=0,
            branch_vertex_count=0,
            largest_component_edge_count=0,
        )
    graph = _adjacency(normalized_edges)
    vertices = set(graph)
    remaining = set(vertices)
    component_edge_counts: list[int] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(graph.get(node, ())):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        component_edge_counts.append(
            sum(1 for edge in normalized_edges if edge[0] in component and edge[1] in component)
        )
    return SeamGraphSummary(
        edge_segment_count=len(normalized_edges),
        vertex_count=len(vertices),
        connected_component_count=len(component_edge_counts),
        endpoint_count=sum(1 for vertex in vertices if len(graph[vertex]) == 1),
        branch_vertex_count=sum(1 for vertex in vertices if len(graph[vertex]) > 2),
        largest_component_edge_count=max(component_edge_counts),
    )


def _face_edges(face: tuple[int, int, int]) -> tuple[Edge, Edge, Edge]:
    a, b, c = face
    return (_normalize_edge(a, b), _normalize_edge(b, c), _normalize_edge(c, a))


def _extract_panel_topologies(
    *,
    vertex_count: int,
    faces: np.ndarray,
    seam_edges: np.ndarray,
) -> list[PanelTopology]:
    seam_edge_set = {_normalize_edge(int(a), int(b)) for a, b in np.asarray(seam_edges, dtype=int)}
    face_tuples = [tuple(int(v) for v in face) for face in np.asarray(faces, dtype=int)]
    mesh_edges = sorted({edge for face in face_tuples for edge in _face_edges(face)})
    remaining_edges = [edge for edge in mesh_edges if edge not in seam_edge_set]
    panels: list[PanelTopology] = []
    for component in _connected_components(vertex_count, remaining_edges):
        component_faces = tuple(
            face for face in face_tuples if all(vertex in component for vertex in face)
        )
        component_vertices = tuple(sorted(component))
        if not component_faces:
            continue
        face_edge_counts = Counter(edge for face in component_faces for edge in _face_edges(face))
        boundary_edges = tuple(
            sorted(edge for edge, count in face_edge_counts.items() if count == 1)
        )
        component_seams = tuple(
            sorted(edge for edge in seam_edge_set if edge[0] in component and edge[1] in component)
        )
        panels.append(
            PanelTopology(
                vertices=component_vertices,
                faces=component_faces,
                boundary_edges=boundary_edges,
                seam_edges=component_seams,
            )
        )
    return panels


def _extract_panel_topologies_from_face_labels(
    *,
    faces: np.ndarray,
    seam_edges: np.ndarray,
    face_labels: np.ndarray,
) -> list[PanelTopology]:
    if int(face_labels.shape[0]) != int(faces.shape[0]):
        raise ValueError(
            "face_labels length must match mesh face count: "
            f"labels={face_labels.shape[0]}, faces={faces.shape[0]}."
        )
    seam_edge_set = {_normalize_edge(int(a), int(b)) for a, b in np.asarray(seam_edges, dtype=int)}
    panels: list[PanelTopology] = []
    for label in sorted({int(value) for value in face_labels}):
        component_faces = tuple(
            tuple(int(vertex) for vertex in faces[idx])
            for idx in np.where(face_labels == int(label))[0]
        )
        component_vertices = tuple(sorted({vertex for face in component_faces for vertex in face}))
        component_vertex_set = set(component_vertices)
        face_edge_counts = Counter(edge for face in component_faces for edge in _face_edges(face))
        boundary_edges = tuple(
            sorted(edge for edge, count in face_edge_counts.items() if count == 1)
        )
        component_seams = tuple(
            sorted(
                edge
                for edge in seam_edge_set
                if edge[0] in component_vertex_set and edge[1] in component_vertex_set
            )
        )
        panels.append(
            PanelTopology(
                vertices=component_vertices,
                faces=component_faces,
                boundary_edges=boundary_edges,
                seam_edges=component_seams,
            )
        )
    return panels


def _render_uv_svg(
    *,
    panels: Sequence[np.ndarray],
    distortions: Sequence[float],
    threshold: float,
    receipt_summary: str,
) -> str:
    width = 1100.0
    height = 760.0
    margin = 60.0
    min_x, max_x, min_y, max_y = _bounds(panels)
    scale = min((width - margin * 2.0) / (max_x - min_x), (height - 140.0) / (max_y - min_y))
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
            f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">'
        ),
        "  <title>Raw panel UV point-cloud diagnostic</title>",
        "  <style>",
        "    text { font-family: Inter, Arial, sans-serif; fill: #242424; }",
        "    .note { fill: #6b6b6b; font-size: 13px; }",
        "    .panel-point { opacity: 0.18; }",
        "    .panel-hull { fill-opacity: 0.08; stroke-width: 1.4; }",
        "  </style>",
        '  <rect x="0" y="0" width="1100" height="760" fill="#fbfaf7"/>',
        (
            '  <text x="36" y="34" font-size="20" font-weight="700">'
            "Raw panel UV point-cloud diagnostic</text>"
        ),
        f'  <text x="36" y="58" class="note">{escape(receipt_summary)}</text>',
        (
            '  <text x="36" y="76" class="note">This view is only UV samples plus '
            "convex hulls; it is not the face-backed panel topology.</text>"
        ),
        (
            '  <text x="36" y="94" class="note">Use diagnostic_flat_cut_sheet.svg '
            "for the actual panel/seam review surface.</text>"
        ),
    ]
    for idx, panel in enumerate(panels):
        color = _panel_color(float(distortions[idx]), threshold)
        transformed = _transform_points(
            panel,
            min_x=min_x,
            max_y=max_y,
            scale=scale,
            offset_x=margin,
            offset_y=125.0,
        )
        hull = _convex_hull(transformed)
        sample_step = max(1, len(transformed) // 1500)
        sample = transformed[::sample_step]
        lines.append(f'  <g id="panel_{idx}" data-distortion="{float(distortions[idx]):.9f}">')
        if len(hull) >= 2:
            lines.append(
                f'    <polygon class="panel-hull" points="{_points_attr(hull)}" '
                f'fill="{color}" stroke="{color}"/>'
            )
        for x, y in sample:
            lines.append(
                f'    <circle class="panel-point" cx="{x:.3f}" cy="{y:.3f}" '
                f'r="1.2" fill="{color}"/>'
            )
        label_x = float(hull[0, 0]) if len(hull) else margin
        label_y = float(hull[0, 1]) - 8.0 if len(hull) else 88.0
        lines.append(
            f'    <text x="{label_x:.3f}" y="{label_y:.3f}" font-size="12">'
            f"P{idx} distortion={float(distortions[idx]):.4f}</text>"
        )
        lines.append("  </g>")
    lines.extend(["</svg>", ""])
    return "\n".join(lines)


def _render_patterns_svg(
    *,
    panels: Sequence[np.ndarray],
    distortions: Sequence[float],
    threshold: float,
    grain_directions: Sequence[str],
    receipt_summary: str,
) -> str:
    cell_width = 220.0
    cell_height = 260.0
    margin = 36.0
    cols = max(1, min(4, len(panels)))
    rows = int(np.ceil(len(panels) / cols))
    width = margin * 2.0 + cell_width * cols
    height = margin * 2.0 + 70.0 + cell_height * rows
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
            f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">'
        ),
        "  <title>Diagnostic 2D pattern preview</title>",
        "  <style>",
        "    text { font-family: Inter, Arial, sans-serif; fill: #242424; }",
        "    .note { fill: #6b6b6b; font-size: 12px; }",
        "    .pattern-outline { fill-opacity: 0.12; stroke-width: 1.6; }",
        "    .grain { stroke: #4b5563; stroke-width: 1.2; stroke-dasharray: 4 3; }",
        "  </style>",
        f'  <rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="#fbfaf7"/>',
        '  <text x="36" y="34" font-size="20" font-weight="700">Diagnostic 2D patterns</text>',
        f'  <text x="36" y="58" class="note">{escape(receipt_summary)}</text>',
        '  <text x="36" y="76" class="note">Convex-hull previews only; not manufacturing-authorized.</text>',
    ]
    for idx, panel in enumerate(panels):
        row = idx // cols
        col = idx % cols
        origin_x = margin + col * cell_width
        origin_y = margin + 92.0 + row * cell_height
        hull = _convex_hull(panel)
        if len(hull) == 0:
            continue
        min_x, max_x, min_y, max_y = _bounds([hull])
        scale = min((cell_width - 42.0) / (max_x - min_x), (cell_height - 74.0) / (max_y - min_y))
        transformed = _transform_points(
            hull,
            min_x=min_x,
            max_y=max_y,
            scale=scale,
            offset_x=origin_x + 20.0,
            offset_y=origin_y + 30.0,
        )
        color = _panel_color(float(distortions[idx]), threshold)
        grain = grain_directions[idx] if idx < len(grain_directions) else "warp"
        mid_y = origin_y + cell_height - 35.0
        lines.extend(
            [
                (
                    f'  <g id="pattern_panel_{idx}" data-diagnostic="true" '
                    f'data-grain="{escape(grain)}" '
                    f'data-distortion="{float(distortions[idx]):.9f}">'
                ),
                (
                    f'    <polygon class="pattern-outline" points="{_points_attr(transformed)}" '
                    f'fill="{color}" stroke="{color}"/>'
                ),
                (
                    f'    <line class="grain" x1="{origin_x + 28.0:.1f}" y1="{mid_y:.1f}" '
                    f'x2="{origin_x + 88.0:.1f}" y2="{mid_y:.1f}"/>'
                ),
                (
                    f'    <text x="{origin_x + 20.0:.1f}" y="{origin_y + 18.0:.1f}" '
                    f'font-size="12">P{idx} {escape(grain)}</text>'
                ),
                (
                    f'    <text x="{origin_x + 20.0:.1f}" y="{origin_y + cell_height - 14.0:.1f}" '
                    f'class="note">distortion {float(distortions[idx]):.4f}</text>'
                ),
                "  </g>",
            ]
        )
    lines.extend(["</svg>", ""])
    return "\n".join(lines)


def _render_cut_sheet_svg(
    *,
    panels: Sequence[np.ndarray],
    topologies: Sequence[PanelTopology],
    distortions: Sequence[float],
    threshold: float,
    grain_directions: Sequence[str],
    receipt_summary: str,
    seam_graph_summary: SeamGraphSummary,
    realized_operators: Mapping[str, object] | None = None,
    correction_tree_materialization: Mapping[str, object] | None = None,
    selected_backend_per_panel: Sequence[str] | None = None,
) -> str:
    cell_width = 520.0
    cell_height = 420.0
    margin = 36.0
    cols = max(1, min(2, len(panels)))
    rows = int(np.ceil(len(panels) / cols))
    width = margin * 2.0 + cell_width * cols
    height = margin * 2.0 + 98.0 + cell_height * rows
    materialization_status = (
        str(correction_tree_materialization.get("status", "not_present"))
        if correction_tree_materialization is not None
        else "not_present"
    )
    materialized_operator_count = (
        int(correction_tree_materialization.get("materialized_operator_count", 0))
        if correction_tree_materialization is not None
        else 0
    )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
            f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" '
            f'data-correction-tree-status="{escape(materialization_status)}" '
            f'data-materialized-operator-count="{materialized_operator_count}">'
        ),
        "  <title>Diagnostic flat cut sheet</title>",
        "  <style>",
        "    text { font-family: Inter, Arial, sans-serif; fill: #242424; }",
        "    .note { fill: #6b6b6b; font-size: 12px; }",
        "    .tri { fill: none; stroke: #9ca3af; stroke-width: 0.35; opacity: 0.45; }",
        "    .boundary { fill: none; stroke: #111827; stroke-width: 1.6; }",
        "    .seam-segment { stroke: #00a6b8; stroke-width: 2.2; stroke-linecap: round; }",
        "    .grain { stroke: #4b5563; stroke-width: 1.2; stroke-dasharray: 4 3; }",
        "    .stretch-zone { fill: #22c55e; opacity: 0.12; stroke: #15803d; stroke-width: 1; stroke-dasharray: 3 3; }",
        "    .operator-label { fill: #166534; font-size: 11px; font-weight: 600; }",
        "  </style>",
        f'  <rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="#fbfaf7"/>',
        '  <text x="36" y="34" font-size="20" font-weight="700">Diagnostic flat cut sheet</text>',
        f'  <text x="36" y="58" class="note">{escape(receipt_summary)}</text>',
        (
            '  <text x="36" y="76" class="note">Draws UV triangle mesh, true patch '
            "boundary edges, and solver seam edge segments. Not manufacturing-authorized.</text>"
        ),
        (
            f'  <text x="36" y="94" class="note">Seam graph: '
            f"{seam_graph_summary.connected_component_count} component(s), "
            f"{seam_graph_summary.edge_segment_count} edge segment(s), "
            f"{seam_graph_summary.endpoint_count} endpoint(s), "
            f"{seam_graph_summary.branch_vertex_count} branch vertex/vertices.</text>"
        ),
    ]
    realized_nodes = []
    if realized_operators is not None:
        nodes = realized_operators.get("nodes", [])
        if isinstance(nodes, list):
            realized_nodes = [
                node for node in nodes if isinstance(node, Mapping) and node.get("realized")
            ]
    operator_families = sorted({str(node.get("operator")) for node in realized_nodes})
    if operator_families:
        lines.append(
            f'  <text x="36" y="112" class="note">Realized operators: '
            f"{escape(', '.join(operator_families))} "
            f"x{len(realized_nodes)}.</text>"
        )
    for idx, (panel, topology) in enumerate(zip(panels, topologies, strict=False)):
        row = idx // cols
        col = idx % cols
        origin_x = margin + col * cell_width
        origin_y = margin + 116.0 + row * cell_height
        min_x, max_x, min_y, max_y = _bounds([panel])
        scale = min((cell_width - 60.0) / (max_x - min_x), (cell_height - 82.0) / (max_y - min_y))
        transformed = _transform_points(
            panel,
            min_x=min_x,
            max_y=max_y,
            scale=scale,
            offset_x=origin_x + 30.0,
            offset_y=origin_y + 38.0,
        )
        color = _panel_color(float(distortions[idx]), threshold)
        grain = grain_directions[idx] if idx < len(grain_directions) else "warp"
        selected_backend = (
            selected_backend_per_panel[idx]
            if selected_backend_per_panel is not None and idx < len(selected_backend_per_panel)
            else "legacy_unknown"
        )
        local_index = {vertex: local for local, vertex in enumerate(topology.vertices)}
        lines.extend(
            [
                (
                    f'  <g id="cut_sheet_panel_{idx}" data-diagnostic="true" '
                    f'data-distortion="{float(distortions[idx]):.9f}" '
                    f'data-grain="{escape(grain)}" '
                    f'data-face-count="{len(topology.faces)}" '
                    f'data-boundary-edge-count="{len(topology.boundary_edges)}" '
                    f'data-seam-segment-count="{len(topology.seam_edges)}" '
                    f'data-seam-edge-count="{len(topology.seam_edges)}">'
                ),
                (
                    f'    <text x="{origin_x + 24.0:.1f}" y="{origin_y + 20.0:.1f}" '
                    f'font-size="13" font-weight="700">P{idx} {escape(grain)}</text>'
                ),
                (
                    f'    <text x="{origin_x + 24.0:.1f}" y="{origin_y + 36.0:.1f}" '
                    f'class="note">distortion {float(distortions[idx]):.4f}; '
                    f"backend {escape(selected_backend)}; faces {len(topology.faces)}; "
                    f"seam segments {len(topology.seam_edges)}</text>"
                ),
            ]
        )
        if realized_nodes:
            zone_x = origin_x + 42.0
            zone_y = origin_y + 58.0
            zone_w = cell_width - 84.0
            zone_h = max(36.0, cell_height * 0.18)
            operator_label = ", ".join(operator_families)
            lines.extend(
                [
                    (
                        f'    <rect class="stretch-zone" data-role="realized_correction_operator" '
                        f'data-operator="{escape(operator_label)}" x="{zone_x:.1f}" y="{zone_y:.1f}" '
                        f'width="{zone_w:.1f}" height="{zone_h:.1f}"/>'
                    ),
                    (
                        f'    <text class="operator-label" x="{zone_x + 8.0:.1f}" '
                        f'y="{zone_y + 20.0:.1f}">{escape(operator_label)} '
                        f"({len(realized_nodes)} priced node(s))</text>"
                    ),
                ]
            )
        face_step = max(1, len(topology.faces) // 3500)
        for face in topology.faces[::face_step]:
            try:
                tri = np.asarray([transformed[local_index[vertex]] for vertex in face], dtype=float)
            except KeyError:
                continue
            lines.append(f'    <polygon class="tri" points="{_points_attr(tri)}"/>')
        for edge in topology.boundary_edges:
            if edge[0] not in local_index or edge[1] not in local_index:
                continue
            a = transformed[local_index[edge[0]]]
            b = transformed[local_index[edge[1]]]
            lines.append(
                f'    <line class="boundary" x1="{a[0]:.3f}" y1="{a[1]:.3f}" '
                f'x2="{b[0]:.3f}" y2="{b[1]:.3f}"/>'
            )
        for edge in topology.seam_edges:
            if edge[0] not in local_index or edge[1] not in local_index:
                continue
            a = transformed[local_index[edge[0]]]
            b = transformed[local_index[edge[1]]]
            lines.append(
                f'    <line class="seam-segment" data-role="solver_seam_segment" '
                f'x1="{a[0]:.3f}" y1="{a[1]:.3f}" '
                f'x2="{b[0]:.3f}" y2="{b[1]:.3f}"/>'
            )
        mid_y = origin_y + cell_height - 28.0
        lines.extend(
            [
                (
                    f'    <line class="grain" x1="{origin_x + 28.0:.1f}" y1="{mid_y:.1f}" '
                    f'x2="{origin_x + 108.0:.1f}" y2="{mid_y:.1f}"/>'
                ),
                (
                    f'    <text x="{origin_x + 116.0:.1f}" y="{mid_y + 4.0:.1f}" '
                    f'class="note">grainline</text>'
                ),
                (
                    f'    <rect x="{origin_x + 22.0:.1f}" y="{origin_y + 44.0:.1f}" '
                    f'width="{cell_width - 44.0:.1f}" height="{cell_height - 62.0:.1f}" '
                    f'fill="none" stroke="{color}" stroke-width="1" stroke-dasharray="5 4" />'
                ),
                "  </g>",
            ]
        )
    lines.extend(["</svg>", ""])
    return "\n".join(lines)


def _project_vertices(
    vertices: np.ndarray, *, yaw_deg: float = -35.0, elev_deg: float = 14.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = np.deg2rad(float(yaw_deg))
    elev = np.deg2rad(float(elev_deg))
    cos_yaw, sin_yaw = float(np.cos(yaw)), float(np.sin(yaw))
    cos_elev, sin_elev = float(np.cos(elev)), float(np.sin(elev))
    rot_z = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_elev, -sin_elev], [0.0, sin_elev, cos_elev]],
        dtype=float,
    )
    projected = (np.asarray(vertices, dtype=float) @ rot_z.T) @ rot_x.T
    return projected[:, 0], projected[:, 2], projected[:, 1]


def _screen_coords(
    u: np.ndarray,
    v: np.ndarray,
    *,
    width: int,
    height: int,
    margin: float = 70.0,
) -> tuple[np.ndarray, np.ndarray]:
    span = max(float(u.max() - u.min()), float(v.max() - v.min()), 1e-6)
    scale = (min(width, height) - margin * 2.0) / span
    sx = (u - (u.min() + u.max()) * 0.5) * scale + width / 2.0
    sy = height - 1.0 - ((v - (v.min() + v.max()) * 0.5) * scale + height / 2.0)
    return sx, sy


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


def _load_seam_edges(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if "seam_edges" not in payload:
        raise KeyError("Seam edges NPZ must contain 'seam_edges'.")
    edges = np.asarray(payload["seam_edges"], dtype=int)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("seam_edges must be shaped (N, 2).")
    return edges


def _load_face_labels(path: Path) -> np.ndarray | None:
    payload = np.load(path, allow_pickle=False)
    if "face_labels" not in payload:
        return None
    labels = np.asarray(payload["face_labels"], dtype=int)
    if labels.ndim != 1:
        raise ValueError("face_labels must be a one-dimensional array.")
    return labels


def _candidate_correction_paths(
    *,
    panel_receipt_path: Path,
    seam_edges_path: Path | None,
    explicit_path: Path | None,
) -> list[Path]:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    if seam_edges_path is not None:
        candidates.append(seam_edges_path.parent / "corrections.json")
    candidates.extend(
        [
            panel_receipt_path.parent / "corrections.json",
            panel_receipt_path.parent.parent / "solver" / "corrections.json",
        ]
    )
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            deduped.append(candidate)
            seen.add(resolved)
    return deduped


def _load_metric_corrections(
    *,
    panel_receipt_path: Path,
    seam_edges_path: Path | None,
    explicit_path: Path | None,
    expected_hash: str | None,
) -> dict[str, object] | None:
    for candidate in _candidate_correction_paths(
        panel_receipt_path=panel_receipt_path,
        seam_edges_path=seam_edges_path,
        explicit_path=explicit_path,
    ):
        if not candidate.exists():
            continue
        candidate_hash = _sha256_file(candidate)
        if expected_hash is not None and candidate_hash != expected_hash:
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Metric corrections payload must be an object: {candidate}")
        return {
            "path": str(candidate),
            "hash": candidate_hash,
            "payload": payload,
        }
    return None


def _correction_panel_index(
    corrections: Mapping[str, object] | None,
) -> dict[int, dict[str, object]]:
    if corrections is None:
        return {}
    payload = corrections.get("payload")
    if not isinstance(payload, Mapping):
        return {}
    panel_index: dict[int, dict[str, object]] = {}
    raw_panels = payload.get("panels", [])
    if isinstance(raw_panels, list):
        for entry in raw_panels:
            if not isinstance(entry, Mapping):
                continue
            try:
                panel_index[int(entry["panel_label"])] = dict(entry)
            except (KeyError, TypeError, ValueError):
                continue
    selected = payload.get("selected_corrections", [])
    if isinstance(selected, list):
        for entry in selected:
            if not isinstance(entry, Mapping):
                continue
            try:
                label = int(entry["panel_label"])
            except (KeyError, TypeError, ValueError):
                continue
            panel = panel_index.setdefault(label, {"panel_label": label})
            families = panel.setdefault("selected_correction_families", [])
            if isinstance(families, list) and "family" in entry:
                families.append(str(entry["family"]))
    for panel in panel_index.values():
        families = panel.get("selected_correction_families")
        if isinstance(families, list):
            panel["selected_correction_families"] = sorted(set(families))
        else:
            panel["selected_correction_families"] = []
    return panel_index


def _float_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _panel_competition_index(receipt: object) -> dict[int, Mapping[str, object]]:
    competition = getattr(receipt, "serialization_competition_receipt", None)
    if not isinstance(competition, Mapping):
        return {}
    panels = competition.get("panels", [])
    if not isinstance(panels, list):
        return {}
    indexed: dict[int, Mapping[str, object]] = {}
    for idx, entry in enumerate(panels):
        if isinstance(entry, Mapping):
            indexed[idx] = entry
    return indexed


def _candidate_score_summary(competition: Mapping[str, object] | None) -> list[dict[str, object]]:
    if competition is None:
        return []
    raw_candidates = competition.get("candidates", [])
    if not isinstance(raw_candidates, list):
        return []
    summaries: list[dict[str, object]] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, Mapping):
            continue
        summaries.append(
            {
                "backend": str(raw_candidate.get("backend", "unknown")),
                "diagnostic_only": bool(raw_candidate.get("diagnostic_only", False)),
                "available": bool(raw_candidate.get("available", False)),
                "distortion": _float_or_none(raw_candidate.get("distortion")),
                "foldovers": raw_candidate.get("foldovers"),
                "boundary_deviation": _float_or_none(raw_candidate.get("boundary_deviation")),
                "score": _float_or_none(raw_candidate.get("score")),
                "promoted": bool(raw_candidate.get("promoted", False)),
                "blockers": (
                    list(raw_candidate.get("blockers", []))
                    if isinstance(raw_candidate.get("blockers"), list)
                    else []
                ),
            }
        )
    return summaries


def _correction_tree_materialization_summary(receipt: object) -> dict[str, object]:
    materialization = getattr(receipt, "correction_tree_materialization_receipt", None)
    if isinstance(materialization, Mapping):
        entries = materialization.get("materializations", [])
        chart_materialized = 0
        if isinstance(entries, list):
            chart_materialized = sum(
                1
                for entry in entries
                if isinstance(entry, Mapping) and bool(entry.get("chart_materialized", False))
            )
        promotion = int(materialization.get("promotion", 0))
        return {
            "schema_version": str(materialization.get("schema_version", "")),
            "status": "materialized" if promotion == 1 else "partial",
            "promotion": promotion,
            "materialized_operator_count": int(
                materialization.get("materialized_operator_count", 0)
            ),
            "chart_materialized_operator_count": chart_materialized,
            "blockers": (
                list(materialization.get("blockers", []))
                if isinstance(materialization.get("blockers"), list)
                else []
            ),
        }
    correction_tree = getattr(receipt, "correction_tree_receipt", None)
    realized = getattr(receipt, "realized_correction_operator_receipt", None)
    if not isinstance(correction_tree, Mapping):
        return {"status": "not_present", "correction_tree_present": False}
    branch_count = int(correction_tree.get("branch_count", 0))
    realized_count = (
        int(realized.get("realized_operator_count", 0)) if isinstance(realized, Mapping) else 0
    )
    unrealized_count = (
        int(realized.get("unrealized_operator_count", 0)) if isinstance(realized, Mapping) else 0
    )
    if branch_count == 0:
        status = "no_tree_nodes"
    elif not isinstance(realized, Mapping) or realized_count == 0:
        status = "unmaterialized"
    elif int(realized.get("promotion", 0)) == 1 and unrealized_count == 0:
        status = "materialized"
    else:
        status = "partial"
    return {
        "status": status,
        "correction_tree_present": True,
        "correction_tree_promotion": int(correction_tree.get("promotion", 0)),
        "branch_count": branch_count,
        "typed_branch_count": int(correction_tree.get("typed_branch_count", 0)),
        "diagnostic_branch_count": int(correction_tree.get("diagnostic_branch_count", 0)),
        "realized_operator_present": isinstance(realized, Mapping),
        "realization_promotion": int(realized.get("promotion", 0))
        if isinstance(realized, Mapping)
        else 0,
        "realized_operator_count": realized_count,
        "unrealized_operator_count": unrealized_count,
    }


def _mesh_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for a, b, c in np.asarray(faces, dtype=int):
        for u, v in ((a, b), (b, c), (c, a)):
            uu = int(u)
            vv = int(v)
            if uu == vv:
                continue
            edges.add((uu, vv) if uu < vv else (vv, uu))
    return np.asarray(sorted(edges), dtype=int)


def _render_mesh_overlay_png(
    *,
    mesh_path: Path,
    seam_edges_path: Path,
    output_path: Path,
    width: int = 1200,
    height: int = 900,
) -> dict[str, object]:
    vertices, faces = _load_mesh(mesh_path)
    seam_edges = _load_seam_edges(seam_edges_path)
    if seam_edges.size and int(seam_edges.max()) >= len(vertices):
        raise ValueError("Seam edge index is outside the mesh vertex range.")

    u, v, depth = _project_vertices(vertices)
    sx, sy = _screen_coords(u, v, width=width, height=height)
    image = Image.new("RGB", (width, height), (250, 248, 245))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((36, 30), "3D mesh seam overlay", fill=(36, 36, 36, 255))
    draw.text(
        (36, 50),
        "Diagnostic render of promoted Afflec mesh + solver seam edges",
        fill=(92, 92, 92, 255),
    )

    face_depth = depth[faces].mean(axis=1)
    face_order = np.argsort(face_depth)
    face_step = max(1, len(face_order) // 6500)
    sampled_face_count = int(len(face_order[::face_step]))
    for face_idx in face_order[::face_step]:
        tri = [
            (float(sx[int(vertex)]), float(sy[int(vertex)]))
            for vertex in np.asarray(faces[int(face_idx)], dtype=int)
        ]
        draw.polygon(tri, fill=(210, 208, 202, 28))

    mesh_edges = _mesh_edges_from_faces(faces)
    order = np.argsort((depth[mesh_edges[:, 0]] + depth[mesh_edges[:, 1]]) * 0.5)
    step = max(1, len(order) // 14000)
    sampled_mesh_edge_count = int(len(order[::step]))
    for edge_idx in order[::step]:
        a, b = mesh_edges[int(edge_idx)]
        draw.line(
            (float(sx[a]), float(sy[a]), float(sx[b]), float(sy[b])),
            fill=(70, 70, 70, 34),
            width=1,
        )

    vertex_order = np.argsort(depth)
    vertex_step = max(1, len(vertex_order) // 8000)
    sampled_vertex_count = int(len(vertex_order[::vertex_step]))
    for idx in vertex_order[::vertex_step]:
        x = float(sx[idx])
        y = float(sy[idx])
        draw.ellipse((x - 1.0, y - 1.0, x + 1.0, y + 1.0), fill=(60, 60, 60, 48))

    for a_raw, b_raw in seam_edges:
        a = int(a_raw)
        b = int(b_raw)
        draw.line(
            (float(sx[a]), float(sy[a]), float(sx[b]), float(sy[b])),
            fill=(0, 190, 210, 245),
            width=4,
        )
        for idx in (a, b):
            x = float(sx[idx])
            y = float(sy[idx])
            draw.ellipse((x - 3.0, y - 3.0, x + 3.0, y + 3.0), fill=(0, 115, 130, 190))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        "mesh_path": str(mesh_path),
        "mesh_hash": _sha256_file(mesh_path),
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "render_mode": "sampled_shaded_faces_plus_wire_overlay",
        "sampled_face_count": sampled_face_count,
        "sampled_mesh_edge_count": sampled_mesh_edge_count,
        "sampled_vertex_count": sampled_vertex_count,
        "render_warning": (
            "Apparent blank regions can come from sampled diagnostic rendering; "
            "use face_count and cut_sheet panel_face_counts for topology coverage."
        ),
        "seam_edges_path": str(seam_edges_path),
        "seam_edges_hash": _sha256_file(seam_edges_path),
        "seam_edge_count": int(seam_edges.shape[0]),
        "output": str(output_path),
        "output_hash": _sha256_file(output_path),
    }


def _render_png_from_patterns(
    *,
    panels: Sequence[np.ndarray],
    distortions: Sequence[float],
    threshold: float,
    output_path: Path,
) -> None:
    width = 1100
    height = 760
    margin = 60.0
    min_x, max_x, min_y, max_y = _bounds(panels)
    scale = min((width - margin * 2.0) / (max_x - min_x), (height - 140.0) / (max_y - min_y))
    image = Image.new("RGB", (width, height), (251, 250, 247))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((36, 28), "Raw panel UV point-cloud diagnostic", fill=(36, 36, 36, 255))
    draw.text(
        (36, 48),
        "Not panel topology; open diagnostic_flat_cut_sheet.svg for panel/seam review.",
        fill=(92, 92, 92, 255),
    )
    for idx, panel in enumerate(panels):
        transformed = _transform_points(
            panel,
            min_x=min_x,
            max_y=max_y,
            scale=scale,
            offset_x=margin,
            offset_y=95.0,
        )
        color = _panel_color(float(distortions[idx]), threshold)
        rgb = tuple(int(color.lstrip("#")[pos : pos + 2], 16) for pos in (0, 2, 4))
        hull = _convex_hull(transformed)
        if len(hull) >= 2:
            polygon = [(float(x), float(y)) for x, y in hull]
            draw.polygon(polygon, fill=(*rgb, 28), outline=(*rgb, 255))
        sample_step = max(1, len(transformed) // 1800)
        for x, y in transformed[::sample_step]:
            draw.ellipse((x - 1.1, y - 1.1, x + 1.1, y + 1.1), fill=(*rgb, 80))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_panel_patterns(
    *,
    panel_receipt_path: Path,
    panel_uvs_path: Path,
    output_dir: Path,
    mesh_path: Path | None = None,
    seam_edges_path: Path | None = None,
    corrections_path: Path | None = None,
) -> dict[str, object]:
    receipt = load_panel_unwrap_receipt(panel_receipt_path)
    uv_hash = _sha256_file(panel_uvs_path)
    if receipt.uv_hash != uv_hash:
        raise ValueError("Panel UV hash does not match PanelUnwrapReceipt.uv_hash.")
    panels = _load_panel_uvs(panel_uvs_path, receipt.panel_count)
    output_dir.mkdir(parents=True, exist_ok=True)

    blockers = receipt.panel_unwrap_blockers or receipt.blocked_consumers
    selected_backend_per_panel = receipt.selected_backend_per_panel or []
    correction_tree_materialization = _correction_tree_materialization_summary(receipt)
    receipt_summary = (
        f"promotion={receipt.promotion}; backend={receipt.unwrap_backend or 'legacy_unknown'}; "
        f"margin={receipt.distortion_margin}; blockers={','.join(blockers)}"
    )
    uv_svg_path = output_dir / "panel_uv_diagnostic.svg"
    uv_png_path = output_dir / "panel_uv_diagnostic.png"
    pattern_svg_path = output_dir / "diagnostic_2d_patterns.svg"
    cut_sheet_svg_path = output_dir / "diagnostic_flat_cut_sheet.svg"
    mesh_overlay_path = output_dir / "mesh_seam_overlay.png"
    summary_path = output_dir / "diagnostic_pattern_summary.json"
    distortions = receipt.per_panel_distortion
    corrections = _load_metric_corrections(
        panel_receipt_path=panel_receipt_path,
        seam_edges_path=seam_edges_path,
        explicit_path=corrections_path,
        expected_hash=receipt.correction_payload_hash,
    )
    correction_panels = _correction_panel_index(corrections)
    competition_panels = _panel_competition_index(receipt)

    uv_svg_path.write_text(
        _render_uv_svg(
            panels=panels,
            distortions=distortions,
            threshold=receipt.distortion_threshold,
            receipt_summary=receipt_summary,
        ),
        encoding="utf-8",
    )
    pattern_svg_path.write_text(
        _render_patterns_svg(
            panels=panels,
            distortions=distortions,
            threshold=receipt.distortion_threshold,
            grain_directions=receipt.grain_directions,
            receipt_summary=receipt_summary,
        ),
        encoding="utf-8",
    )
    _render_png_from_patterns(
        panels=panels,
        distortions=distortions,
        threshold=receipt.distortion_threshold,
        output_path=uv_png_path,
    )
    mesh_overlay: dict[str, object] | None = None
    cut_sheet: dict[str, object] | None = None
    topologies: list[PanelTopology] | None = None
    seam_graph_summary: SeamGraphSummary | None = None
    topology_source = "unavailable_without_mesh_and_seam_edges"
    if mesh_path is not None and seam_edges_path is not None:
        vertices, faces = _load_mesh(mesh_path)
        seam_edges = _load_seam_edges(seam_edges_path)
        face_labels = _load_face_labels(seam_edges_path)
        seam_graph_summary = _seam_graph_summary(
            [_normalize_edge(int(a), int(b)) for a, b in seam_edges]
        )
        if face_labels is not None:
            topology_source = "cut_graph_face_labels"
            topologies = _extract_panel_topologies_from_face_labels(
                faces=faces,
                seam_edges=seam_edges,
                face_labels=face_labels,
            )
        else:
            topology_source = "mesh_components_after_solver_seams"
            topologies = _extract_panel_topologies(
                vertex_count=int(vertices.shape[0]),
                faces=faces,
                seam_edges=seam_edges,
            )
        if len(topologies) == len(panels):
            boundary_counts = [len(topology.boundary_edges) for topology in topologies]
            seam_segment_counts = [len(topology.seam_edges) for topology in topologies]
            cut_sheet_warnings: list[str] = []
            if receipt.promotion != 1:
                cut_sheet_warnings.append("panel_unwrap_not_promoted")
            if len(topologies) == 1:
                cut_sheet_warnings.append("single_panel_cut_sheet")
            if len(topologies) <= 1 or not all(count > 0 for count in boundary_counts):
                cut_sheet_warnings.append("seam_graph_not_cut_graph")
            if any(count == 0 for count in boundary_counts):
                cut_sheet_warnings.append("no_patch_boundary_edges")
                cut_sheet_warnings.append("no_cut_mesh_boundary")
            if (
                seam_graph_summary.endpoint_count != 0
                or seam_graph_summary.branch_vertex_count != 0
            ):
                cut_sheet_warnings.append("open_or_branched_seam_graph")
            if not any(count > 0 for count in seam_segment_counts):
                cut_sheet_warnings.append("no_solver_seam_segments")
            cut_sheet_svg_path.write_text(
                _render_cut_sheet_svg(
                    panels=panels,
                    topologies=topologies,
                    distortions=distortions,
                    threshold=receipt.distortion_threshold,
                    grain_directions=receipt.grain_directions,
                    receipt_summary=receipt_summary,
                    seam_graph_summary=seam_graph_summary,
                    realized_operators=receipt.realized_correction_operator_receipt,
                    correction_tree_materialization=correction_tree_materialization,
                    selected_backend_per_panel=selected_backend_per_panel,
                ),
                encoding="utf-8",
            )
            cut_sheet = {
                "output": str(cut_sheet_svg_path),
                "output_hash": _sha256_file(cut_sheet_svg_path),
                "panel_face_counts": [len(topology.faces) for topology in topologies],
                "panel_boundary_edge_counts": boundary_counts,
                "panel_seam_segment_counts": seam_segment_counts,
                "panel_cut_edge_counts": seam_segment_counts,
                "deprecated_fields": {
                    "panel_cut_edge_counts": "Use panel_seam_segment_counts; these are solver seam edge segments, not independent cuts."
                },
                "seam_graph_summary": {
                    "edge_segment_count": seam_graph_summary.edge_segment_count,
                    "vertex_count": seam_graph_summary.vertex_count,
                    "connected_component_count": seam_graph_summary.connected_component_count,
                    "endpoint_count": seam_graph_summary.endpoint_count,
                    "branch_vertex_count": seam_graph_summary.branch_vertex_count,
                    "largest_component_edge_count": (
                        seam_graph_summary.largest_component_edge_count
                    ),
                },
                "cut_sheet_warnings": cut_sheet_warnings,
                "realized_correction_operator_receipt": receipt.realized_correction_operator_receipt,
            }
        mesh_overlay = _render_mesh_overlay_png(
            mesh_path=mesh_path,
            seam_edges_path=seam_edges_path,
            output_path=mesh_overlay_path,
        )

    panel_summaries: list[dict[str, object]] = []
    corrected_residuals = receipt.per_panel_corrected_residual
    for idx in range(receipt.panel_count):
        topology = topologies[idx] if topologies is not None and idx < len(topologies) else None
        correction_panel = correction_panels.get(idx, {})
        competition_panel = competition_panels.get(idx)
        corrected_residual = (
            corrected_residuals[idx]
            if corrected_residuals is not None and idx < len(corrected_residuals)
            else _float_or_none(correction_panel.get("corrected_metric_residual"))
        )
        selected_families = correction_panel.get("selected_correction_families", [])
        panel_summaries.append(
            {
                "panel_label": idx,
                "face_count": len(topology.faces) if topology is not None else None,
                "boundary_edge_count": (
                    len(topology.boundary_edges) if topology is not None else None
                ),
                "seam_segment_count": len(topology.seam_edges) if topology is not None else None,
                "raw_uv_distortion": float(distortions[idx]),
                "selected_backend": (
                    selected_backend_per_panel[idx]
                    if idx < len(selected_backend_per_panel)
                    else receipt.unwrap_backend or "legacy_unknown"
                ),
                "serialization_candidates": _candidate_score_summary(competition_panel),
                "raw_metric_residual": _float_or_none(correction_panel.get("raw_metric_residual")),
                "corrected_metric_residual": corrected_residual,
                "selected_correction_families": (
                    list(selected_families) if isinstance(selected_families, list) else []
                ),
            }
        )

    manufacturing_blockers = sorted(
        set(list(blockers) + list(cut_sheet.get("cut_sheet_warnings", []) if cut_sheet else []))
    )
    solver_mode = (
        "metric_panelization"
        if corrections is not None or receipt.correction_payload_hash is not None
        else "unknown_from_panel_receipt"
    )
    correction_payload = corrections.get("payload") if corrections is not None else None
    correction_source = (
        {
            "path": corrections["path"],
            "hash": corrections["hash"],
            "families": (
                list(correction_payload.get("families", []))
                if isinstance(correction_payload, Mapping)
                else []
            ),
            "selected_count": (
                int(correction_payload.get("selected_count", 0))
                if isinstance(correction_payload, Mapping)
                else 0
            ),
        }
        if corrections is not None
        else (
            {
                "path": None,
                "hash": receipt.correction_payload_hash,
                "families": [],
                "selected_count": None,
                "status": "referenced_payload_not_found",
            }
            if receipt.correction_payload_hash is not None
            else None
        )
    )

    summary: dict[str, object] = {
        "diagnostic_only": True,
        "manufacturing_authorized": False,
        "manufacturing_blocked_because": manufacturing_blockers,
        "panel_unwrap_promotion": int(receipt.promotion),
        "panel_unwrap_blockers": list(blockers),
        "panel_count": int(receipt.panel_count),
        "panels": panel_summaries,
        "worst_panel_distortion": float(receipt.worst_panel_distortion),
        "distortion_threshold": float(receipt.distortion_threshold),
        "distortion_margin": receipt.distortion_margin,
        "serialization_promoted": receipt.serialization_promoted,
        "selected_backend_per_panel": selected_backend_per_panel,
        "serialization_competition_receipt": receipt.serialization_competition_receipt,
        "realized_correction_operator_receipt": receipt.realized_correction_operator_receipt,
        "correction_tree_materialization": correction_tree_materialization,
        "correction_tree_materialization_receipt": (
            receipt.correction_tree_materialization_receipt
        ),
        "panel_receipt_hash": _sha256_file(panel_receipt_path),
        "panel_uv_hash": uv_hash,
        "provenance": {
            "solver_mode": solver_mode,
            "topology_source": topology_source,
            "unwrap_backend": receipt.unwrap_backend or "legacy_unknown",
            "selected_backend_per_panel": selected_backend_per_panel,
            "correction_source": correction_source,
            "manufacturing_authorized": False,
            "manufacturing_blocked_because": manufacturing_blockers,
        },
        "artifact_hierarchy": [
            {
                "name": "mesh_seam_overlay.png",
                "role": "where_solver_seams_landed_on_body",
                "primary": False,
                "path": str(mesh_overlay_path) if mesh_overlay is not None else None,
            },
            {
                "name": "panel_uv_diagnostic.svg",
                "role": "raw_uv_point_cloud_not_panel_topology",
                "primary": False,
                "topology_backed": False,
                "path": str(uv_svg_path),
            },
            {
                "name": "panel_uv_diagnostic.png",
                "role": "raw_uv_point_cloud_not_panel_topology",
                "primary": False,
                "topology_backed": False,
                "path": str(uv_png_path),
            },
            {
                "name": "diagnostic_flat_cut_sheet.svg",
                "role": "primary_face_backed_cut_sheet_review_artifact",
                "primary": cut_sheet is not None,
                "topology_backed": cut_sheet is not None,
                "path": str(cut_sheet_svg_path) if cut_sheet is not None else None,
            },
            {
                "name": "diagnostic_2d_patterns.svg",
                "role": "legacy_coarse_hull_preview",
                "primary": False,
                "deprecated": True,
                "path": str(pattern_svg_path),
            },
        ],
        "outputs": {
            "uv_svg": str(uv_svg_path),
            "uv_png": str(uv_png_path),
            "patterns_svg": str(pattern_svg_path),
        },
        "output_roles": {
            "uv_svg": "raw_uv_point_cloud_not_panel_topology",
            "uv_png": "raw_uv_point_cloud_not_panel_topology",
            "patterns_svg": "legacy_coarse_hull_preview_deprecated",
        },
        "visual_review_guidance": {
            "primary_panel_review_artifact": (
                str(cut_sheet_svg_path) if cut_sheet is not None else None
            ),
            "raw_uv_diagnostic_is_panel_topology": False,
            "raw_uv_diagnostic_note": (
                "panel_uv_diagnostic shows sampled UV coordinates and convex hulls only; "
                "it intentionally does not prove or display face-backed panel topology."
            ),
        },
        "output_hashes": {
            "uv_svg": _sha256_file(uv_svg_path),
            "uv_png": _sha256_file(uv_png_path),
            "patterns_svg": _sha256_file(pattern_svg_path),
        },
    }
    if cut_sheet is not None:
        outputs = summary["outputs"]
        output_hashes = summary["output_hashes"]
        if isinstance(outputs, dict):
            outputs["cut_sheet_svg"] = str(cut_sheet_svg_path)
        if isinstance(output_hashes, dict):
            output_hashes["cut_sheet_svg"] = _sha256_file(cut_sheet_svg_path)
        output_roles = summary["output_roles"]
        if isinstance(output_roles, dict):
            output_roles["cut_sheet_svg"] = "primary_face_backed_cut_sheet_review_artifact"
        summary["cut_sheet"] = cut_sheet
    if mesh_overlay is not None:
        outputs = summary["outputs"]
        output_hashes = summary["output_hashes"]
        if isinstance(outputs, dict):
            outputs["mesh_overlay_png"] = str(mesh_overlay_path)
        if isinstance(output_hashes, dict):
            output_hashes["mesh_overlay_png"] = _sha256_file(mesh_overlay_path)
        output_roles = summary["output_roles"]
        if isinstance(output_roles, dict):
            output_roles["mesh_overlay_png"] = "where_solver_seams_landed_on_body"
        summary["mesh_overlay"] = mesh_overlay
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote diagnostic UV SVG to {uv_svg_path}")
    print(f"Wrote diagnostic UV PNG to {uv_png_path}")
    print(f"Wrote diagnostic 2D patterns to {pattern_svg_path}")
    if cut_sheet is not None:
        print(f"Wrote diagnostic flat cut sheet to {cut_sheet_svg_path}")
    if mesh_overlay is not None:
        print(f"Wrote diagnostic 3D mesh overlay to {mesh_overlay_path}")
    print(f"Wrote diagnostic pattern summary to {summary_path}")
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-receipt", type=Path, required=True)
    parser.add_argument("--panel-uvs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--seam-edges", type=Path, default=None)
    parser.add_argument(
        "--corrections",
        type=Path,
        default=None,
        help="Optional metric corrections JSON emitted by metric_panelization.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    render_panel_patterns(
        panel_receipt_path=args.panel_receipt,
        panel_uvs_path=args.panel_uvs,
        output_dir=args.out_dir,
        mesh_path=args.mesh,
        seam_edges_path=args.seam_edges,
        corrections_path=args.corrections,
    )


if __name__ == "__main__":
    main()
