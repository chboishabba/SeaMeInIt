#!/usr/bin/env python3
"""Validate solver seam segments as an admissible cut topology."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from smii.seams import CutTopologyReceipt, CutTopologyRepairReceipt, load_solver_promotion_receipt
from smii.seams.panel_serialization_competition import panel_chart_diagnostics

Edge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class SeamGraphSummary:
    edge_segment_count: int
    vertex_count: int
    connected_component_count: int
    endpoint_count: int
    branch_vertex_count: int


@dataclass(frozen=True, slots=True)
class PanelTopologySummary:
    panel_count: int
    face_counts: list[int]
    boundary_edge_counts: list[int]
    panels_are_disks: bool


@dataclass(frozen=True, slots=True)
class SeamGraphClassification:
    ordinary_boundary_component_count: int
    typed_operator_count: int
    invalid_fragmentation_count: int
    classifications: list[str]


@dataclass(frozen=True, slots=True)
class ChartPanel:
    vertices: tuple[int, ...]
    edges: tuple[Edge, ...]
    faces: tuple[tuple[int, int, int], ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_correction_family_counts(path: Path | None) -> Counter[str]:
    if path is None:
        return Counter()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Correction payload must contain a JSON object: {path}")
    selected = payload.get("selected_corrections", [])
    if not isinstance(selected, list):
        return Counter()
    counts: Counter[str] = Counter()
    for entry in selected:
        if not isinstance(entry, Mapping):
            continue
        family = str(entry.get("family", ""))
        if bool(entry.get("selected", True)):
            counts[family] += 1
    return counts


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


def _load_seam_edges(path: Path) -> tuple[Edge, ...]:
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
    normalized = tuple(sorted({_normalize_edge(a, b) for a, b in seam_edges}))
    if not normalized:
        return SeamGraphSummary(0, 0, 0, 0, 0)
    graph = _adjacency(normalized)
    vertices = set(graph)
    remaining = set(vertices)
    component_count = 0
    while remaining:
        component_count += 1
        start = min(remaining)
        remaining.remove(start)
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(graph.get(node, ())):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return SeamGraphSummary(
        edge_segment_count=len(normalized),
        vertex_count=len(vertices),
        connected_component_count=component_count,
        endpoint_count=sum(1 for vertex in vertices if len(graph[vertex]) == 1),
        branch_vertex_count=sum(1 for vertex in vertices if len(graph[vertex]) > 2),
    )


def _panel_topology_summary(
    *,
    vertex_count: int,
    faces: np.ndarray,
    seam_edges: Sequence[Edge],
) -> PanelTopologySummary:
    seam_edge_set = set(seam_edges)
    face_tuples = [tuple(int(v) for v in face) for face in np.asarray(faces, dtype=int)]
    edge_to_faces: dict[Edge, list[int]] = defaultdict(list)
    for face_idx, face in enumerate(face_tuples):
        for edge in _face_edges(face):
            edge_to_faces[edge].append(face_idx)

    face_graph: dict[int, set[int]] = defaultdict(set)
    for edge, face_indices in edge_to_faces.items():
        if edge in seam_edge_set:
            continue
        if len(face_indices) != 2:
            continue
        a, b = face_indices
        face_graph[a].add(b)
        face_graph[b].add(a)

    face_counts: list[int] = []
    boundary_counts: list[int] = []
    panels_are_disks = True
    remaining_faces = set(range(len(face_tuples)))
    while remaining_faces:
        start = min(remaining_faces)
        remaining_faces.remove(start)
        component_face_indices = {start}
        queue: deque[int] = deque([start])
        while queue:
            face_idx = queue.popleft()
            for neighbor in sorted(face_graph.get(face_idx, ())):
                if neighbor in remaining_faces:
                    remaining_faces.remove(neighbor)
                    component_face_indices.add(neighbor)
                    queue.append(neighbor)
        component_faces = tuple(face_tuples[idx] for idx in sorted(component_face_indices))
        component_vertices = {vertex for face in component_faces for vertex in face}
        component_edges = {
            edge
            for face in component_faces
            for edge in _face_edges(face)
            if edge not in seam_edge_set
        }
        face_edge_counts = Counter(edge for face in component_faces for edge in _face_edges(face))
        face_counts.append(len(component_faces))
        boundary_counts.append(sum(1 for count in face_edge_counts.values() if count == 1))
        chi = len(component_vertices) - len(component_edges) + len(component_faces)
        if chi < 1:
            panels_are_disks = False
    return PanelTopologySummary(
        panel_count=len(face_counts),
        face_counts=face_counts,
        boundary_edge_counts=boundary_counts,
        panels_are_disks=panels_are_disks and bool(face_counts),
    )


def _panel_chart_domains(
    *,
    faces: np.ndarray,
    seam_edges: Sequence[Edge],
) -> list[ChartPanel]:
    seam_edge_set = set(seam_edges)
    face_tuples = [
        (int(face[0]), int(face[1]), int(face[2])) for face in np.asarray(faces, dtype=int)
    ]
    edge_to_faces: dict[Edge, list[int]] = defaultdict(list)
    for face_idx, face in enumerate(face_tuples):
        for edge in _face_edges(face):
            edge_to_faces[edge].append(face_idx)

    face_graph: dict[int, set[int]] = defaultdict(set)
    for edge, face_indices in edge_to_faces.items():
        if edge in seam_edge_set or len(face_indices) != 2:
            continue
        a, b = face_indices
        face_graph[a].add(b)
        face_graph[b].add(a)

    panels: list[ChartPanel] = []
    remaining_faces = set(range(len(face_tuples)))
    while remaining_faces:
        start = min(remaining_faces)
        remaining_faces.remove(start)
        component_face_indices = {start}
        queue: deque[int] = deque([start])
        while queue:
            face_idx = queue.popleft()
            for neighbor in sorted(face_graph.get(face_idx, ())):
                if neighbor in remaining_faces:
                    remaining_faces.remove(neighbor)
                    component_face_indices.add(neighbor)
                    queue.append(neighbor)
        component_faces = tuple(face_tuples[idx] for idx in sorted(component_face_indices))
        component_vertices = tuple(sorted({vertex for face in component_faces for vertex in face}))
        component_edges = tuple(
            sorted({edge for face in component_faces for edge in _face_edges(face)})
        )
        panels.append(
            ChartPanel(
                vertices=component_vertices,
                edges=component_edges,
                faces=component_faces,
            )
        )
    return panels


def _cut_topology_repair_receipt(
    *,
    cut_topology_receipt_path: Path,
    mesh_path: Path,
    seam_edges_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    seam_edges: Sequence[Edge],
    topology_blockers: Sequence[str],
) -> CutTopologyRepairReceipt:
    panels = _panel_chart_domains(faces=faces, seam_edges=seam_edges)
    panel_checks: list[dict[str, object]] = []
    repair_blockers = list(topology_blockers)
    for idx, panel in enumerate(panels):
        diagnostics = panel_chart_diagnostics(vertices, panel)
        blockers = [str(blocker) for blocker in diagnostics.get("blockers", [])]
        repair_blockers.extend(blockers)
        panel_checks.append(
            {
                "panel_id": f"P{idx}",
                "face_count": len(panel.faces),
                "vertex_count": len(panel.vertices),
                "connected_components": diagnostics["connected_components"],
                "nonmanifold_edges": diagnostics["nonmanifold_edges"],
                "boundary_loops": diagnostics["boundary_loops"],
                "oriented_faces": diagnostics["oriented_faces"],
                "duplicate_faces": diagnostics["duplicate_faces"],
                "degenerate_triangles": diagnostics["degenerate_triangles"],
                "isolated_vertices": diagnostics["isolated_vertices"],
                "backend_serializable": diagnostics["backend_serializable"],
                "blockers": blockers,
            }
        )
    repair_blockers = list(dict.fromkeys(str(blocker) for blocker in repair_blockers))
    if any(not bool(check["backend_serializable"]) for check in panel_checks):
        repair_blockers.append("chart_domain_not_backend_serializable")
    repair_blockers = list(dict.fromkeys(repair_blockers))
    return CutTopologyRepairReceipt(
        input_cut_topology_hash=_sha256_file(cut_topology_receipt_path),
        mesh_hash=_sha256_file(mesh_path),
        seam_edges_hash=_sha256_file(seam_edges_path),
        panel_count=len(panel_checks),
        panel_checks=panel_checks,
        repairs=[],
        promotion=1 if not repair_blockers else 0,
        blocked_consumers=[],
        repair_blockers=repair_blockers,
    )


def _mesh_boundary_vertices(faces: np.ndarray) -> set[int]:
    face_tuples = [tuple(int(v) for v in face) for face in np.asarray(faces, dtype=int)]
    edge_counts = Counter(edge for face in face_tuples for edge in _face_edges(face))
    return {vertex for edge, count in edge_counts.items() if count == 1 for vertex in edge}


def _topology_blockers(
    seam_summary: SeamGraphSummary,
    panel_summary: PanelTopologySummary,
    *,
    seam_edges: Sequence[Edge],
    faces: np.ndarray,
    classification: SeamGraphClassification,
) -> list[str]:
    blockers: list[str] = []
    if seam_summary.edge_segment_count == 0:
        blockers.append("no_seam_segments")
    graph = _adjacency(seam_edges)
    seam_endpoints = {vertex for vertex, neighbors in graph.items() if len(neighbors) == 1}
    boundary_vertices = _mesh_boundary_vertices(faces)
    open_endpoint_count = len(seam_endpoints - boundary_vertices)
    untyped_branch_count = max(
        0,
        seam_summary.branch_vertex_count - classification.typed_operator_count,
    )
    if untyped_branch_count > 0 and classification.invalid_fragmentation_count != 0:
        blockers.append("untyped_branch_operator")
    if open_endpoint_count != 0 and classification.invalid_fragmentation_count != 0:
        blockers.append("unresolved_open_boundary")
    if classification.invalid_fragmentation_count != 0:
        blockers.append("panel_fragmentation_invalid")
    if panel_summary.panel_count <= 1 or not all(
        count > 0 for count in panel_summary.boundary_edge_counts
    ):
        blockers.append("seam_graph_not_cut_graph")
    if any(count == 0 for count in panel_summary.boundary_edge_counts):
        blockers.append("no_cut_mesh_boundary")
    if not panel_summary.panels_are_disks:
        blockers.append("panels_not_disks")
    return list(dict.fromkeys(blockers))


def _seam_graph_classification(
    *,
    seam_edges: Sequence[Edge],
    faces: np.ndarray,
    authorized_operator_count: int,
) -> SeamGraphClassification:
    graph = _adjacency(seam_edges)
    boundary_vertices = _mesh_boundary_vertices(faces)
    component_labels: list[str] = []
    typed_remaining = max(0, int(authorized_operator_count))
    for component in _connected_components_from_edges(seam_edges):
        branch_count = sum(1 for vertex in component if len(graph.get(vertex, ())) > 2)
        interior_endpoint_count = sum(
            1
            for vertex in component
            if len(graph.get(vertex, ())) == 1 and vertex not in boundary_vertices
        )
        if interior_endpoint_count == 0 and branch_count == 0:
            component_labels.append("ordinary_boundary")
        elif typed_remaining > 0:
            component_labels.append("typed_correction_operator")
            typed_remaining -= 1
        else:
            component_labels.append("invalid_fragmentation")
    return SeamGraphClassification(
        ordinary_boundary_component_count=sum(
            1 for label in component_labels if label == "ordinary_boundary"
        ),
        typed_operator_count=sum(
            1 for label in component_labels if label == "typed_correction_operator"
        ),
        invalid_fragmentation_count=sum(
            1 for label in component_labels if label == "invalid_fragmentation"
        ),
        classifications=component_labels,
    )


def _connected_components_from_edges(edges: Sequence[Edge]) -> list[set[int]]:
    if not edges:
        return []
    graph = _adjacency(edges)
    remaining = set(graph)
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


def validate_cut_topology(
    *,
    solver_receipt_path: Path,
    seam_edges_path: Path,
    mesh_path: Path,
    receipt_path: Path,
    repair_receipt_path: Path | None = None,
    typed_dart_count: int = 0,
    typed_gusset_count: int = 0,
    typed_relief_cut_count: int = 0,
    typed_ease_count: int = 0,
    typed_stretch_zone_count: int = 0,
    corrections_path: Path | None = None,
) -> CutTopologyReceipt:
    """Validate cut topology and emit a hash-linked receipt."""

    solver_receipt = load_solver_promotion_receipt(solver_receipt_path)
    if solver_receipt.promotion != 1:
        raise ValueError(
            f"SolverPromotionReceipt not promoted ({solver_receipt.promotion}). "
            f"Blocked: {solver_receipt.blocked_consumers}"
        )
    seam_edges_hash = _sha256_file(seam_edges_path)
    if solver_receipt.seam_hash != seam_edges_hash:
        raise ValueError("Seam edges hash does not match SolverPromotionReceipt.seam_hash.")
    vertices, faces = _load_mesh(mesh_path)
    seam_edges = _load_seam_edges(seam_edges_path)
    if seam_edges and max(max(edge) for edge in seam_edges) >= len(vertices):
        raise ValueError("Seam edge index is outside the mesh vertex range.")

    seam_summary = _seam_graph_summary(seam_edges)
    panel_summary = _panel_topology_summary(
        vertex_count=int(vertices.shape[0]),
        faces=faces,
        seam_edges=seam_edges,
    )
    correction_counts = _selected_correction_family_counts(corrections_path)
    typed_dart_count = max(0, int(typed_dart_count)) + correction_counts["dart"]
    typed_gusset_count = max(0, int(typed_gusset_count)) + correction_counts["gusset"]
    typed_relief_cut_count = max(0, int(typed_relief_cut_count)) + correction_counts["relief_cut"]
    typed_ease_count = max(0, int(typed_ease_count)) + correction_counts["ease"]
    typed_stretch_zone_count = (
        max(0, int(typed_stretch_zone_count)) + correction_counts["stretch_zone"]
    )
    classification = _seam_graph_classification(
        seam_edges=seam_edges,
        faces=faces,
        authorized_operator_count=(
            typed_dart_count
            + typed_gusset_count
            + typed_relief_cut_count
            + typed_ease_count
            + typed_stretch_zone_count
        ),
    )
    blockers = _topology_blockers(
        seam_summary,
        panel_summary,
        seam_edges=seam_edges,
        faces=faces,
        classification=classification,
    )
    promotion = 1 if not blockers else 0
    receipt = CutTopologyReceipt(
        solver_receipt_hash=_sha256_file(solver_receipt_path),
        mesh_hash=_sha256_file(mesh_path),
        seam_edges_hash=seam_edges_hash,
        seam_edge_segment_count=seam_summary.edge_segment_count,
        seam_vertex_count=seam_summary.vertex_count,
        seam_connected_component_count=seam_summary.connected_component_count,
        seam_endpoint_count=seam_summary.endpoint_count,
        seam_branch_vertex_count=seam_summary.branch_vertex_count,
        panel_count=panel_summary.panel_count,
        panel_face_counts=panel_summary.face_counts,
        panel_boundary_edge_counts=panel_summary.boundary_edge_counts,
        panels_are_disks=panel_summary.panels_are_disks,
        typed_dart_count=typed_dart_count,
        typed_gusset_count=typed_gusset_count,
        ordinary_boundary_component_count=classification.ordinary_boundary_component_count,
        typed_operator_count=classification.typed_operator_count,
        typed_relief_cut_count=typed_relief_cut_count,
        typed_ease_count=typed_ease_count,
        typed_stretch_zone_count=typed_stretch_zone_count,
        invalid_fragmentation_count=classification.invalid_fragmentation_count,
        seam_graph_classifications=classification.classifications,
        promotion=promotion,
        blocked_consumers=[],
        cut_topology_blockers=blockers,
    )
    receipt.to_json(receipt_path)
    target_repair_receipt_path = (
        repair_receipt_path or receipt_path.with_name("cut_topology_repair_receipt.json")
    )
    repair_receipt = _cut_topology_repair_receipt(
        cut_topology_receipt_path=receipt_path,
        mesh_path=mesh_path,
        seam_edges_path=seam_edges_path,
        vertices=vertices,
        faces=faces,
        seam_edges=seam_edges,
        topology_blockers=blockers,
    )
    repair_receipt.to_json(target_repair_receipt_path)
    print(f"Wrote cut topology receipt to {receipt_path}")
    print(f"Wrote cut topology repair receipt to {target_repair_receipt_path}")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-receipt", type=Path, required=True)
    parser.add_argument("--seam-edges", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out-cut-topology-receipt", type=Path, required=True)
    parser.add_argument("--out-cut-topology-repair-receipt", type=Path, default=None)
    parser.add_argument("--typed-dart-count", type=int, default=0)
    parser.add_argument("--typed-gusset-count", type=int, default=0)
    parser.add_argument("--typed-relief-cut-count", type=int, default=0)
    parser.add_argument("--typed-ease-count", type=int, default=0)
    parser.add_argument("--typed-stretch-zone-count", type=int, default=0)
    parser.add_argument(
        "--corrections",
        type=Path,
        default=None,
        help="Optional metric-panelization correction payload used to infer typed operators.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_cut_topology(
        solver_receipt_path=args.solver_receipt,
        seam_edges_path=args.seam_edges,
        mesh_path=args.mesh,
        receipt_path=args.out_cut_topology_receipt,
        repair_receipt_path=args.out_cut_topology_repair_receipt,
        typed_dart_count=args.typed_dart_count,
        typed_gusset_count=args.typed_gusset_count,
        typed_relief_cut_count=args.typed_relief_cut_count,
        typed_ease_count=args.typed_ease_count,
        typed_stretch_zone_count=args.typed_stretch_zone_count,
        corrections_path=args.corrections,
    )


if __name__ == "__main__":
    main()
