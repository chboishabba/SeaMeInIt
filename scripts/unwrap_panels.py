#!/usr/bin/env python3
"""Unwrap promoted seam topology into receipted panel UV artifacts."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

from smii.seams import (
    PANEL_SERIALIZATION_BACKENDS,
    CorrectionTreeMaterializationEntry,
    CorrectionTreeMaterializationReceipt,
    PanelUnwrapReceipt,
    build_panel_serialization_competition_receipt,
    can_consume_cut_topology_receipt,
    can_consume_metric_correction_receipt,
    can_consume_solver_promotion_receipt,
    load_fabric_profile,
    load_cut_topology_receipt,
    load_metric_correction_receipt,
    load_solver_promotion_receipt,
    price_correction_operator_tree,
    select_serialization_candidate,
    serialize_panel,
)
from smii.seams.unwrap_backends import (
    BOOTSTRAP_BACKEND,
    UNWRAP_BACKENDS,
    unwrap_panel_vertices,
)
from smii.seams.panel_serialization_competition import panel_chart_diagnostics

Edge = tuple[int, int]
UNWRAP_COMPATIBLE_CORRECTION_TYPES = {
    "dart",
    "relief_cut",
    "gusset",
    "ease",
    "stretch_zone",
    "abstain",
}
UNWRAP_COMPATIBLE_CORRECTION_STATES = {
    "correctionOk",
    "correctionDegraded",
    "correctionAbstained",
}

VARIANT_SELECTION_PROFILES: dict[str, dict[str, float]] = {
    "default": {
        "worst_distortion": 18.0,
        "foldovers": 2.0,
        "boundary_deviation": 0.5,
        "chart_count": 0.25,
        "exportable_penalty": 250.0,
    },
    "debug_geometry": {
        "worst_distortion": 42.0,
        "foldovers": 8.0,
        "boundary_deviation": 1.0,
        "chart_count": 0.5,
        "exportable_penalty": 400.0,
    },
    "knit_4way_light": {
        "worst_distortion": 30.0,
        "foldovers": 5.0,
        "boundary_deviation": 1.0,
        "chart_count": 0.25,
        "exportable_penalty": 350.0,
    },
    "home_sewing": {
        "worst_distortion": 16.0,
        "foldovers": 2.0,
        "boundary_deviation": 1.5,
        "chart_count": 4.0,
        "exportable_penalty": 500.0,
    },
    "performance_squat": {
        "worst_distortion": 36.0,
        "foldovers": 4.0,
        "boundary_deviation": 0.75,
        "chart_count": 0.5,
        "exportable_penalty": 300.0,
    },
}
PARETO_METRIC_KEYS = (
    "score",
    "worst_distortion",
    "foldovers",
    "boundary_deviation",
    "chart_count",
)
OPERATOR_BASIS_SEARCH_DEPTH = 2
OPERATOR_BASIS_SEARCH_BEAM_WIDTH = 8
OPERATOR_BASIS_FAMILIES = (
    "stretch_zone",
    "gusset_patch",
    "gusset_parent_replacement",
    "dart_wedge",
    "lens_patch",
    "relief_path",
    "relief_tree",
    "drain_path",
    "grain_rotation",
    "ease_band",
    "seam_relocation",
    "boundary_split",
    "variable_knit_proxy",
)


def _json_float(value: object) -> float:
    return float(cast(Any, value))


def _json_int(value: object) -> int:
    return int(cast(Any, value))


def _json_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


@dataclass(frozen=True, slots=True)
class PanelPatch:
    """Post-cut mesh component used by the bootstrap panel unwrapper."""

    vertices: tuple[int, ...]
    edges: tuple[Edge, ...]
    faces: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class SeamPayload:
    """Seam edge payload with optional face-region labels."""

    edges: tuple[Edge, ...]
    face_labels: np.ndarray | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, object] | None) -> str | None:
    if payload is None:
        return None
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    return f"{minutes}m {remainder:02d}s"


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


def _load_seam_payload(path: Path) -> SeamPayload:
    payload = np.load(path, allow_pickle=True)
    if "seam_edges" not in payload:
        raise KeyError("Seam edges NPZ must contain a 'seam_edges' array.")
    raw_edges = np.asarray(payload["seam_edges"], dtype=int)
    if raw_edges.ndim != 2 or raw_edges.shape[1] != 2:
        raise ValueError("seam_edges must be shaped (N, 2).")
    face_labels = None
    if "face_labels" in payload:
        face_labels = np.asarray(payload["face_labels"], dtype=int)
        if face_labels.ndim != 1:
            raise ValueError("face_labels must be a one-dimensional array.")
    return SeamPayload(
        edges=tuple(sorted({_normalize_edge(int(a), int(b)) for a, b in raw_edges})),
        face_labels=face_labels,
    )


def _load_corrected_residuals(
    path: Path | None,
    *,
    expected_hash: str | None,
    panel_count: int,
) -> tuple[list[float] | None, str | None, list[str]]:
    if path is None:
        return None, None, []
    if not path.exists():
        return None, None, ["missing_correction_payload"]
    payload_hash = _sha256_file(path)
    if expected_hash is not None and payload_hash != expected_hash:
        return None, payload_hash, ["correction_payload_hash_mismatch"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, payload_hash, ["invalid_correction_payload"]
    if not isinstance(payload, Mapping):
        return None, payload_hash, ["invalid_correction_payload"]
    raw_panels = payload.get("panels", [])
    if not isinstance(raw_panels, list):
        return None, payload_hash, ["invalid_correction_payload"]
    by_label: dict[int, float] = {}
    for entry in raw_panels:
        if not isinstance(entry, Mapping):
            continue
        try:
            label = int(entry["panel_label"])
            residual = float(entry["corrected_metric_residual"])
        except (KeyError, TypeError, ValueError):
            continue
        if residual >= 0.0 and np.isfinite(residual):
            by_label[label] = residual
    if len(by_label) != panel_count:
        return None, payload_hash, ["correction_panel_count_mismatch"]
    return [float(by_label[idx]) for idx in range(panel_count)], payload_hash, []


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
    face_tuples = [
        (int(face[0]), int(face[1]), int(face[2])) for face in np.asarray(faces, dtype=int)
    ]
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


def _extract_panels_from_face_labels(
    *,
    faces: np.ndarray,
    face_labels: np.ndarray,
) -> list[PanelPatch]:
    if int(face_labels.shape[0]) != int(faces.shape[0]):
        raise ValueError(
            "face_labels length must match mesh face count: "
            f"labels={face_labels.shape[0]}, faces={faces.shape[0]}."
        )
    panels: list[PanelPatch] = []
    for label in sorted({int(value) for value in face_labels}):
        local_faces = tuple(
            (int(faces[idx][0]), int(faces[idx][1]), int(faces[idx][2]))
            for idx in np.where(face_labels == int(label))[0]
        )
        local_vertices = tuple(sorted({vertex for face in local_faces for vertex in face}))
        local_edges = _face_edges(local_faces)
        if local_faces:
            panels.append(
                PanelPatch(
                    vertices=local_vertices,
                    edges=local_edges,
                    faces=local_faces,
                )
            )
    return panels


def _seam_graph_blockers(seam_edges: Sequence[Edge]) -> list[str]:
    if not seam_edges:
        return []
    graph = _adjacency(seam_edges)
    endpoint_count = sum(1 for vertex in graph if len(graph[vertex]) == 1)
    branch_vertex_count = sum(1 for vertex in graph if len(graph[vertex]) > 2)
    if endpoint_count != 0 or branch_vertex_count != 0:
        return ["unpriced_correction_tree_node"]
    return []


def _correction_tree_receipt(
    seam_edges: Sequence[Edge],
    *,
    typed_operator_count: int = 0,
    operator_scoring_receipt: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Classify seam graph branches as correction-tree nodes rather than topology failures."""

    graph = _adjacency(seam_edges)
    raw_scored_nodes_value = (
        operator_scoring_receipt.get("nodes", []) if operator_scoring_receipt is not None else []
    )
    raw_scored_nodes = raw_scored_nodes_value if isinstance(raw_scored_nodes_value, list) else []
    scored_nodes = {
        int(node["branch_vertex"]): node
        for node in raw_scored_nodes
        if isinstance(node, Mapping) and "branch_vertex" in node
    }
    branch_nodes: list[dict[str, object]] = []
    endpoint_vertices = [
        int(vertex) for vertex, neighbors in sorted(graph.items()) if len(neighbors) == 1
    ]
    for branch_idx, (vertex, neighbors) in enumerate(
        (item for item in sorted(graph.items()) if len(item[1]) > 2)
    ):
        typed = branch_idx < typed_operator_count
        scored_node = scored_nodes.get(int(vertex))
        scored_promoted = bool(scored_node.get("promoted")) if scored_node is not None else False
        selected_operator = (
            str(scored_node.get("selected_operator"))
            if scored_node is not None
            else ("typed_correction_operator" if typed else "diagnostic_unresolved")
        )
        branch_nodes.append(
            {
                "branch_vertex": int(vertex),
                "incident_degree": int(len(neighbors)),
                "operator_family_candidates": [
                    "dart_apex",
                    "gusset_corner",
                    "ease_convergence",
                    "stretch_zone",
                    "seam_junction",
                    "diagnostic_unresolved",
                ],
                "selected_operator_family": selected_operator,
                "promotion": "typed_priced_operator"
                if scored_promoted
                else ("typed_pending_metric_price" if typed else "diagnostic_only"),
                "blockers": [] if typed or scored_promoted else ["unpriced_correction_tree_node"],
            }
        )
    scored_typed_count = 0
    if operator_scoring_receipt is not None:
        scored_typed_count = _json_int(operator_scoring_receipt.get("typed_branch_count", 0))
    typed_branch_count = min(
        max(0, int(typed_operator_count), scored_typed_count), len(branch_nodes)
    )
    diagnostic_branch_count = len(branch_nodes) - typed_branch_count
    diagnostic_endpoint_count = len(endpoint_vertices) if typed_branch_count <= 0 else 0
    blocker = bool(diagnostic_branch_count or diagnostic_endpoint_count)
    return {
        "schema_version": "smii.correction_tree_receipt.v1",
        "morphology_boundary": "branches_are_operator_nodes_not_intrinsic_blockers",
        "branch_count": len(branch_nodes),
        "typed_branch_count": typed_branch_count,
        "diagnostic_branch_count": diagnostic_branch_count,
        "endpoint_count": len(endpoint_vertices),
        "diagnostic_endpoint_count": diagnostic_endpoint_count,
        "endpoint_vertices": endpoint_vertices,
        "branch_nodes": branch_nodes,
        "promotion": 0 if blocker else 1,
        "blockers": ["unpriced_correction_tree_node"] if blocker else [],
    }


def _cut_topology_blockers(
    *,
    vertices: np.ndarray,
    panels: Sequence[PanelPatch],
) -> list[str]:
    blockers: list[str] = []
    if not panels:
        return ["panel_fragmentation_invalid"]
    for panel in panels:
        diagnostics = panel_chart_diagnostics(vertices, panel)
        blockers.extend(str(blocker) for blocker in _json_list(diagnostics["blockers"]))
    if blockers:
        blockers.append("chart_domain_not_backend_serializable")
    return blockers


def _fabric_allowance_for_grain(profile: object, grain: str) -> float:
    compliance = getattr(profile, "compliance")
    values = [
        float(getattr(compliance, "s_parallel")),
        float(getattr(compliance, "s_perp")),
        float(getattr(compliance, "s_shear")),
    ]
    max_compliance = max(values + [1e-6])
    if grain == "warp":
        directional = float(getattr(compliance, "s_parallel"))
    elif grain == "weft":
        directional = float(getattr(compliance, "s_perp"))
    else:
        directional = 0.5 * (
            float(getattr(compliance, "s_parallel")) + float(getattr(compliance, "s_perp"))
        )
    shear = float(getattr(compliance, "s_shear"))
    normalized = max(
        0.0, min(1.0, 0.7 * directional / max_compliance + 0.3 * shear / max_compliance)
    )
    absolute_compliance = max(0.0, min(1.0, max_compliance))
    return 0.02 + 0.18 * normalized * absolute_compliance


def _fabric_metric_receipt(
    *,
    fabric_profile_path: Path,
    distortions: Sequence[float],
    corrected_residuals: Sequence[float] | None,
    grain_directions: Sequence[str],
) -> dict[str, object]:
    """Build a fabric-relative metric gate from panel distortion/residual proxies."""

    profile = load_fabric_profile(fabric_profile_path)
    panels: list[dict[str, object]] = []
    violation_scores: list[float] = []
    for idx, distortion in enumerate(distortions):
        grain = grain_directions[idx] if idx < len(grain_directions) else "warp"
        allowable = _fabric_allowance_for_grain(profile, grain)
        residual = (
            float(corrected_residuals[idx])
            if corrected_residuals is not None and idx < len(corrected_residuals)
            else float(distortion)
        )
        strain_violation = max(0.0, float(distortion) - allowable)
        residual_violation = max(0.0, residual - allowable)
        score = max(strain_violation, residual_violation)
        violation_scores.append(score)
        panels.append(
            {
                "panel_id": f"P{idx}",
                "grain_direction": grain,
                "generic_distortion": float(distortion),
                "corrected_metric_residual": residual,
                "allowable_metric_mismatch": allowable,
                "warp_strain_violation": strain_violation if grain == "warp" else 0.0,
                "weft_strain_violation": strain_violation if grain == "weft" else 0.0,
                "shear_violation_proxy": residual_violation,
                "violation_score": score,
            }
        )
    worst_violation = max(violation_scores) if violation_scores else 0.0
    return {
        "schema_version": "smii.fabric_aware_panel_metric.v1",
        "fabric_profile": profile.fabric_id,
        "fabric_profile_hash": _sha256_file(fabric_profile_path),
        "fabric_profile_path": str(fabric_profile_path),
        "metric_boundary": "generic_uv_distortion_is_only_a_proxy",
        "panel_count": len(panels),
        "panels": panels,
        "worst_fabric_violation": worst_violation,
        "mean_fabric_violation": float(sum(violation_scores) / len(violation_scores))
        if violation_scores
        else 0.0,
        "promotion": 1 if worst_violation <= 1e-12 else 0,
        "blockers": ["fabric_metric_violation_exceeds_profile"] if worst_violation > 1e-12 else [],
    }


def _realized_correction_operator_receipt(
    *,
    operator_scoring_receipt: Mapping[str, object],
    fabric_metric_receipt: Mapping[str, object],
    residual_gate: float,
) -> dict[str, object]:
    """Realize priced correction operators as local metric/fabric overrides."""

    nodes = _json_list(operator_scoring_receipt.get("nodes", []))
    panels = _json_list(fabric_metric_receipt.get("panels", []))
    fabric_before = _json_float(operator_scoring_receipt.get("fabric_violation_before", 0.0))
    residual_before = _json_float(operator_scoring_receipt.get("residual_before", 0.0))
    estimated_fabric_after = _json_float(
        operator_scoring_receipt.get("estimated_worst_fabric_violation_after", fabric_before)
    )
    estimated_residual_after = _json_float(
        operator_scoring_receipt.get("estimated_worst_residual_after", residual_before)
    )
    fabric_ratio = estimated_fabric_after / fabric_before if fabric_before > 1e-12 else 1.0
    residual_ratio = estimated_residual_after / residual_before if residual_before > 1e-12 else 1.0

    realized_nodes: list[dict[str, object]] = []
    realized_count = 0
    companion_count = 0
    blockers: list[str] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        selected_operator = str(node.get("selected_operator", "diagnostic_carry"))
        promoted = bool(node.get("promoted", False))
        can_realize = promoted and selected_operator == "stretch_zone"
        node_blockers: list[str] = []
        if promoted and selected_operator != "stretch_zone":
            node_blockers.append("operator_realization_not_implemented")
        if not promoted:
            node_blockers.append("operator_not_promoted_for_realization")
        if can_realize:
            realized_count += 1
        blockers.extend(node_blockers)
        realized_nodes.append(
            {
                "branch_id": str(node.get("branch_id", "")),
                "branch_vertex": int(node.get("branch_vertex", -1)),
                "operator": selected_operator,
                "realization": "local_fabric_strain_cone_override" if can_realize else "none",
                "support_region": "panel_local_branch_neighbourhood",
                "pattern_annotation": "stretch_zone" if can_realize else "diagnostic_unrealized",
                "fabric_metric_before": fabric_before,
                "fabric_metric_after_realized": estimated_fabric_after
                if can_realize
                else fabric_before,
                "residual_before": residual_before,
                "residual_after_realized": estimated_residual_after
                if can_realize
                else residual_before,
                "realized": can_realize,
                "blockers": node_blockers,
            }
        )

    realized_fabric_after = estimated_fabric_after if realized_count else fabric_before
    realized_residual_after = estimated_residual_after if realized_count else residual_before
    needs_gusset_companion = bool(realized_count) and realized_residual_after > float(residual_gate)
    if needs_gusset_companion:
        gusset_fabric_ratio = 0.85
        gusset_residual_ratio = 0.62
        for node in nodes:
            if not isinstance(node, Mapping) or not bool(node.get("promoted", False)):
                continue
            companion_count += 1
            realized_count += 1
            realized_nodes.append(
                {
                    "branch_id": f"{str(node.get('branch_id', ''))}:gusset",
                    "branch_vertex": int(node.get("branch_vertex", -1)),
                    "operator": "gusset_corner",
                    "realization": "inserted_metric_patch_residual_relief",
                    "support_region": "panel_local_branch_neighbourhood",
                    "pattern_annotation": "gusset_corner",
                    "fabric_metric_before": realized_fabric_after,
                    "fabric_metric_after_realized": realized_fabric_after * gusset_fabric_ratio,
                    "residual_before": realized_residual_after,
                    "residual_after_realized": realized_residual_after * gusset_residual_ratio,
                    "realized": True,
                    "blockers": [],
                }
            )
        if companion_count:
            realized_fabric_after *= gusset_fabric_ratio
            realized_residual_after *= gusset_residual_ratio

    fabric_ratio = realized_fabric_after / fabric_before if fabric_before > 1e-12 else 1.0
    residual_ratio = realized_residual_after / residual_before if residual_before > 1e-12 else 1.0
    realized_panels: list[dict[str, object]] = []
    realized_panel_violations: list[float] = []
    realized_panel_residuals: list[float] = []
    for panel in panels:
        if not isinstance(panel, Mapping):
            continue
        violation_before = _json_float(panel.get("violation_score", 0.0))
        residual_panel_before = _json_float(panel.get("corrected_metric_residual", 0.0))
        violation_after = violation_before * fabric_ratio if realized_count else violation_before
        residual_after = (
            residual_panel_before * residual_ratio if realized_count else residual_panel_before
        )
        realized_panel_violations.append(violation_after)
        realized_panel_residuals.append(residual_after)
        realized_panels.append(
            {
                "panel_id": str(panel.get("panel_id", "")),
                "fabric_violation_before": violation_before,
                "fabric_violation_after_realized": violation_after,
                "corrected_metric_residual_before": residual_panel_before,
                "corrected_metric_residual_after_realized": residual_after,
                "local_allowable_metric_override": "stretch_zone+gusset_corner"
                if companion_count
                else ("stretch_zone" if realized_count else "no_realized_operator"),
            }
        )

    worst_fabric_after = (
        max(realized_panel_violations) if realized_panel_violations else fabric_before
    )
    worst_residual_after = (
        max(realized_panel_residuals) if realized_panel_residuals else residual_before
    )
    blocker = bool(blockers)
    return {
        "schema_version": "smii.realized_correction_operator.v1",
        "claim_boundary": "stretch_zones_and_gussets_are_local_metric_overrides_not_cloth_simulation",
        "branch_count": int(operator_scoring_receipt.get("branch_count", 0)),
        "realized_operator_count": realized_count,
        "companion_operator_count": companion_count,
        "unrealized_operator_count": max(0, len(realized_nodes) - realized_count),
        "operator_families": sorted(
            {str(node["operator"]) for node in realized_nodes if node.get("realized")}
        ),
        "nodes": realized_nodes,
        "panels": realized_panels,
        "estimated_worst_fabric_violation_after": estimated_fabric_after,
        "realized_worst_fabric_violation_after": worst_fabric_after,
        "estimated_worst_residual_after": estimated_residual_after,
        "realized_worst_residual_after": worst_residual_after,
        "residual_gate": float(residual_gate),
        "estimate_realization_delta": abs(worst_fabric_after - estimated_fabric_after),
        "promotion": 0 if blocker else 1,
        "blockers": sorted(set(blockers)),
    }


def _affected_panels_for_branch(
    panels: Sequence[PanelPatch],
    branch_vertex: int,
) -> list[int]:
    return [
        idx
        for idx, panel in enumerate(panels)
        if int(branch_vertex) in {int(vertex) for vertex in panel.vertices}
    ]


def _materialization_kind(operator: str, realized: bool) -> tuple[str, bool, bool]:
    if not realized:
        return "annotation_only", False, False
    if operator == "stretch_zone":
        return "backend_hint", True, True
    if operator == "gusset_corner":
        return "inserted_patch", True, True
    if operator == "dart_apex":
        return "relief_cut_hint", True, True
    if operator == "ease_convergence":
        return "ease_interval_hint", True, True
    if operator == "grain_rotation":
        return "grainline_override", True, True
    return "annotation_only", False, False


def _attachment_edges_for_branch(
    panels: Sequence[PanelPatch],
    affected_panels: Sequence[int],
    branch_vertex: int,
) -> list[dict[str, object]]:
    attachments: list[dict[str, object]] = []
    for panel_idx in affected_panels:
        if panel_idx < 0 or panel_idx >= len(panels):
            continue
        panel = panels[panel_idx]
        incident_edges = [
            [int(a), int(b)]
            for a, b in panel.edges
            if int(a) == int(branch_vertex) or int(b) == int(branch_vertex)
        ]
        if incident_edges:
            attachments.append(
                {
                    "panel_id": f"P{panel_idx}",
                    "branch_vertex": int(branch_vertex),
                    "incident_edges": incident_edges,
                }
            )
    return attachments


def _operator_geometry_payload(
    *,
    operator: str,
    node: Mapping[str, object],
    panels: Sequence[PanelPatch],
    affected_panels: Sequence[int],
) -> dict[str, object] | None:
    branch_vertex = int(node.get("branch_vertex", -1))
    if operator == "gusset_corner":
        attachments = _attachment_edges_for_branch(panels, affected_panels, branch_vertex)
        return {
            "creates_new_chart": True,
            "patch_shape": "diamond",
            "branch_vertex": branch_vertex,
            "support_region": "panel_local_branch_neighbourhood",
            "attachment_edges": attachments,
            "parent_panels_modified": [f"P{idx}" for idx in affected_panels],
        }
    if operator == "stretch_zone":
        return {
            "creates_new_chart": False,
            "support_region": "panel_local_branch_neighbourhood",
            "branch_vertex": branch_vertex,
            "grain_axis_hint": "panel_primary_axis",
        }
    return None


def _correction_tree_materialization_receipt(
    *,
    correction_tree_hash: str,
    correction_tree_receipt_hash: str,
    correction_operator_scoring_receipt_hash: str | None,
    realized_correction_operator_receipt: Mapping[str, object] | None,
    panels: Sequence[PanelPatch],
) -> dict[str, object] | None:
    """Materialize metric-realized correction operators into chart/backend constraints."""

    if realized_correction_operator_receipt is None:
        return None
    materializations: list[CorrectionTreeMaterializationEntry] = []
    top_blockers: list[str] = []
    for node in _json_list(realized_correction_operator_receipt.get("nodes", [])):
        if not isinstance(node, Mapping):
            continue
        operator = str(node.get("operator", "diagnostic_carry"))
        realized = bool(node.get("realized", False))
        kind, chart_materialized, backend_constraints_emitted = _materialization_kind(
            operator,
            realized,
        )
        node_blockers = [str(blocker) for blocker in _json_list(node.get("blockers", []))]
        if realized and not chart_materialized:
            node_blockers.append("operator_not_chart_materialized")
        if not realized:
            node_blockers.append("operator_not_metric_realized")
        branch_vertex = int(node.get("branch_vertex", -1))
        affected_panels = _affected_panels_for_branch(panels, branch_vertex)
        if realized and not affected_panels:
            node_blockers.append("operator_support_panel_not_found")
        geometry = _operator_geometry_payload(
            operator=operator,
            node=node,
            panels=panels,
            affected_panels=affected_panels,
        )
        node_blockers = list(dict.fromkeys(node_blockers))
        top_blockers.extend(node_blockers)
        materializations.append(
            CorrectionTreeMaterializationEntry(
                node_id=str(node.get("branch_id", "")),
                operator_family=operator,
                metric_realized=realized,
                chart_materialized=chart_materialized,
                materialization_kind=kind,
                affected_panels=affected_panels,
                backend_constraints_emitted=backend_constraints_emitted,
                promotion=1 if not node_blockers else 0,
                blockers=node_blockers,
                geometry=geometry,
            )
        )
    top_blockers = list(dict.fromkeys(top_blockers))
    receipt = CorrectionTreeMaterializationReceipt(
        correction_tree_hash=correction_tree_hash,
        correction_tree_receipt_hash=correction_tree_receipt_hash,
        correction_operator_scoring_receipt_hash=correction_operator_scoring_receipt_hash,
        materializations=materializations,
        materialized_operator_count=len(materializations),
        promotion=0 if top_blockers else 1,
        blocked_consumers=[],
        blockers=top_blockers,
    )
    return receipt.to_dict()


def _panel_materialization_constraints(
    materialization_receipt: Mapping[str, object] | None,
    panel_idx: int,
    *,
    chart_panel_id: str | None = None,
) -> dict[str, object] | None:
    if materialization_receipt is None:
        return None
    entries: list[dict[str, object]] = []
    for entry in _json_list(materialization_receipt.get("materializations", [])):
        if not isinstance(entry, Mapping):
            continue
        affected = [int(value) for value in _json_list(entry.get("affected_panels", []))]
        if int(panel_idx) not in affected:
            continue
        entries.append(
            {
                "node_id": str(entry.get("node_id", "")),
                "operator_family": str(entry.get("operator_family", "")),
                "materialization_kind": str(entry.get("materialization_kind", "")),
                "chart_materialized": bool(entry.get("chart_materialized", False)),
                "backend_constraints_emitted": bool(
                    entry.get("backend_constraints_emitted", False)
                ),
                "geometry": dict(entry.get("geometry", {}))
                if isinstance(entry.get("geometry"), Mapping)
                else {},
            }
        )
    return {
        "schema_version": str(materialization_receipt.get("schema_version", "")),
        "promotion": int(materialization_receipt.get("promotion", 0)),
        "panel_id": f"P{panel_idx}",
        "chart_panel_id": chart_panel_id or f"P{panel_idx}",
        "constraint_count": len(entries),
        "constraints": entries,
    }


def _materialization_requires_chart_split(
    materialization_receipt: Mapping[str, object] | None,
    panel_idx: int,
) -> bool:
    if materialization_receipt is None:
        return False
    for entry in _json_list(materialization_receipt.get("materializations", [])):
        if not isinstance(entry, Mapping):
            continue
        affected = [int(value) for value in _json_list(entry.get("affected_panels", []))]
        if int(panel_idx) not in affected:
            continue
        kind = str(entry.get("materialization_kind", ""))
        if kind in {"inserted_patch", "patch_placeholder", "relief_cut_hint"}:
            return True
    return False


def _gusset_branch_vertices_for_panel(
    materialization_receipt: Mapping[str, object] | None,
    panel_idx: int,
) -> list[int]:
    if materialization_receipt is None:
        return []
    branch_vertices: list[int] = []
    for entry in _json_list(materialization_receipt.get("materializations", [])):
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("operator_family", "")) != "gusset_corner":
            continue
        if str(entry.get("materialization_kind", "")) != "inserted_patch":
            continue
        affected = [int(value) for value in _json_list(entry.get("affected_panels", []))]
        if int(panel_idx) not in affected:
            continue
        geometry = entry.get("geometry", {})
        if not isinstance(geometry, Mapping):
            continue
        branch_vertex = int(geometry.get("branch_vertex", -1))
        if branch_vertex >= 0:
            branch_vertices.append(branch_vertex)
    return list(dict.fromkeys(branch_vertices))


def _carved_gusset_parent_panels(
    panel: PanelPatch,
    branch_vertices: Sequence[int],
    *,
    ring_depth: int = 1,
) -> list[PanelPatch]:
    branch_set = {int(vertex) for vertex in branch_vertices}
    if not branch_set or not panel.faces:
        return [panel]
    removal_vertices = set(branch_set)
    for _ in range(max(1, int(ring_depth)) - 1):
        next_vertices = set(removal_vertices)
        for face in panel.faces:
            face_vertices = {int(vertex) for vertex in face}
            if face_vertices & removal_vertices:
                next_vertices.update(face_vertices)
        removal_vertices = next_vertices
    remaining_faces = tuple(
        face for face in panel.faces if not any(int(vertex) in removal_vertices for vertex in face)
    )
    if not remaining_faces or len(remaining_faces) == len(panel.faces):
        return [panel]
    carved = _panel_from_faces(panel, remaining_faces)
    if not carved:
        return [panel]
    return carved


def _branch_ring_faces(
    panel: PanelPatch,
    branch_vertices: Sequence[int],
    *,
    ring_depth: int,
) -> tuple[tuple[int, int, int], ...]:
    branch_set = {int(vertex) for vertex in branch_vertices}
    if not branch_set or not panel.faces:
        return ()
    active_vertices = set(branch_set)
    ring_faces: set[tuple[int, int, int]] = set()
    for _ in range(max(1, int(ring_depth))):
        next_vertices = set(active_vertices)
        for face in panel.faces:
            face_vertices = {int(vertex) for vertex in face}
            if face_vertices & active_vertices:
                ring_faces.add(face)
                next_vertices.update(face_vertices)
        active_vertices = next_vertices
    return tuple(face for face in panel.faces if face in ring_faces)


def _branch_spoke_split_parent_panels(
    panel: PanelPatch,
    branch_vertices: Sequence[int],
    *,
    ring_depth: int = 1,
) -> list[PanelPatch]:
    """Split branch-local faces into charts without deleting parent fabric."""

    branch_faces = set(_branch_ring_faces(panel, branch_vertices, ring_depth=ring_depth))
    if not branch_faces or len(branch_faces) == len(panel.faces):
        return [panel]
    remainder_faces = tuple(face for face in panel.faces if face not in branch_faces)
    split_panels = _panel_from_faces(panel, tuple(branch_faces)) + _panel_from_faces(
        panel,
        remainder_faces,
    )
    if len(split_panels) < 2:
        return [panel]
    return split_panels


def _face_distortion(
    vertices: np.ndarray,
    face: tuple[int, int, int],
    uv_by_vertex: Mapping[int, np.ndarray],
) -> float:
    distortions: list[float] = []
    for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
        if a not in uv_by_vertex or b not in uv_by_vertex:
            continue
        length_3d = float(np.linalg.norm(vertices[a] - vertices[b]))
        length_2d = float(np.linalg.norm(uv_by_vertex[a] - uv_by_vertex[b]))
        if length_3d > 1e-12:
            distortions.append(abs(length_2d - length_3d) / length_3d)
    return float(sum(distortions) / len(distortions)) if distortions else 0.0


def _face_signed_area(face: tuple[int, int, int], uv_by_vertex: Mapping[int, np.ndarray]) -> float:
    try:
        a, b, c = (uv_by_vertex[int(vertex)] for vertex in face)
    except KeyError:
        return 0.0
    ab = b - a
    ac = c - a
    return 0.5 * float(ab[0] * ac[1] - ab[1] * ac[0])


def _relief_boundary_edges(
    selected_faces: Sequence[tuple[int, int, int]],
    remainder_faces: Sequence[tuple[int, int, int]],
) -> list[list[int]]:
    selected_edges = set(_face_edges(selected_faces))
    remainder_edges = set(_face_edges(remainder_faces))
    return [[int(a), int(b)] for a, b in sorted(selected_edges & remainder_edges)]


def _drain_boundary_edges(
    selected_faces: Sequence[tuple[int, int, int]],
    remainder_faces: Sequence[tuple[int, int, int]],
) -> list[list[int]]:
    selected_edges = set(_face_edges(selected_faces))
    remainder_edges = set(_face_edges(remainder_faces))
    return [[int(a), int(b)] for a, b in sorted(selected_edges & remainder_edges)]


def _serialization_failure_field_receipt(
    *,
    panel_id: str,
    backend: str,
    vertices: np.ndarray,
    panel: PanelPatch,
    uv: np.ndarray,
    seam_edges: Sequence[Edge] | None = None,
    distortion_threshold: float,
) -> dict[str, object]:
    """Measure face-local serialization failure evidence for downstream relief paths."""

    uv_by_vertex = {int(vertex): uv[idx] for idx, vertex in enumerate(panel.vertices)}
    face_distortions = [_face_distortion(vertices, face, uv_by_vertex) for face in panel.faces]
    signed_areas = [_face_signed_area(face, uv_by_vertex) for face in panel.faces]
    nonzero_signs = [1 if area > 0.0 else -1 for area in signed_areas if abs(area) > 1e-12]
    majority_sign = (
        1
        if sum(1 for sign in nonzero_signs if sign > 0)
        >= sum(1 for sign in nonzero_signs if sign < 0)
        else -1
    )
    foldover_faces = [
        idx
        for idx, area in enumerate(signed_areas)
        if abs(area) > 1e-12 and (1 if area > 0.0 else -1) != majority_sign
    ]
    high_distortion_faces = [
        idx for idx, distortion in enumerate(face_distortions) if distortion > distortion_threshold
    ]
    failure_face_indices = sorted(set(foldover_faces) | set(high_distortion_faces))
    source = "foldover_cluster_path" if foldover_faces else "distortion_gradient_path"
    failure_face_components = _face_connected_components(panel.faces, failure_face_indices)
    candidate_relief_paths: list[dict[str, object]] = []
    candidate_drain_paths: list[dict[str, object]] = []
    candidate_wedge_reliefs: list[dict[str, object]] = []
    candidate_guided_dart_wedges: list[dict[str, object]] = []
    candidate_lens_patches: list[dict[str, object]] = []
    _, boundary_faces = _face_adjacency_and_boundary_faces(panel.faces)
    seam_boundary_faces = _faces_touching_edges(panel.faces, seam_edges or [])
    face_costs = [
        float(distortion)
        + (5.0 if idx in foldover_faces else 0.0)
        + (1.0 if idx in high_distortion_faces else 0.0)
        for idx, distortion in enumerate(face_distortions)
    ]
    for component_indices in failure_face_components:
        selected_faces = [panel.faces[idx] for idx in component_indices]
        remainder_faces = [
            face for idx, face in enumerate(panel.faces) if idx not in set(component_indices)
        ]
        edge_path = _relief_boundary_edges(selected_faces, remainder_faces)
        candidate_relief_paths.append(
            {
                "source": source,
                "edge_path": edge_path,
                "failure_face_indices": component_indices,
                "separates_bad_region": bool(
                    component_indices and len(component_indices) < len(panel.faces)
                ),
                "face_partition_preserves_faces": bool(
                    component_indices and len(component_indices) < len(panel.faces)
                ),
            }
        )
        candidate_guided_dart_wedges.extend(
            _candidate_guided_dart_wedges(
                panel,
                source=source,
                relief_path=candidate_relief_paths[-1],
                boundary_faces=sorted(boundary_faces),
                seam_boundary_faces=seam_boundary_faces,
                face_costs=face_costs,
            )
        )
        candidate_lens_patches.extend(
            _candidate_failure_lens_patches(
                panel,
                source=source,
                relief_path=candidate_relief_paths[-1],
                face_costs=face_costs,
            )
        )
        drain_face_indices = _shortest_face_path_to_targets(
            panel.faces,
            component_indices,
            sorted(boundary_faces),
            face_costs,
        )
        drain_extension = set(drain_face_indices) - set(component_indices)
        if drain_face_indices and drain_extension and len(drain_face_indices) < len(panel.faces):
            drain_selected_faces = [panel.faces[idx] for idx in drain_face_indices]
            drain_remainder_faces = [
                face for idx, face in enumerate(panel.faces) if idx not in set(drain_face_indices)
            ]
            candidate_drain_paths.append(
                {
                    "source": source,
                    "sink": "panel_boundary",
                    "edge_path": _drain_boundary_edges(
                        drain_selected_faces,
                        drain_remainder_faces,
                    ),
                    "failure_face_indices": component_indices,
                    "drain_face_indices": drain_face_indices,
                    "path_length": int(len(drain_face_indices)),
                    "drains_to_boundary": True,
                    "separates_bad_region": bool(
                        component_indices and len(drain_face_indices) < len(panel.faces)
                    ),
                    "face_partition_preserves_faces": bool(
                        component_indices and len(drain_face_indices) < len(panel.faces)
                    ),
                }
            )
        if seam_boundary_faces:
            seam_drain_face_indices = _shortest_face_path_to_targets(
                panel.faces,
                component_indices,
                seam_boundary_faces,
                face_costs,
            )
            seam_drain_extension = set(seam_drain_face_indices) - set(component_indices)
            if (
                seam_drain_face_indices
                and seam_drain_extension
                and len(seam_drain_face_indices) < len(panel.faces)
            ):
                seam_selected_faces = [panel.faces[idx] for idx in seam_drain_face_indices]
                seam_remainder_faces = [
                    face
                    for idx, face in enumerate(panel.faces)
                    if idx not in set(seam_drain_face_indices)
                ]
                candidate_drain_paths.append(
                    {
                        "source": source,
                        "sink": "seam_boundary",
                        "edge_path": _drain_boundary_edges(
                            seam_selected_faces,
                            seam_remainder_faces,
                        ),
                        "failure_face_indices": component_indices,
                        "drain_face_indices": seam_drain_face_indices,
                        "path_length": int(len(seam_drain_face_indices)),
                        "drains_to_boundary": True,
                        "separates_bad_region": bool(
                            component_indices and len(seam_drain_face_indices) < len(panel.faces)
                        ),
                        "face_partition_preserves_faces": bool(
                            component_indices and len(seam_drain_face_indices) < len(panel.faces)
                        ),
                    }
                )
        candidate_wedge_reliefs.extend(
            _candidate_failure_wedge_reliefs(
                panel,
                source=source,
                component_indices=component_indices,
                boundary_faces=sorted(boundary_faces),
                seam_boundary_faces=seam_boundary_faces,
                face_costs=face_costs,
            )
        )
    selected_drain_paths = _selected_failure_drain_paths(
        {
            "candidate_drain_paths": candidate_drain_paths,
        }
    )
    selected_wedge_reliefs = _selected_failure_wedge_reliefs(
        {
            "candidate_wedge_reliefs": candidate_wedge_reliefs,
        }
    )
    selected_guided_dart_wedges = _selected_guided_dart_wedges(
        {
            "candidate_guided_dart_wedges": candidate_guided_dart_wedges,
        }
    )
    selected_lens_patches = _selected_failure_lens_patches(
        {
            "candidate_lens_patches": candidate_lens_patches,
        }
    )
    return {
        "schema_version": "smii.serialization_failure_field.v1",
        "panel_id": panel_id,
        "backend": backend,
        "foldover_faces": foldover_faces,
        "high_distortion_faces": high_distortion_faces,
        "face_distortions": [float(value) for value in face_distortions],
        "failure_face_components": failure_face_components,
        "distortion_ridges": [
            edge
            for path in candidate_relief_paths
            for edge in _json_list(path.get("edge_path", []))
        ]
        + [edge for path in candidate_drain_paths for edge in _json_list(path.get("edge_path", []))]
        + [
            edge
            for path in candidate_wedge_reliefs
            for edge in _json_list(path.get("edge_path", []))
        ]
        + [
            edge
            for path in candidate_guided_dart_wedges
            for edge in _json_list(path.get("edge_path", []))
        ]
        + [
            edge
            for path in candidate_lens_patches
            for edge in _json_list(path.get("edge_path", []))
        ],
        "candidate_relief_paths": candidate_relief_paths,
        "candidate_drain_paths": candidate_drain_paths,
        "candidate_wedge_reliefs": candidate_wedge_reliefs,
        "candidate_guided_dart_wedges": candidate_guided_dart_wedges,
        "candidate_lens_patches": candidate_lens_patches,
        "selected_drain_paths": selected_drain_paths,
        "selected_wedge_reliefs": selected_wedge_reliefs,
        "selected_guided_dart_wedges": selected_guided_dart_wedges,
        "selected_lens_patches": selected_lens_patches,
    }


def _failure_relief_variant_id(failure_field: Mapping[str, object] | None) -> str:
    if failure_field is None:
        return "failure_relief_path"
    candidate_count = sum(
        1
        for path in _json_list(failure_field.get("candidate_relief_paths", []))
        if isinstance(path, Mapping) and bool(path.get("separates_bad_region", False))
    )
    return "failure_relief_tree" if candidate_count > 1 else "failure_relief_path"


def _failure_drain_variant_id(failure_field: Mapping[str, object] | None) -> str:
    if failure_field is None:
        return "failure_drain_path"
    candidate_count = sum(
        1
        for path in _json_list(failure_field.get("selected_drain_paths", []))
        if isinstance(path, Mapping) and bool(path.get("drains_to_boundary", False))
    )
    return "failure_drain_tree" if candidate_count > 1 else "failure_drain_path"


def _failure_wedge_variant_id(failure_field: Mapping[str, object] | None) -> str:
    if failure_field is None:
        return "failure_wedge_relief"
    candidate_count = sum(
        1
        for path in _json_list(failure_field.get("selected_wedge_reliefs", []))
        if isinstance(path, Mapping) and bool(path.get("creates_wedge_chart", False))
    )
    return "failure_wedge_tree" if candidate_count > 1 else "failure_wedge_relief"


def _guided_dart_wedge_variant_id(failure_field: Mapping[str, object] | None) -> str:
    if failure_field is None:
        return "pareto_guided_dart_wedge"
    candidate_count = sum(
        1
        for path in _json_list(failure_field.get("selected_guided_dart_wedges", []))
        if isinstance(path, Mapping) and bool(path.get("creates_dart_chart", False))
    )
    return "pareto_guided_dart_wedge_tree" if candidate_count > 1 else "pareto_guided_dart_wedge"


def _failure_lens_patch_variant_id(failure_field: Mapping[str, object] | None) -> str:
    if failure_field is None:
        return "failure_lens_patch"
    candidate_count = sum(
        1
        for patch in _json_list(failure_field.get("selected_lens_patches", []))
        if isinstance(patch, Mapping) and bool(patch.get("creates_patch_chart", False))
    )
    return "failure_lens_patch_tree" if candidate_count > 1 else "failure_lens_patch"


def _failure_relief_split_parent_panels(
    panel: PanelPatch,
    failure_field: Mapping[str, object] | None,
) -> list[PanelPatch]:
    """Split measured failure faces from the parent remainder without deleting fabric."""

    if failure_field is None or not panel.faces:
        return [panel]
    failure_groups: list[tuple[int, ...]] = []
    for path in _json_list(failure_field.get("candidate_relief_paths", [])):
        if not isinstance(path, Mapping):
            continue
        if not bool(path.get("separates_bad_region", False)):
            continue
        failure_indices = tuple(
            sorted(
                {
                    int(idx)
                    for idx in _json_list(path.get("failure_face_indices", []))
                    if 0 <= int(idx) < len(panel.faces)
                }
            )
        )
        if failure_indices and len(failure_indices) < len(panel.faces):
            failure_groups.append(failure_indices)
    if not failure_groups:
        return [panel]
    unique_groups = sorted(set(failure_groups), key=lambda group: (group[0], len(group)))
    child_panels: list[PanelPatch] = []
    covered_indices: set[int] = set()
    for group in unique_groups:
        covered_indices.update(group)
        group_faces = tuple(face for idx, face in enumerate(panel.faces) if idx in group)
        child_panels.extend(_panel_from_faces(panel, group_faces))
    remainder_faces = tuple(
        face for idx, face in enumerate(panel.faces) if idx not in covered_indices
    )
    if remainder_faces:
        child_panels.extend(_panel_from_faces(panel, remainder_faces))
    if len(child_panels) < 2:
        return [panel]
    return child_panels


def _failure_drain_split_parent_panels(
    panel: PanelPatch,
    failure_field: Mapping[str, object] | None,
) -> list[PanelPatch]:
    """Split measured failure corridors into charts that drain to a boundary sink."""

    if failure_field is None or not panel.faces:
        return [panel]
    drain_groups: list[tuple[int, ...]] = []
    selected_drain_paths = _json_list(failure_field.get("selected_drain_paths", []))
    source_paths = selected_drain_paths or _json_list(
        failure_field.get("candidate_drain_paths", [])
    )
    for path in source_paths:
        if not isinstance(path, Mapping):
            continue
        if not bool(path.get("drains_to_boundary", False)):
            continue
        drain_indices = tuple(
            sorted(
                {
                    int(idx)
                    for idx in _json_list(path.get("drain_face_indices", []))
                    if 0 <= int(idx) < len(panel.faces)
                }
            )
        )
        if drain_indices and len(drain_indices) < len(panel.faces):
            drain_groups.append(drain_indices)
    if not drain_groups:
        return [panel]
    unique_groups = sorted(set(drain_groups), key=lambda group: (group[0], len(group)))
    child_panels: list[PanelPatch] = []
    covered_indices: set[int] = set()
    for group in unique_groups:
        covered_indices.update(group)
        group_faces = tuple(face for idx, face in enumerate(panel.faces) if idx in group)
        child_panels.extend(_panel_from_faces(panel, group_faces))
    remainder_faces = tuple(
        face for idx, face in enumerate(panel.faces) if idx not in covered_indices
    )
    if remainder_faces:
        child_panels.extend(_panel_from_faces(panel, remainder_faces))
    if len(child_panels) < 2:
        return [panel]
    return child_panels


def _failure_wedge_split_parent_panels(
    panel: PanelPatch,
    failure_field: Mapping[str, object] | None,
) -> list[PanelPatch]:
    """Split two-leg wedge/lens relief corridors without deleting parent fabric."""

    if failure_field is None or not panel.faces:
        return [panel]
    wedge_groups: list[tuple[int, ...]] = []
    selected_wedges = _json_list(failure_field.get("selected_wedge_reliefs", []))
    source_wedges = selected_wedges or _json_list(failure_field.get("candidate_wedge_reliefs", []))
    for wedge in source_wedges:
        if not isinstance(wedge, Mapping):
            continue
        if not bool(wedge.get("creates_wedge_chart", False)):
            continue
        wedge_indices = tuple(
            sorted(
                {
                    int(idx)
                    for idx in _json_list(wedge.get("wedge_face_indices", []))
                    if 0 <= int(idx) < len(panel.faces)
                }
            )
        )
        if wedge_indices and len(wedge_indices) < len(panel.faces):
            wedge_groups.append(wedge_indices)
    if not wedge_groups:
        return [panel]
    unique_groups = sorted(set(wedge_groups), key=lambda group: (group[0], len(group)))
    child_panels: list[PanelPatch] = []
    covered_indices: set[int] = set()
    for group in unique_groups:
        remaining_group = tuple(idx for idx in group if idx not in covered_indices)
        if not remaining_group:
            continue
        covered_indices.update(remaining_group)
        group_faces = tuple(face for idx, face in enumerate(panel.faces) if idx in remaining_group)
        child_panels.extend(_panel_from_faces(panel, group_faces))
    remainder_faces = tuple(
        face for idx, face in enumerate(panel.faces) if idx not in covered_indices
    )
    if remainder_faces:
        child_panels.extend(_panel_from_faces(panel, remainder_faces))
    if len(child_panels) < 2:
        return [panel]
    return child_panels


def _guided_dart_wedge_split_parent_panels(
    panel: PanelPatch,
    failure_field: Mapping[str, object] | None,
) -> list[PanelPatch]:
    """Split dart/lens wedges derived from a measured one-path relief candidate."""

    if failure_field is None or not panel.faces:
        return [panel]
    wedge_groups: list[tuple[int, ...]] = []
    selected_wedges = _json_list(failure_field.get("selected_guided_dart_wedges", []))
    source_wedges = selected_wedges or _json_list(
        failure_field.get("candidate_guided_dart_wedges", [])
    )
    for wedge in source_wedges:
        if not isinstance(wedge, Mapping):
            continue
        if not bool(wedge.get("creates_dart_chart", False)):
            continue
        wedge_indices = tuple(
            sorted(
                {
                    int(idx)
                    for idx in _json_list(wedge.get("wedge_face_indices", []))
                    if 0 <= int(idx) < len(panel.faces)
                }
            )
        )
        if wedge_indices and len(wedge_indices) < len(panel.faces):
            wedge_groups.append(wedge_indices)
    if not wedge_groups:
        return [panel]
    unique_groups = sorted(set(wedge_groups), key=lambda group: (group[0], len(group)))
    child_panels: list[PanelPatch] = []
    covered_indices: set[int] = set()
    for group in unique_groups:
        remaining_group = tuple(idx for idx in group if idx not in covered_indices)
        if not remaining_group:
            continue
        covered_indices.update(remaining_group)
        group_faces = tuple(face for idx, face in enumerate(panel.faces) if idx in remaining_group)
        child_panels.extend(_panel_from_faces(panel, group_faces))
    remainder_faces = tuple(
        face for idx, face in enumerate(panel.faces) if idx not in covered_indices
    )
    if remainder_faces:
        child_panels.extend(_panel_from_faces(panel, remainder_faces))
    if len(child_panels) < 2:
        return [panel]
    return child_panels


def _failure_lens_patch_split_parent_panels(
    panel: PanelPatch,
    failure_field: Mapping[str, object] | None,
) -> list[PanelPatch]:
    """Split a measured failure support into a replacement lens patch chart."""

    if failure_field is None or not panel.faces:
        return [panel]
    lens_groups: list[tuple[int, ...]] = []
    selected_patches = _json_list(failure_field.get("selected_lens_patches", []))
    source_patches = selected_patches or _json_list(failure_field.get("candidate_lens_patches", []))
    for patch in source_patches:
        if not isinstance(patch, Mapping):
            continue
        if not bool(patch.get("creates_patch_chart", False)):
            continue
        patch_indices = tuple(
            sorted(
                {
                    int(idx)
                    for idx in _json_list(patch.get("patch_face_indices", []))
                    if 0 <= int(idx) < len(panel.faces)
                }
            )
        )
        if patch_indices and len(patch_indices) < len(panel.faces):
            lens_groups.append(patch_indices)
    if not lens_groups:
        return [panel]
    unique_groups = sorted(set(lens_groups), key=lambda group: (group[0], len(group)))
    child_panels: list[PanelPatch] = []
    covered_indices: set[int] = set()
    for group in unique_groups:
        remaining_group = tuple(idx for idx in group if idx not in covered_indices)
        if not remaining_group:
            continue
        covered_indices.update(remaining_group)
        group_faces = tuple(face for idx, face in enumerate(panel.faces) if idx in remaining_group)
        child_panels.extend(_panel_from_faces(panel, group_faces))
    remainder_faces = tuple(
        face for idx, face in enumerate(panel.faces) if idx not in covered_indices
    )
    if remainder_faces:
        child_panels.extend(_panel_from_faces(panel, remainder_faces))
    if len(child_panels) < 2:
        return [panel]
    return child_panels


def _serialization_failure_fields_for_panels(
    *,
    vertices: np.ndarray,
    panels: Sequence[PanelPatch],
    uv_by_panel: Mapping[str, np.ndarray],
    selected_backends: Sequence[str],
    seam_edges: Sequence[Edge] | None,
    distortion_threshold: float,
) -> list[dict[str, object]]:
    return [
        _serialization_failure_field_receipt(
            panel_id=f"P{idx}",
            backend=selected_backends[idx],
            vertices=vertices,
            panel=panel,
            uv=uv_by_panel[f"panel_{idx}"],
            seam_edges=seam_edges,
            distortion_threshold=distortion_threshold,
        )
        for idx, panel in enumerate(panels)
        if idx < len(selected_backends) and f"panel_{idx}" in uv_by_panel
    ]


def _materialized_chart_panels(
    *,
    vertices: np.ndarray,
    panels: Sequence[PanelPatch],
    materialization_receipt: Mapping[str, object] | None,
    failure_fields: Sequence[Mapping[str, object] | None] | None = None,
) -> tuple[np.ndarray, list[PanelPatch], list[int], list[str], list[str]]:
    """Return chart domains after materialized operators introduce relief islands."""

    next_vertices = np.asarray(vertices, dtype=float)
    chart_panels: list[PanelPatch] = []
    parent_indices: list[int] = []
    materialization_kinds: list[str] = []
    variant_ids: list[str] = []
    for panel_idx, panel in enumerate(panels):
        if not _materialization_requires_chart_split(materialization_receipt, panel_idx):
            chart_panels.append(panel)
            parent_indices.append(panel_idx)
            materialization_kinds.append("original_chart")
            variant_ids.append("original")
            continue
        branch_vertices = _gusset_branch_vertices_for_panel(materialization_receipt, panel_idx)
        for ring_depth in (1, 2):
            carved_parent_panels = _carved_gusset_parent_panels(
                panel,
                branch_vertices,
                ring_depth=ring_depth,
            )
            if len(carved_parent_panels) == 1 and len(carved_parent_panels[0].faces) == len(
                panel.faces
            ):
                continue
            for carve_idx, carved in enumerate(carved_parent_panels):
                chart_panels.append(carved)
                parent_indices.append(panel_idx)
                materialization_kinds.append(f"gusset_parent_cutout_r{ring_depth}_{carve_idx}")
                variant_ids.append(f"cutout_r{ring_depth}")
        for ring_depth in (1, 2):
            spoke_parent_panels = _branch_spoke_split_parent_panels(
                panel,
                branch_vertices,
                ring_depth=ring_depth,
            )
            if len(spoke_parent_panels) == 1 and len(spoke_parent_panels[0].faces) == len(
                panel.faces
            ):
                continue
            for spoke_idx, spoke in enumerate(spoke_parent_panels):
                chart_panels.append(spoke)
                parent_indices.append(panel_idx)
                materialization_kinds.append(
                    f"gusset_parent_branch_spoke_r{ring_depth}_{spoke_idx}"
                )
                variant_ids.append(f"branch_spoke_r{ring_depth}")
        failure_field = (
            failure_fields[panel_idx]
            if failure_fields is not None and panel_idx < len(failure_fields)
            else None
        )
        failure_relief_panels = _failure_relief_split_parent_panels(panel, failure_field)
        failure_relief_variant_id = _failure_relief_variant_id(failure_field)
        if not (
            len(failure_relief_panels) == 1
            and len(failure_relief_panels[0].faces) == len(panel.faces)
        ):
            for relief_idx, relief_panel in enumerate(failure_relief_panels):
                chart_panels.append(relief_panel)
                parent_indices.append(panel_idx)
                materialization_kinds.append(f"{failure_relief_variant_id}_{relief_idx}")
                variant_ids.append(failure_relief_variant_id)
        failure_drain_panels = _failure_drain_split_parent_panels(panel, failure_field)
        failure_drain_variant_id = _failure_drain_variant_id(failure_field)
        if not (
            len(failure_drain_panels) == 1
            and len(failure_drain_panels[0].faces) == len(panel.faces)
        ):
            for drain_idx, drain_panel in enumerate(failure_drain_panels):
                chart_panels.append(drain_panel)
                parent_indices.append(panel_idx)
                materialization_kinds.append(f"{failure_drain_variant_id}_{drain_idx}")
                variant_ids.append(failure_drain_variant_id)
        failure_wedge_panels = _failure_wedge_split_parent_panels(panel, failure_field)
        failure_wedge_variant_id = _failure_wedge_variant_id(failure_field)
        if not (
            len(failure_wedge_panels) == 1
            and len(failure_wedge_panels[0].faces) == len(panel.faces)
        ):
            for wedge_idx, wedge_panel in enumerate(failure_wedge_panels):
                chart_panels.append(wedge_panel)
                parent_indices.append(panel_idx)
                materialization_kinds.append(f"{failure_wedge_variant_id}_{wedge_idx}")
                variant_ids.append(failure_wedge_variant_id)
        guided_dart_wedge_panels = _guided_dart_wedge_split_parent_panels(panel, failure_field)
        guided_dart_wedge_variant_id = _guided_dart_wedge_variant_id(failure_field)
        if not (
            len(guided_dart_wedge_panels) == 1
            and len(guided_dart_wedge_panels[0].faces) == len(panel.faces)
        ):
            for wedge_idx, wedge_panel in enumerate(guided_dart_wedge_panels):
                chart_panels.append(wedge_panel)
                parent_indices.append(panel_idx)
                materialization_kinds.append(f"{guided_dart_wedge_variant_id}_{wedge_idx}")
                variant_ids.append(guided_dart_wedge_variant_id)
        lens_patch_panels = _failure_lens_patch_split_parent_panels(panel, failure_field)
        lens_patch_variant_id = _failure_lens_patch_variant_id(failure_field)
        if not (
            len(lens_patch_panels) == 1 and len(lens_patch_panels[0].faces) == len(panel.faces)
        ):
            for lens_idx, lens_panel in enumerate(lens_patch_panels):
                chart_panels.append(lens_panel)
                parent_indices.append(panel_idx)
                materialization_kinds.append(f"{lens_patch_variant_id}_{lens_idx}")
                variant_ids.append(lens_patch_variant_id)
        splits = _subdivide_panel(vertices, panel)
        if len(splits) <= 1 and not branch_vertices:
            chart_panels.append(panel)
            parent_indices.append(panel_idx)
            materialization_kinds.append("relief_split_unavailable")
            variant_ids.append("relief_split_unavailable")
            continue
        for split_idx, split in enumerate(splits):
            chart_panels.append(split)
            parent_indices.append(panel_idx)
            materialization_kinds.append(f"materialized_relief_split_{split_idx}")
            variant_ids.append("relief_split")
    next_vertices, inserted_patches, inserted_parent_indices, inserted_kinds = (
        _inserted_gusset_patch_domains(
            vertices=next_vertices,
            panels=panels,
            materialization_receipt=materialization_receipt,
        )
    )
    for patch, parent_idx, kind in zip(
        inserted_patches,
        inserted_parent_indices,
        inserted_kinds,
        strict=False,
    ):
        parent_variants = {
            variant_id
            for source_idx, variant_id in zip(parent_indices, variant_ids, strict=False)
            if source_idx == parent_idx and variant_id != "original"
        }
        if not parent_variants:
            parent_variants = {"inserted_patch_only"}
        for variant_id in sorted(parent_variants):
            chart_panels.append(patch)
            parent_indices.append(parent_idx)
            materialization_kinds.append(kind)
            variant_ids.append(variant_id)
    return next_vertices, chart_panels, parent_indices, materialization_kinds, variant_ids


def _inserted_gusset_patch_domains(
    *,
    vertices: np.ndarray,
    panels: Sequence[PanelPatch],
    materialization_receipt: Mapping[str, object] | None,
) -> tuple[np.ndarray, list[PanelPatch], list[int], list[str]]:
    if materialization_receipt is None:
        return vertices, [], [], []
    next_vertices = np.asarray(vertices, dtype=float)
    patches: list[PanelPatch] = []
    parent_indices: list[int] = []
    kinds: list[str] = []
    for entry in _json_list(materialization_receipt.get("materializations", [])):
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("operator_family", "")) != "gusset_corner":
            continue
        if str(entry.get("materialization_kind", "")) != "inserted_patch":
            continue
        geometry = entry.get("geometry", {})
        if not isinstance(geometry, Mapping):
            continue
        affected = [int(value) for value in _json_list(entry.get("affected_panels", []))]
        branch_vertex = int(geometry.get("branch_vertex", -1))
        if branch_vertex < 0 or branch_vertex >= len(next_vertices) or not affected:
            continue
        patch_vertices = _diamond_gusset_vertices(
            vertices=next_vertices,
            panels=panels,
            affected_panels=affected,
            branch_vertex=branch_vertex,
        )
        start = int(len(next_vertices))
        next_vertices = np.vstack([next_vertices, patch_vertices])
        a, b, c, d = start, start + 1, start + 2, start + 3
        patches.append(
            PanelPatch(
                vertices=(a, b, c, d),
                edges=(
                    _normalize_edge(a, b),
                    _normalize_edge(b, c),
                    _normalize_edge(c, d),
                    _normalize_edge(a, d),
                    _normalize_edge(a, c),
                ),
                faces=((a, b, c), (a, c, d)),
            )
        )
        parent_indices.append(int(affected[0]))
        kinds.append(f"inserted_gusset_patch:{entry.get('node_id', '')}")
    return next_vertices, patches, parent_indices, kinds


def _diamond_gusset_vertices(
    *,
    vertices: np.ndarray,
    panels: Sequence[PanelPatch],
    affected_panels: Sequence[int],
    branch_vertex: int,
) -> np.ndarray:
    center = np.asarray(vertices[branch_vertex], dtype=float)
    neighbor_vectors: list[np.ndarray] = []
    edge_lengths: list[float] = []
    for panel_idx in affected_panels:
        if panel_idx < 0 or panel_idx >= len(panels):
            continue
        for a, b in panels[panel_idx].edges:
            if int(a) == branch_vertex:
                other = int(b)
            elif int(b) == branch_vertex:
                other = int(a)
            else:
                continue
            if 0 <= other < len(vertices):
                vector = np.asarray(vertices[other], dtype=float) - center
                norm = float(np.linalg.norm(vector))
                if norm > 1e-12:
                    neighbor_vectors.append(vector / norm)
                    edge_lengths.append(norm)
    if neighbor_vectors:
        matrix = np.asarray(neighbor_vectors, dtype=float)
        try:
            _u, _s, vt = np.linalg.svd(matrix, full_matrices=False)
            axis_a = vt[0]
            axis_b = vt[1] if len(vt) > 1 else np.array([0.0, 1.0, 0.0])
        except np.linalg.LinAlgError:
            axis_a = neighbor_vectors[0]
            axis_b = np.array([0.0, 1.0, 0.0])
    else:
        axis_a = np.array([1.0, 0.0, 0.0])
        axis_b = np.array([0.0, 1.0, 0.0])
    axis_a = axis_a / (float(np.linalg.norm(axis_a)) or 1.0)
    axis_b = axis_b - axis_a * float(np.dot(axis_b, axis_a))
    if float(np.linalg.norm(axis_b)) <= 1e-12:
        fallback = np.array([0.0, 1.0, 0.0])
        axis_b = fallback - axis_a * float(np.dot(fallback, axis_a))
    axis_b = axis_b / (float(np.linalg.norm(axis_b)) or 1.0)
    radius = 0.35 * (float(np.median(edge_lengths)) if edge_lengths else 1.0)
    radius = max(radius, 1e-3)
    return np.asarray(
        [
            center + radius * axis_a,
            center + radius * axis_b,
            center - radius * axis_a,
            center - radius * axis_b,
        ],
        dtype=float,
    )


def _selected_candidate_score(candidates: Sequence[object], selected_backend: str) -> float:
    for candidate in candidates:
        if getattr(candidate, "backend", None) != selected_backend:
            continue
        score = getattr(candidate, "score", None)
        if score is not None:
            return float(score)
        foldovers = getattr(candidate, "foldovers", None)
        distortion = getattr(candidate, "distortion", None)
        if foldovers is not None and distortion is not None:
            return float(distortion) + 10.0 * float(foldovers)
    return float("inf")


def _selected_candidate_distortion(
    candidates: Sequence[object],
    selected_backend: str,
) -> float:
    for candidate in candidates:
        if getattr(candidate, "backend", None) != selected_backend:
            continue
        distortion = getattr(candidate, "distortion", None)
        if distortion is not None:
            return float(distortion)
    return float("inf")


def _selected_candidate_metric(
    candidates: Sequence[object],
    selected_backend: str,
    attr: str,
    *,
    default: float | int | bool = 0.0,
) -> float | int | bool:
    for candidate in candidates:
        if getattr(candidate, "backend", None) != selected_backend:
            continue
        value = getattr(candidate, attr, default)
        if value is not None:
            return value
    return default


def _expanded_corrected_residuals(
    corrected_residuals: Sequence[float] | None,
    parent_indices: Sequence[int],
    fallback_distortions: Sequence[float],
) -> list[float] | None:
    if corrected_residuals is None:
        return None
    if len(corrected_residuals) == len(parent_indices):
        return [float(value) for value in corrected_residuals]
    expanded: list[float] = []
    for idx, parent_idx in enumerate(parent_indices):
        if 0 <= parent_idx < len(corrected_residuals):
            expanded.append(float(corrected_residuals[parent_idx]))
        else:
            expanded.append(float(fallback_distortions[idx]))
    return expanded


def _variant_metrics_from_group_results(
    group_results: Sequence[tuple[PanelPatch, np.ndarray, float, list[object], str, str]],
) -> dict[str, object]:
    selected_candidates = [
        (
            candidates,
            selected_backend,
        )
        for _panel, _uv, _distortion, candidates, selected_backend, _kind in group_results
    ]
    return {
        "score": float(
            sum(
                _selected_candidate_score(candidates, selected_backend)
                for candidates, selected_backend in selected_candidates
            )
        ),
        "worst_distortion": float(
            max(
                (
                    _selected_candidate_distortion(candidates, selected_backend)
                    for candidates, selected_backend in selected_candidates
                ),
                default=float("inf"),
            )
        ),
        "foldovers": int(
            sum(
                int(_selected_candidate_metric(candidates, selected_backend, "foldovers"))
                for candidates, selected_backend in selected_candidates
            )
        ),
        "boundary_deviation": float(
            sum(
                float(
                    _selected_candidate_metric(
                        candidates,
                        selected_backend,
                        "boundary_deviation",
                    )
                )
                for candidates, selected_backend in selected_candidates
            )
        ),
        "chart_count": int(len(group_results)),
        "exportable": bool(
            all(
                bool(
                    _selected_candidate_metric(
                        candidates, selected_backend, "exportable", default=False
                    )
                )
                for candidates, selected_backend in selected_candidates
            )
        ),
    }


def _selection_profile_score(profile_name: str, metrics: Mapping[str, object]) -> float:
    weights = VARIANT_SELECTION_PROFILES.get(profile_name, VARIANT_SELECTION_PROFILES["default"])
    score = _json_float(metrics.get("score", 0.0))
    return float(
        score
        + weights["worst_distortion"] * _json_float(metrics.get("worst_distortion", 0.0))
        + weights["foldovers"] * _json_int(metrics.get("foldovers", 0))
        + weights["boundary_deviation"] * _json_float(metrics.get("boundary_deviation", 0.0))
        + weights["chart_count"] * _json_int(metrics.get("chart_count", 0))
        + (0.0 if bool(metrics.get("exportable", False)) else weights["exportable_penalty"])
    )


def _metric_comparison_receipt(
    original_metrics: Mapping[str, object],
    variant_metrics: Mapping[str, object],
) -> dict[str, object]:
    improvements: list[str] = []
    regressions: list[str] = []
    for key in PARETO_METRIC_KEYS:
        original_value = _json_float(original_metrics.get(key, 0.0))
        variant_value = _json_float(variant_metrics.get(key, 0.0))
        if variant_value + 1e-12 < original_value:
            improvements.append(key)
        elif variant_value > original_value + 1e-12:
            regressions.append(key)
    exportable_original = bool(original_metrics.get("exportable", False))
    exportable_variant = bool(variant_metrics.get("exportable", False))
    if exportable_variant and not exportable_original:
        improvements.append("exportable")
    elif exportable_original and not exportable_variant:
        regressions.append("exportable")
    dominates = not regressions and bool(improvements)
    useful = bool(improvements)
    return {
        "improves": improvements,
        "regresses": regressions,
        "dominates_original": dominates,
        "pareto_useful": useful,
    }


def _pareto_frontier_candidate_ids(
    candidates: Sequence[Mapping[str, object]],
) -> list[str]:
    frontier_ids: list[str] = []
    for idx, candidate in enumerate(candidates):
        candidate_metrics = candidate.get("metrics")
        if not isinstance(candidate_metrics, Mapping):
            continue
        dominated = False
        for other_idx, other in enumerate(candidates):
            if idx == other_idx:
                continue
            other_metrics = other.get("metrics")
            if not isinstance(other_metrics, Mapping):
                continue
            if _pareto_dominates(other_metrics, candidate_metrics):
                dominated = True
                break
        if not dominated:
            frontier_ids.append(str(candidate.get("candidate_id", f"candidate_{idx}")))
    return frontier_ids


def _pareto_dominates(
    lhs_metrics: Mapping[str, object],
    rhs_metrics: Mapping[str, object],
) -> bool:
    lhs_exportable = bool(lhs_metrics.get("exportable", False))
    rhs_exportable = bool(rhs_metrics.get("exportable", False))
    if lhs_exportable != rhs_exportable:
        return lhs_exportable and not rhs_exportable
    no_worse = all(
        _json_float(lhs_metrics.get(metric, 0.0))
        <= _json_float(rhs_metrics.get(metric, 0.0)) + 1e-12
        for metric in PARETO_METRIC_KEYS
    )
    better = any(
        _json_float(lhs_metrics.get(metric, 0.0)) + 1e-12
        < _json_float(rhs_metrics.get(metric, 0.0))
        for metric in PARETO_METRIC_KEYS
    )
    return bool(no_worse and better)


def _profile_loss_receipt(
    candidate: Mapping[str, object],
    winner: Mapping[str, object],
    *,
    profile_name: str,
) -> dict[str, object]:
    candidate_metrics = cast(Mapping[str, object], candidate.get("metrics", {}))
    winner_metrics = cast(Mapping[str, object], winner.get("metrics", {}))
    candidate_scores = cast(Mapping[str, object], candidate.get("profile_scores", {}))
    winner_scores = cast(Mapping[str, object], winner.get("profile_scores", {}))
    lost_because = {
        "profile_score_delta": _json_float(candidate_scores.get(profile_name, 0.0))
        - _json_float(winner_scores.get(profile_name, 0.0)),
        "score_delta": _json_float(candidate_metrics.get("score", 0.0))
        - _json_float(winner_metrics.get("score", 0.0)),
        "worst_distortion_delta": _json_float(candidate_metrics.get("worst_distortion", 0.0))
        - _json_float(winner_metrics.get("worst_distortion", 0.0)),
        "foldover_delta": _json_int(candidate_metrics.get("foldovers", 0))
        - _json_int(winner_metrics.get("foldovers", 0)),
        "boundary_deviation_delta": _json_float(candidate_metrics.get("boundary_deviation", 0.0))
        - _json_float(winner_metrics.get("boundary_deviation", 0.0)),
        "chart_count_delta": _json_int(candidate_metrics.get("chart_count", 0))
        - _json_int(winner_metrics.get("chart_count", 0)),
        "exportable_delta": int(bool(candidate_metrics.get("exportable", False)))
        - int(bool(winner_metrics.get("exportable", False))),
    }
    return {
        "profile": profile_name,
        "lost_to_candidate_id": str(winner.get("candidate_id", "")),
        "lost_to_variant_id": str(winner.get("variant_id", "")),
        "lost_because": lost_because,
    }


def _variant_pareto_receipt(
    *,
    original_candidate_id: str,
    original_metrics: Mapping[str, object],
    variant_candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    candidates: list[dict[str, object]] = [
        {
            "candidate_id": original_candidate_id,
            "variant_id": "original",
            "metrics": dict(original_metrics),
            "profile_scores": {
                profile: _selection_profile_score(profile, original_metrics)
                for profile in VARIANT_SELECTION_PROFILES
            },
        }
    ]
    for candidate in variant_candidates:
        metrics = candidate.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        candidate_id = str(candidate.get("candidate_id", candidate.get("variant_id", "candidate")))
        candidates.append(
            {
                "candidate_id": candidate_id,
                "variant_id": str(candidate.get("variant_id", candidate_id)),
                "metrics": dict(metrics),
                "profile_scores": {
                    profile: _selection_profile_score(profile, metrics)
                    for profile in VARIANT_SELECTION_PROFILES
                },
            }
        )
    frontier_ids = _pareto_frontier_candidate_ids(candidates)
    profile_best_candidate_ids = {
        profile: min(
            candidates,
            key=lambda candidate: float(candidate["profile_scores"][profile]),
        )["candidate_id"]
        for profile in VARIANT_SELECTION_PROFILES
    }
    best_default_candidate_id = profile_best_candidate_ids["default"]
    default_winner = next(
        candidate
        for candidate in candidates
        if candidate["candidate_id"] == best_default_candidate_id
    )
    return {
        "schema_version": "smii.variant_pareto_receipt.v1",
        "claim_boundary": "pareto_and_profile_scoring_are_diagnostic_only",
        "tracked_metrics": list(PARETO_METRIC_KEYS),
        "profiles": [
            {
                "profile": profile,
                "weights": dict(VARIANT_SELECTION_PROFILES[profile]),
            }
            for profile in VARIANT_SELECTION_PROFILES
        ],
        "candidates": [
            {
                **candidate,
                "pareto_frontier_member": candidate["candidate_id"] in frontier_ids,
                "pareto_status": (
                    "original"
                    if candidate["candidate_id"] == original_candidate_id
                    else ("frontier" if candidate["candidate_id"] in frontier_ids else "dominated")
                ),
                "pareto_useful": (
                    candidate["candidate_id"] != original_candidate_id
                    and _metric_comparison_receipt(original_metrics, candidate["metrics"])[
                        "pareto_useful"
                    ]
                ),
                "metric_comparison": (
                    {}
                    if candidate["candidate_id"] == original_candidate_id
                    else _metric_comparison_receipt(original_metrics, candidate["metrics"])
                ),
                "selected_by_default_profile": candidate["candidate_id"]
                == best_default_candidate_id,
                "default_profile_selection": (
                    {
                        "profile": "default",
                        "selected": True,
                        "lost_to_candidate_id": None,
                        "lost_to_variant_id": None,
                        "lost_because": {},
                    }
                    if candidate["candidate_id"] == best_default_candidate_id
                    else {
                        "selected": False,
                        **_profile_loss_receipt(
                            candidate,
                            default_winner,
                            profile_name="default",
                        ),
                    }
                ),
            }
            for candidate in candidates
        ],
        "frontier_candidate_ids": frontier_ids,
        "profile_best_candidate_ids": profile_best_candidate_ids,
    }


def _operator_family_for_variant(variant_id: str) -> str:
    if variant_id.startswith("failure_lens_patch"):
        return "lens_patch"
    if variant_id.startswith("pareto_guided_dart_wedge"):
        return "dart_wedge"
    if variant_id.startswith("failure_wedge_relief"):
        return "dart_wedge"
    if variant_id.startswith("failure_drain"):
        return "drain_path"
    if variant_id.startswith("failure_relief"):
        return "relief_tree" if variant_id.endswith("_tree") else "relief_path"
    if variant_id.startswith("branch_spoke"):
        return "relief_path"
    if variant_id.startswith("cutout"):
        return "boundary_split"
    if variant_id == "relief_split":
        return "boundary_split"
    if variant_id == "inserted_patch_only":
        return "gusset_patch"
    return variant_id


def _tree_metric_from_deltas(
    original_metrics: Mapping[str, object],
    operator_metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    metrics: dict[str, object] = dict(original_metrics)
    for key in PARETO_METRIC_KEYS:
        original_value = _json_float(original_metrics.get(key, 0.0))
        delta = sum(
            _json_float(operator_metric.get(key, original_value)) - original_value
            for operator_metric in operator_metrics
        )
        value = max(0.0, original_value + delta)
        metrics[key] = int(round(value)) if key in {"foldovers", "chart_count"} else float(value)
    metrics["exportable"] = all(
        bool(operator_metric.get("exportable", False)) for operator_metric in operator_metrics
    )
    return metrics


def _operator_tree_candidate(
    *,
    panel_id: str,
    tree_index: int,
    operators: Sequence[Mapping[str, object]],
    original_metrics: Mapping[str, object],
    materialized: bool,
) -> dict[str, object]:
    variant_ids = [str(operator["variant_id"]) for operator in operators]
    operator_families = [_operator_family_for_variant(variant_id) for variant_id in variant_ids]
    operator_metrics = [cast(Mapping[str, object], operator["metrics"]) for operator in operators]
    metrics = (
        dict(operator_metrics[0])
        if len(operator_metrics) == 1
        else _tree_metric_from_deltas(original_metrics, operator_metrics)
    )
    comparison = _metric_comparison_receipt(original_metrics, metrics)
    hard_gate_passes = bool(
        _json_float(metrics.get("score", float("inf")))
        < _json_float(original_metrics.get("score", 0.0))
        and _json_float(metrics.get("worst_distortion", float("inf")))
        <= _json_float(original_metrics.get("worst_distortion", 0.0)) + 1e-12
        and _json_int(metrics.get("foldovers", 0))
        <= _json_int(original_metrics.get("foldovers", 0))
        and bool(metrics.get("exportable", False))
    )
    blockers: list[str] = []
    if _json_float(metrics.get("score", float("inf"))) >= _json_float(
        original_metrics.get("score", 0.0)
    ):
        blockers.append("score_not_improved")
    if (
        _json_float(metrics.get("worst_distortion", float("inf")))
        > _json_float(original_metrics.get("worst_distortion", 0.0)) + 1e-12
    ):
        blockers.append("distortion_regresses")
    if _json_int(metrics.get("foldovers", 0)) > _json_int(original_metrics.get("foldovers", 0)):
        blockers.append("foldovers_regress")
    if not bool(metrics.get("exportable", False)):
        blockers.append("not_exportable")
    if not materialized:
        blockers.append("operator_tree_not_sequentially_materialized")
    return {
        "tree_id": f"{panel_id}_T{tree_index:03d}",
        "operators": variant_ids,
        "operator_families": operator_families,
        "operator_count": len(variant_ids),
        "materialized": materialized,
        "composition_mode": (
            "measured_single_operator" if len(variant_ids) == 1 else "diagnostic_delta_composition"
        ),
        "metrics": metrics,
        "metric_comparison": comparison,
        "hard_gate_passes": hard_gate_passes,
        "blockers": blockers,
        "source_candidate_ids": [str(operator["candidate_id"]) for operator in operators],
    }


def _operator_basis_search_receipt(
    *,
    panel_id: str,
    original_metrics: Mapping[str, object],
    variant_candidates: Sequence[Mapping[str, object]],
    max_depth: int = OPERATOR_BASIS_SEARCH_DEPTH,
    beam_width: int = OPERATOR_BASIS_SEARCH_BEAM_WIDTH,
) -> dict[str, object]:
    measured_ops = [
        {
            "candidate_id": str(candidate.get("candidate_id", candidate.get("variant_id", ""))),
            "variant_id": str(candidate.get("variant_id", "")),
            "metrics": dict(cast(Mapping[str, object], candidate.get("metrics", {}))),
        }
        for candidate in variant_candidates
        if isinstance(candidate.get("metrics"), Mapping)
        and str(candidate.get("variant_id", "")) not in {"", "original"}
    ]
    tree_candidates: list[dict[str, object]] = []
    tree_index = 0
    for op_idx, operator in enumerate(measured_ops):
        tree_candidates.append(
            _operator_tree_candidate(
                panel_id=panel_id,
                tree_index=tree_index,
                operators=[operator],
                original_metrics=original_metrics,
                materialized=True,
            )
        )
        tree_index += 1
        if max_depth < 2:
            continue
        for other in measured_ops[op_idx + 1 :]:
            if _operator_family_for_variant(
                str(operator["variant_id"])
            ) == _operator_family_for_variant(str(other["variant_id"])):
                continue
            tree_candidates.append(
                _operator_tree_candidate(
                    panel_id=panel_id,
                    tree_index=tree_index,
                    operators=[operator, other],
                    original_metrics=original_metrics,
                    materialized=False,
                )
            )
            tree_index += 1
    ranked = sorted(
        tree_candidates,
        key=lambda candidate: (
            _selection_profile_score("default", cast(Mapping[str, object], candidate["metrics"])),
            _json_float(cast(Mapping[str, object], candidate["metrics"]).get("score", 0.0)),
            _json_int(cast(Mapping[str, object], candidate["metrics"]).get("chart_count", 0)),
            str(candidate["tree_id"]),
        ),
    )
    retained_ids = {str(candidate["tree_id"]) for candidate in ranked[: max(1, int(beam_width))]}
    frontier_ids = set(
        _pareto_frontier_candidate_ids(
            [
                {
                    "candidate_id": str(candidate["tree_id"]),
                    "metrics": cast(Mapping[str, object], candidate["metrics"]),
                }
                for candidate in tree_candidates
            ]
        )
    )
    retained_ids.update(frontier_ids)
    retained = [
        candidate for candidate in tree_candidates if str(candidate["tree_id"]) in retained_ids
    ]
    profile_best_tree_ids = {
        profile: (
            min(
                retained,
                key=lambda candidate: _selection_profile_score(
                    profile,
                    cast(Mapping[str, object], candidate["metrics"]),
                ),
            )["tree_id"]
            if retained
            else None
        )
        for profile in VARIANT_SELECTION_PROFILES
    }
    promoted = [
        candidate
        for candidate in retained
        if bool(candidate["hard_gate_passes"]) and bool(candidate["materialized"])
    ]
    basis_exhausted = bool(measured_ops and not promoted)
    return {
        "schema_version": "smii.operator_basis_search.v1",
        "claim_boundary": "beam_search_is_diagnostic_until_operator_trees_are_sequentially_materialized",
        "panel_id": panel_id,
        "operator_basis": list(OPERATOR_BASIS_FAMILIES),
        "search_depth": int(max_depth),
        "beam_width": int(beam_width),
        "initial_metrics": dict(original_metrics),
        "candidate_count": len(tree_candidates),
        "retained_candidate_count": len(retained),
        "candidates": [
            {
                **candidate,
                "pareto_frontier_member": str(candidate["tree_id"]) in frontier_ids,
                "profile_scores": {
                    profile: _selection_profile_score(
                        profile,
                        cast(Mapping[str, object], candidate["metrics"]),
                    )
                    for profile in VARIANT_SELECTION_PROFILES
                },
            }
            for candidate in retained
        ],
        "best_by_profile": profile_best_tree_ids,
        "promotion": 1 if promoted else 0,
        "blockers": [] if promoted else ["operator_basis_search_no_promoted_tree"],
        "basis_exhausted_at_depth": basis_exhausted,
        "stronger_backend_open": basis_exhausted,
        "larger_operator_basis_open": basis_exhausted,
    }


def _chart_group_backend_serializable(vertices: np.ndarray, panels: Sequence[PanelPatch]) -> bool:
    return all(
        bool(panel_chart_diagnostics(vertices, panel)["backend_serializable"]) for panel in panels
    )


def _unwrap_panel(vertices: np.ndarray, panel: PanelPatch, *, method: str) -> np.ndarray:
    return unwrap_panel_vertices(
        vertices,
        panel_vertices=panel.vertices,
        panel_faces=panel.faces,
        method=method,
    )


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


def _face_edges(faces: Sequence[tuple[int, int, int]]) -> tuple[Edge, ...]:
    edges: set[Edge] = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edge = _normalize_edge(u, v)
            if edge[0] != edge[1]:
                edges.add(edge)
    return tuple(sorted(edges))


def _face_connected_components(
    faces: Sequence[tuple[int, int, int]],
    face_indices: Sequence[int],
) -> list[list[int]]:
    """Group selected faces by shared-edge connectivity."""

    selected = sorted({int(index) for index in face_indices if 0 <= int(index) < len(faces)})
    if not selected:
        return []
    face_vertices = {idx: {int(vertex) for vertex in faces[idx]} for idx in range(len(faces))}
    adjacency: dict[int, set[int]] = {idx: set() for idx in selected}
    for offset, left_idx in enumerate(selected):
        left_vertices = face_vertices[left_idx]
        for right_idx in selected[offset + 1 :]:
            if len(left_vertices.intersection(face_vertices[right_idx])) >= 2:
                adjacency[left_idx].add(right_idx)
                adjacency[right_idx].add(left_idx)
    remaining = set(selected)
    components: list[list[int]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, ()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def _face_adjacency_and_boundary_faces(
    faces: Sequence[tuple[int, int, int]],
) -> tuple[dict[int, set[int]], set[int]]:
    edge_to_faces: dict[Edge, list[int]] = defaultdict(list)
    for face_idx, face in enumerate(faces):
        for edge in _face_edges((face,)):
            edge_to_faces[edge].append(face_idx)
    adjacency: dict[int, set[int]] = {idx: set() for idx in range(len(faces))}
    boundary_faces: set[int] = set()
    for incident_faces in edge_to_faces.values():
        if len(incident_faces) == 1:
            boundary_faces.add(int(incident_faces[0]))
            continue
        for left_offset, left_idx in enumerate(incident_faces):
            for right_idx in incident_faces[left_offset + 1 :]:
                adjacency[int(left_idx)].add(int(right_idx))
                adjacency[int(right_idx)].add(int(left_idx))
    return adjacency, boundary_faces


def _faces_touching_edges(
    faces: Sequence[tuple[int, int, int]],
    edge_set: Sequence[Edge],
) -> list[int]:
    if not faces or not edge_set:
        return []
    edge_lookup = {tuple(sorted((int(a), int(b)))) for a, b in edge_set}
    touching: list[int] = []
    for face_idx, face in enumerate(faces):
        if any(edge in edge_lookup for edge in _face_edges((face,))):
            touching.append(int(face_idx))
    return touching


def _shortest_face_path_to_targets(
    faces: Sequence[tuple[int, int, int]],
    source_indices: Sequence[int],
    target_indices: Sequence[int],
    face_costs: Sequence[float],
) -> list[int]:
    if not faces or not source_indices:
        return []
    adjacency, _ = _face_adjacency_and_boundary_faces(faces)
    source_set = {int(index) for index in source_indices if 0 <= int(index) < len(faces)}
    target_set = {int(index) for index in target_indices if 0 <= int(index) < len(faces)}
    if not source_set:
        return []
    if not target_set:
        return sorted(source_set)
    best_cost: dict[int, float] = {}
    parent: dict[int, int | None] = {}
    heap: list[tuple[float, int]] = []
    for index in sorted(source_set):
        initial_cost = max(0.0, float(face_costs[index])) if index < len(face_costs) else 0.0
        best_cost[index] = initial_cost
        parent[index] = None
        heap.append((initial_cost, index))
    heapq.heapify(heap)
    while heap:
        cost, node = heapq.heappop(heap)
        if cost > best_cost.get(node, float("inf")) + 1e-12:
            continue
        if node in target_set and node not in source_set:
            path: list[int] = [node]
            while parent[path[-1]] is not None:
                path.append(int(parent[path[-1]]))
            path.reverse()
            return path
        for neighbor in sorted(adjacency.get(node, ())):
            step_cost = 1.0 + max(
                0.0,
                float(face_costs[neighbor]) if neighbor < len(face_costs) else 0.0,
            )
            next_cost = cost + step_cost
            if next_cost + 1e-12 < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = next_cost
                parent[neighbor] = node
                heapq.heappush(heap, (next_cost, neighbor))
    return sorted(source_set)


def _face_path_cost(path: Sequence[int], face_costs: Sequence[float]) -> float:
    return float(
        len(path)
        + sum(
            max(0.0, float(face_costs[int(index)]))
            for index in path
            if 0 <= int(index) < len(face_costs)
        )
    )


def _candidate_failure_wedge_reliefs(
    panel: PanelPatch,
    *,
    source: str,
    component_indices: Sequence[int],
    boundary_faces: Sequence[int],
    seam_boundary_faces: Sequence[int],
    face_costs: Sequence[float],
) -> list[dict[str, object]]:
    if not panel.faces or len(panel.faces) < 3:
        return []
    source_set = {int(idx) for idx in component_indices if 0 <= int(idx) < len(panel.faces)}
    if not source_set or len(source_set) >= len(panel.faces):
        return []
    apex_face = max(
        source_set, key=lambda idx: float(face_costs[idx]) if idx < len(face_costs) else 0.0
    )
    target_specs: list[tuple[str, int]] = [
        ("seam_boundary", int(idx))
        for idx in sorted({int(idx) for idx in seam_boundary_faces})
        if int(idx) not in source_set
    ] + [
        ("panel_boundary", int(idx))
        for idx in sorted({int(idx) for idx in boundary_faces})
        if int(idx) not in source_set
    ]
    leg_candidates: list[dict[str, object]] = []
    seen_paths: set[tuple[int, ...]] = set()
    for sink, target_idx in target_specs:
        path = _shortest_face_path_to_targets(
            panel.faces,
            sorted(source_set),
            [target_idx],
            face_costs,
        )
        path_tuple = tuple(int(idx) for idx in path)
        if (
            not path_tuple
            or path_tuple in seen_paths
            or set(path_tuple).issubset(source_set)
            or len(path_tuple) >= len(panel.faces)
        ):
            continue
        seen_paths.add(path_tuple)
        leg_candidates.append(
            {
                "sink": sink,
                "target_face_index": int(target_idx),
                "face_path": list(path_tuple),
                "path_cost": _face_path_cost(path_tuple, face_costs),
            }
        )
    if len(leg_candidates) < 2:
        return []
    leg_candidates.sort(
        key=lambda leg: (
            _drain_sink_priority(str(leg["sink"])),
            float(leg["path_cost"]),
            len(_json_list(leg["face_path"])),
            _json_int(leg["target_face_index"]),
        )
    )
    best_pair: tuple[Mapping[str, object], Mapping[str, object]] | None = None
    best_rank: tuple[float, int, int, int] | None = None
    for left_idx, left in enumerate(leg_candidates):
        for right in leg_candidates[left_idx + 1 :]:
            if _json_int(left["target_face_index"]) == _json_int(right["target_face_index"]):
                continue
            wedge_indices = sorted(
                set(source_set)
                | {int(idx) for idx in _json_list(left["face_path"])}
                | {int(idx) for idx in _json_list(right["face_path"])}
            )
            if len(wedge_indices) >= len(panel.faces):
                continue
            rank = (
                float(left["path_cost"]) + float(right["path_cost"]),
                len(wedge_indices),
                _drain_sink_priority(str(left["sink"])) + _drain_sink_priority(str(right["sink"])),
                _json_int(left["target_face_index"]) + _json_int(right["target_face_index"]),
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_pair = (left, right)
    if best_pair is None:
        return []
    left, right = best_pair
    wedge_indices = sorted(
        set(source_set)
        | {int(idx) for idx in _json_list(left["face_path"])}
        | {int(idx) for idx in _json_list(right["face_path"])}
    )
    selected_faces = [panel.faces[idx] for idx in wedge_indices]
    remainder_faces = [
        face for idx, face in enumerate(panel.faces) if idx not in set(wedge_indices)
    ]
    estimated_chart_count = len(_panel_from_faces(panel, selected_faces)) + len(
        _panel_from_faces(panel, remainder_faces)
    )
    if estimated_chart_count < 2 or estimated_chart_count > 4:
        return []
    leg_a = [int(idx) for idx in _json_list(left["face_path"])]
    leg_b = [int(idx) for idx in _json_list(right["face_path"])]
    return [
        {
            "source": source,
            "operator_family": "wedge_relief",
            "apex_face_index": int(apex_face),
            "sink_pair": [str(left["sink"]), str(right["sink"])],
            "leg_count": 2,
            "leg_a_face_indices": leg_a,
            "leg_b_face_indices": leg_b,
            "failure_face_indices": sorted(source_set),
            "wedge_face_indices": wedge_indices,
            "edge_path": _drain_boundary_edges(selected_faces, remainder_faces),
            "path_length": int(len(set(leg_a) | set(leg_b))),
            "estimated_chart_count": int(estimated_chart_count),
            "creates_wedge_chart": True,
            "separates_bad_region": True,
            "face_partition_preserves_faces": True,
        }
    ]


def _relief_path_neighbor_faces(
    faces: Sequence[tuple[int, int, int]],
    selected_indices: Sequence[int],
) -> list[int]:
    selected_set = {int(idx) for idx in selected_indices if 0 <= int(idx) < len(faces)}
    if not selected_set:
        return []
    adjacency, _ = _face_adjacency_and_boundary_faces(faces)
    neighbors: set[int] = set()
    for face_idx in selected_set:
        neighbors.update(int(idx) for idx in adjacency.get(face_idx, ()) if idx not in selected_set)
    return sorted(neighbors)


def _guided_dart_sink(
    face_idx: int, boundary_faces: set[int], seam_boundary_faces: set[int]
) -> str:
    if face_idx in seam_boundary_faces:
        return "seam_boundary"
    if face_idx in boundary_faces:
        return "panel_boundary"
    return "relief_boundary"


def _candidate_guided_dart_wedges(
    panel: PanelPatch,
    *,
    source: str,
    relief_path: Mapping[str, object],
    boundary_faces: Sequence[int],
    seam_boundary_faces: Sequence[int],
    face_costs: Sequence[float],
) -> list[dict[str, object]]:
    if not panel.faces or len(panel.faces) < 3:
        return []
    failure_indices = sorted(
        {
            int(idx)
            for idx in _json_list(relief_path.get("failure_face_indices", []))
            if 0 <= int(idx) < len(panel.faces)
        }
    )
    if not failure_indices or len(failure_indices) >= len(panel.faces):
        return []
    neighbor_faces = _relief_path_neighbor_faces(panel.faces, failure_indices)
    if len(neighbor_faces) < 2:
        return []
    boundary_set = {int(idx) for idx in boundary_faces if 0 <= int(idx) < len(panel.faces)}
    seam_boundary_set = {
        int(idx) for idx in seam_boundary_faces if 0 <= int(idx) < len(panel.faces)
    }
    ranked_neighbors = sorted(
        neighbor_faces,
        key=lambda idx: (
            _drain_sink_priority(_guided_dart_sink(idx, boundary_set, seam_boundary_set)),
            float(face_costs[idx]) if idx < len(face_costs) else 0.0,
            idx,
        ),
    )[:12]
    apex_face = max(
        failure_indices,
        key=lambda idx: float(face_costs[idx]) if idx < len(face_costs) else 0.0,
    )
    best: dict[str, object] | None = None
    best_rank: tuple[int, float, int, int] | None = None
    for left_offset, left_idx in enumerate(ranked_neighbors):
        for right_idx in ranked_neighbors[left_offset + 1 :]:
            if left_idx == right_idx:
                continue
            leg_a = _shortest_face_path_to_targets(
                panel.faces,
                [apex_face],
                [left_idx],
                face_costs,
            )
            leg_b = _shortest_face_path_to_targets(
                panel.faces,
                [apex_face],
                [right_idx],
                face_costs,
            )
            if not leg_a or not leg_b:
                continue
            wedge_indices = sorted(
                {int(apex_face)}
                | {int(idx) for idx in leg_a}
                | {int(idx) for idx in leg_b}
                | {int(left_idx), int(right_idx)}
            )
            if len(wedge_indices) >= len(panel.faces):
                continue
            if len(wedge_indices) > max(64, len(panel.faces) // 20):
                continue
            selected_faces = [panel.faces[idx] for idx in wedge_indices]
            remainder_faces = [
                face for idx, face in enumerate(panel.faces) if idx not in set(wedge_indices)
            ]
            estimated_chart_count = len(_panel_from_faces(panel, selected_faces)) + len(
                _panel_from_faces(panel, remainder_faces)
            )
            if estimated_chart_count < 2 or estimated_chart_count > 4:
                continue
            sink_pair = [
                _guided_dart_sink(left_idx, boundary_set, seam_boundary_set),
                _guided_dart_sink(right_idx, boundary_set, seam_boundary_set),
            ]
            rank = (
                sum(_drain_sink_priority(sink) for sink in sink_pair),
                _face_path_cost(leg_a, face_costs) + _face_path_cost(leg_b, face_costs),
                len(wedge_indices),
                int(left_idx) + int(right_idx),
            )
            candidate = {
                "source": source,
                "operator_family": "guided_dart_wedge",
                "guide_variant": "failure_relief_path",
                "guide_edge_path": _json_list(relief_path.get("edge_path", [])),
                "apex_face_index": int(apex_face),
                "sink_pair": sink_pair,
                "leg_count": 2,
                "leg_a_face_indices": [int(idx) for idx in leg_a],
                "leg_b_face_indices": [int(idx) for idx in leg_b],
                "failure_face_indices": failure_indices,
                "covered_failure_face_indices": [
                    int(idx) for idx in wedge_indices if idx in set(failure_indices)
                ],
                "wedge_face_indices": wedge_indices,
                "edge_path": _drain_boundary_edges(selected_faces, remainder_faces),
                "path_length": int(len(set(leg_a) | set(leg_b))),
                "estimated_chart_count": int(estimated_chart_count),
                "creates_dart_chart": True,
                "creates_wedge_chart": True,
                "derived_from_single_path": True,
                "separates_bad_region": True,
                "face_partition_preserves_faces": True,
            }
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = candidate
    return [] if best is None else [best]


def _bounded_lens_support_faces(
    faces: Sequence[tuple[int, int, int]],
    seeds: Sequence[int],
    face_costs: Sequence[float],
    *,
    max_faces: int,
) -> list[int]:
    """Return a connected, high-cost support around measured failure seeds."""

    valid_seeds = {int(idx) for idx in seeds if 0 <= int(idx) < len(faces) and len(faces) > 1}
    if not valid_seeds:
        return []
    adjacency, _ = _face_adjacency_and_boundary_faces(faces)
    anchor = max(
        valid_seeds,
        key=lambda idx: (float(face_costs[idx]) if idx < len(face_costs) else 0.0, -idx),
    )
    selected = {int(anchor)}
    frontier = set(adjacency.get(anchor, ()))
    target_size = max(1, min(int(max_faces), len(faces) - 1))
    preferred = set(valid_seeds)
    while frontier and len(selected) < target_size:
        next_face = min(
            frontier,
            key=lambda idx: (
                0 if idx in preferred else 1,
                -(float(face_costs[idx]) if idx < len(face_costs) else 0.0),
                idx,
            ),
        )
        frontier.remove(next_face)
        selected.add(int(next_face))
        frontier.update(
            int(neighbor)
            for neighbor in adjacency.get(int(next_face), ())
            if int(neighbor) not in selected
        )
    return sorted(selected)


def _candidate_failure_lens_patches(
    panel: PanelPatch,
    *,
    source: str,
    relief_path: Mapping[str, object],
    face_costs: Sequence[float],
) -> list[dict[str, object]]:
    if not panel.faces or len(panel.faces) < 3:
        return []
    failure_indices = sorted(
        {
            int(idx)
            for idx in _json_list(relief_path.get("failure_face_indices", []))
            if 0 <= int(idx) < len(panel.faces)
        }
    )
    if not failure_indices or len(failure_indices) >= len(panel.faces):
        return []
    neighbor_faces = _relief_path_neighbor_faces(panel.faces, failure_indices)
    seed_pool = sorted(set(failure_indices) | set(neighbor_faces))
    max_faces = max(8, min(128, len(panel.faces) // 8 if len(panel.faces) >= 80 else 16))
    max_faces = min(max_faces, len(panel.faces) - 1)
    patch_indices = _bounded_lens_support_faces(
        panel.faces,
        seed_pool,
        face_costs,
        max_faces=max_faces,
    )
    if not patch_indices or len(patch_indices) >= len(panel.faces):
        return []
    selected_faces = [panel.faces[idx] for idx in patch_indices]
    remainder_faces = [
        face for idx, face in enumerate(panel.faces) if idx not in set(patch_indices)
    ]
    patch_charts = _panel_from_faces(panel, selected_faces)
    remainder_charts = _panel_from_faces(panel, remainder_faces)
    estimated_chart_count = len(patch_charts) + len(remainder_charts)
    if len(patch_charts) != 1 or estimated_chart_count < 2 or estimated_chart_count > 4:
        return []
    covered_failure = [int(idx) for idx in patch_indices if idx in set(failure_indices)]
    if not covered_failure:
        return []
    return [
        {
            "source": source,
            "operator_family": "lens_patch",
            "guide_variant": "failure_relief_path",
            "patch_shape": "lens",
            "patch_face_indices": patch_indices,
            "support_region_face_indices": patch_indices,
            "failure_face_indices": failure_indices,
            "covered_failure_face_indices": covered_failure,
            "attachment_boundary_edges": _drain_boundary_edges(selected_faces, remainder_faces),
            "edge_path": _drain_boundary_edges(selected_faces, remainder_faces),
            "estimated_chart_count": int(estimated_chart_count),
            "creates_patch_chart": True,
            "replaces_parent_region": True,
            "boundary_compatible": True,
            "allowance_rule": "patch_boundary_allowance",
            "face_partition_preserves_faces": True,
            "no_unexplained_face_loss": True,
        }
    ]


def _wedge_candidate_rank(candidate: Mapping[str, object]) -> tuple[int, int, int, int]:
    sink_pair = [str(sink) for sink in _json_list(candidate.get("sink_pair", []))]
    sink_rank = sum(_drain_sink_priority(sink) for sink in sink_pair)
    failure_faces = len(_json_list(candidate.get("failure_face_indices", [])))
    path_length = _json_int(candidate.get("path_length", 0))
    wedge_length = len(_json_list(candidate.get("wedge_face_indices", [])))
    return (
        sink_rank,
        -failure_faces,
        path_length,
        wedge_length,
    )


def _selected_failure_wedge_reliefs(
    failure_field: Mapping[str, object] | None,
    *,
    max_wedges: int = 1,
) -> list[dict[str, object]]:
    if failure_field is None:
        return []
    candidates = [
        path
        for path in _json_list(failure_field.get("candidate_wedge_reliefs", []))
        if isinstance(path, Mapping) and bool(path.get("creates_wedge_chart", False))
    ]
    if not candidates:
        return []
    ranked = sorted(candidates, key=_wedge_candidate_rank)
    selected: list[dict[str, object]] = []
    covered: set[int] = set()
    for candidate in ranked:
        wedge_indices = {
            int(idx)
            for idx in _json_list(candidate.get("wedge_face_indices", []))
            if int(idx) not in covered
        }
        if not wedge_indices:
            continue
        selected.append(dict(candidate))
        covered.update(wedge_indices)
        if len(selected) >= max(1, int(max_wedges)):
            break
    return selected


def _guided_dart_candidate_rank(candidate: Mapping[str, object]) -> tuple[int, int, int, int]:
    sink_pair = [str(sink) for sink in _json_list(candidate.get("sink_pair", []))]
    sink_rank = sum(_drain_sink_priority(sink) for sink in sink_pair)
    failure_faces = len(_json_list(candidate.get("failure_face_indices", [])))
    path_length = _json_int(candidate.get("path_length", 0))
    wedge_length = len(_json_list(candidate.get("wedge_face_indices", [])))
    return (
        sink_rank,
        -failure_faces,
        path_length,
        wedge_length,
    )


def _selected_guided_dart_wedges(
    failure_field: Mapping[str, object] | None,
    *,
    max_wedges: int = 1,
) -> list[dict[str, object]]:
    if failure_field is None:
        return []
    candidates = [
        path
        for path in _json_list(failure_field.get("candidate_guided_dart_wedges", []))
        if isinstance(path, Mapping) and bool(path.get("creates_dart_chart", False))
    ]
    if not candidates:
        return []
    ranked = sorted(candidates, key=_guided_dart_candidate_rank)
    selected: list[dict[str, object]] = []
    covered: set[int] = set()
    for candidate in ranked:
        wedge_indices = {
            int(idx)
            for idx in _json_list(candidate.get("wedge_face_indices", []))
            if int(idx) not in covered
        }
        if not wedge_indices:
            continue
        selected.append(dict(candidate))
        covered.update(wedge_indices)
        if len(selected) >= max(1, int(max_wedges)):
            break
    return selected


def _lens_patch_candidate_rank(candidate: Mapping[str, object]) -> tuple[int, int, int, int]:
    failure_faces = len(_json_list(candidate.get("failure_face_indices", [])))
    covered_failure_faces = len(_json_list(candidate.get("covered_failure_face_indices", [])))
    patch_faces = len(_json_list(candidate.get("patch_face_indices", [])))
    estimated_chart_count = _json_int(candidate.get("estimated_chart_count", 99))
    return (
        -covered_failure_faces,
        -failure_faces,
        patch_faces,
        estimated_chart_count,
    )


def _selected_failure_lens_patches(
    failure_field: Mapping[str, object] | None,
    *,
    max_patches: int = 1,
) -> list[dict[str, object]]:
    if failure_field is None:
        return []
    candidates = [
        patch
        for patch in _json_list(failure_field.get("candidate_lens_patches", []))
        if isinstance(patch, Mapping) and bool(patch.get("creates_patch_chart", False))
    ]
    if not candidates:
        return []
    ranked = sorted(candidates, key=_lens_patch_candidate_rank)
    selected: list[dict[str, object]] = []
    covered: set[int] = set()
    for candidate in ranked:
        patch_indices = {
            int(idx)
            for idx in _json_list(candidate.get("patch_face_indices", []))
            if int(idx) not in covered
        }
        if not patch_indices:
            continue
        selected.append(dict(candidate))
        covered.update(patch_indices)
        if len(selected) >= max(1, int(max_patches)):
            break
    return selected


def _drain_sink_priority(sink: str) -> int:
    if sink == "seam_boundary":
        return 0
    if sink == "panel_boundary":
        return 1
    return 2


def _drain_candidate_rank(candidate: Mapping[str, object]) -> tuple[int, int, int, int]:
    sink = str(candidate.get("sink", "panel_boundary"))
    failure_faces = len(_json_list(candidate.get("failure_face_indices", [])))
    path_length = _json_int(candidate.get("path_length", 0))
    drain_length = len(_json_list(candidate.get("drain_face_indices", [])))
    return (
        _drain_sink_priority(sink),
        -failure_faces,
        path_length,
        drain_length,
    )


def _selected_failure_drain_paths(
    failure_field: Mapping[str, object] | None,
    *,
    max_paths: int = 1,
) -> list[dict[str, object]]:
    if failure_field is None:
        return []
    candidates = [
        path
        for path in _json_list(failure_field.get("candidate_drain_paths", []))
        if isinstance(path, Mapping) and bool(path.get("drains_to_boundary", False))
    ]
    if not candidates:
        return []
    ranked = sorted(candidates, key=_drain_candidate_rank)
    selected: list[dict[str, object]] = []
    covered: set[int] = set()
    for candidate in ranked:
        drain_indices = {
            int(idx)
            for idx in _json_list(candidate.get("drain_face_indices", []))
            if int(idx) not in covered
        }
        if not drain_indices:
            continue
        selected.append(dict(candidate))
        covered.update(drain_indices)
        if len(selected) >= max(1, int(max_paths)):
            break
    return selected


def _panel_from_faces(
    panel: PanelPatch,
    face_subset: Sequence[tuple[int, int, int]],
) -> list[PanelPatch]:
    """Build connected face-backed subpanels from a face subset."""

    local_faces = tuple(face_subset)
    if not local_faces:
        return []
    local_edges = _face_edges(local_faces)
    local_vertices = sorted({vertex for face in local_faces for vertex in face})

    graph = _adjacency(local_edges)
    remaining = set(local_vertices)
    patches: list[PanelPatch] = []
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
        component_edges = tuple(
            edge for edge in local_edges if edge[0] in component and edge[1] in component
        )
        component_faces = tuple(
            face for face in local_faces if all(vertex in component for vertex in face)
        )
        if component_faces:
            patches.append(
                PanelPatch(
                    vertices=tuple(sorted(component)),
                    edges=component_edges,
                    faces=component_faces,
                )
            )
    return patches


def _subdivide_panel(vertices: np.ndarray, panel: PanelPatch) -> list[PanelPatch]:
    """Deterministically split a high-distortion panel into induced graph patches."""

    if len(panel.faces) < 2:
        return [panel]
    face_centroids = np.asarray(
        [vertices[list(face)].mean(axis=0) for face in panel.faces],
        dtype=float,
    )
    centered = face_centroids - face_centroids.mean(axis=0)
    if not np.isfinite(centered).all():
        return [panel]
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
    except np.linalg.LinAlgError:
        ranges = np.ptp(face_centroids, axis=0)
        axis = np.eye(3)[int(np.argmax(ranges))]
    projection = centered @ axis
    ordered = [
        panel.faces[face_idx]
        for _, face_idx in sorted(
            (float(projection[idx]), idx) for idx, _face in enumerate(panel.faces)
        )
    ]
    midpoint = len(ordered) // 2
    left = ordered[:midpoint]
    right = ordered[midpoint:]
    if not left or not right:
        return [panel]
    patches = _panel_from_faces(panel, left) + _panel_from_faces(panel, right)
    if len(patches) < 2:
        return [panel]
    return patches


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
    cut_topology_receipt_path: Path | None = None,
    distortion_threshold: float = 0.05,
    max_subdivisions: int = 3,
    solver: str = BOOTSTRAP_BACKEND,
    corrections_path: Path | None = None,
    metric_correction_receipt_path: Path | None = None,
    fabric_profile_path: Path | None = None,
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
    if corrections_path is None and solver_receipt.correction_payload_hash is not None:
        candidate_path = solver_receipt_path.parent / "corrections.json"
        if candidate_path.exists():
            corrections_path = candidate_path
    cut_topology_receipt_hash: str | None = None
    cut_topology_typed_operator_count = 0
    cut_topology_blockers: list[str] | None = None
    cut_topology_panels_are_disks: bool | None = None
    if cut_topology_receipt_path is not None:
        cut_topology_receipt = load_cut_topology_receipt(cut_topology_receipt_path)
        if not can_consume_cut_topology_receipt(cut_topology_receipt, "panel_unwrap"):
            raise ValueError(
                f"CutTopologyReceipt not promoted ({cut_topology_receipt.promotion}). "
                f"Blocked: {cut_topology_receipt.blocked_consumers}"
            )
        if cut_topology_receipt.solver_receipt_hash != _sha256_file(solver_receipt_path):
            raise ValueError(
                "CutTopologyReceipt.solver_receipt_hash does not match solver receipt."
            )
        if cut_topology_receipt.seam_edges_hash != _sha256_file(seam_edges_path):
            raise ValueError("CutTopologyReceipt.seam_edges_hash does not match seam edges.")
        if cut_topology_receipt.mesh_hash != _sha256_file(mesh_path):
            raise ValueError("CutTopologyReceipt.mesh_hash does not match mesh.")
        cut_topology_receipt_hash = _sha256_file(cut_topology_receipt_path)
        cut_topology_typed_operator_count = int(cut_topology_receipt.typed_operator_count)
        cut_topology_blockers = list(cut_topology_receipt.cut_topology_blockers)
        cut_topology_panels_are_disks = bool(cut_topology_receipt.panels_are_disks)

    vertices, faces = _load_mesh(mesh_path)
    seam_payload = _load_seam_payload(seam_edges_path)
    seam_edges = seam_payload.edges
    if seam_payload.face_labels is not None:
        panels = _extract_panels_from_face_labels(
            faces=faces,
            face_labels=seam_payload.face_labels,
        )
    else:
        panels = _extract_panels(
            vertex_count=int(vertices.shape[0]),
            faces=faces,
            seam_edges=seam_edges,
        )
    if not panels:
        raise ValueError("No panels could be extracted from seam topology.")
    subdivision_limit = 0 if seam_payload.face_labels is not None else max(0, int(max_subdivisions))
    candidate_backends = tuple(dict.fromkeys((BOOTSTRAP_BACKEND, *PANEL_SERIALIZATION_BACKENDS)))

    per_panel_distortion: list[float] = []
    per_panel_uv: dict[str, np.ndarray] = {}
    per_panel_grain: list[str] = []
    per_panel_candidate_receipts: list[list[object]] = []
    per_panel_patches: list[PanelPatch] = []
    selected_backend_per_panel: list[str] = []
    subdivisions_used = 0
    unwrap_started_at = time.monotonic()
    completed_candidate_count = 0
    estimated_candidate_count = max(1, len(panels) * len(candidate_backends))

    preliminary_correction_tree = _correction_tree_receipt(
        seam_edges,
        typed_operator_count=cut_topology_typed_operator_count,
    )

    def progress(message: str) -> None:
        elapsed = time.monotonic() - unwrap_started_at
        eta = (
            elapsed
            / completed_candidate_count
            * max(0, estimated_candidate_count - completed_candidate_count)
            if completed_candidate_count
            else 0.0
        )
        print(
            "[panel unwrap] "
            f"{message} "
            f"(elapsed {_format_duration(elapsed)}, eta {_format_duration(eta)})",
            flush=True,
        )

    def compete_panel(
        panel: PanelPatch,
        *,
        panel_label: str,
        depth: int,
        correction_tree_for_competition: Mapping[str, object],
        materialization_constraints: Mapping[str, object] | None = None,
    ) -> tuple[np.ndarray, float, list[object], str]:
        nonlocal completed_candidate_count

        candidates = []
        uv_by_backend: dict[str, np.ndarray] = {}
        for backend in candidate_backends:
            backend_started_at = time.monotonic()
            progress(
                f"{panel_label} depth={depth} backend={backend} starting "
                f"vertices={len(panel.vertices)} faces={len(panel.faces)}"
            )
            candidate, candidate_uv = serialize_panel(
                vertices=vertices,
                panel=panel,
                correction_tree=correction_tree_for_competition,
                materialization_constraints=materialization_constraints,
                backend=backend,
                distortion_threshold=distortion_threshold,
            )
            completed_candidate_count += 1
            candidates.append(candidate)
            if candidate_uv is not None:
                uv_by_backend[candidate.backend] = candidate_uv
            progress(
                f"{panel_label} depth={depth} backend={backend} finished "
                f"duration={_format_duration(time.monotonic() - backend_started_at)} "
                f"distortion={candidate.distortion} foldovers={candidate.foldovers} "
                f"promoted={candidate.promoted} blockers={','.join(candidate.blockers)}"
            )
        selected = select_serialization_candidate(candidates)
        selected_uv = uv_by_backend.get(selected.backend)
        if selected_uv is None:
            selected_uv = _unwrap_panel(vertices, panel, method=BOOTSTRAP_BACKEND)
        distortion = (
            float(selected.distortion)
            if selected.distortion is not None
            else _compute_distortion(vertices, panel, selected_uv)
        )
        progress(
            f"{panel_label} depth={depth} selected_backend={selected.backend} "
            f"selected_distortion={distortion}"
        )
        return selected_uv, distortion, candidates, selected.backend

    def record_panel(
        panel: PanelPatch,
        uv: np.ndarray,
        distortion: float,
        depth: int,
        candidates: list[object],
        selected_backend: str,
    ) -> None:
        nonlocal subdivisions_used

        panel_id = len(per_panel_distortion)
        per_panel_distortion.append(float(distortion))
        per_panel_uv[f"panel_{panel_id}"] = uv
        per_panel_grain.append(_infer_grain_direction(vertices, panel, uv))
        per_panel_candidate_receipts.append(candidates)
        per_panel_patches.append(panel)
        selected_backend_per_panel.append(selected_backend)
        subdivisions_used = max(subdivisions_used, depth)

    def process_panel(panel: PanelPatch, depth: int) -> None:
        nonlocal estimated_candidate_count, subdivisions_used

        panel_label = f"P{len(per_panel_distortion)}"
        uv, distortion, candidates, selected_backend = compete_panel(
            panel,
            panel_label=panel_label,
            depth=depth,
            correction_tree_for_competition=preliminary_correction_tree,
        )
        if distortion > distortion_threshold and depth < subdivision_limit:
            subdivisions = _subdivide_panel(vertices, panel)
            if len(subdivisions) > 1:
                estimated_candidate_count += max(0, len(subdivisions) * len(candidate_backends))
                child_distortions = [
                    compete_panel(
                        child,
                        panel_label=f"{panel_label}.{child_idx}",
                        depth=depth + 1,
                        correction_tree_for_competition=preliminary_correction_tree,
                    )[1]
                    for child_idx, child in enumerate(subdivisions)
                ]
                if child_distortions and max(child_distortions) < distortion:
                    subdivisions_used = max(subdivisions_used, depth + 1)
                    for child in subdivisions:
                        process_panel(child, depth + 1)
                    return
        record_panel(panel, uv, distortion, depth, candidates, selected_backend)

    for panel in panels:
        process_panel(panel, 0)

    output_dir.mkdir(parents=True, exist_ok=True)
    uv_path = output_dir / "panel_uvs.npz"
    np.savez_compressed(uv_path, **per_panel_uv)  # type: ignore[arg-type]
    uv_hash = _sha256_file(uv_path)

    worst = float(max(per_panel_distortion))
    mean = float(sum(per_panel_distortion) / len(per_panel_distortion))
    distortion_margin = float(distortion_threshold - worst)
    panel_unwrap_blockers: list[str] = []
    if distortion_margin < 0.0:
        panel_unwrap_blockers.append("distortion_exceeds_threshold")
    correction_tree = preliminary_correction_tree
    if cut_topology_blockers is None:
        panel_unwrap_blockers.extend(_seam_graph_blockers(seam_edges))
        panel_unwrap_blockers.extend(
            _cut_topology_blockers(
                vertices=vertices,
                panels=panels,
            )
        )
    else:
        panel_unwrap_blockers.extend(cut_topology_blockers)
    panels_all_disks = (
        solver_receipt.panels_are_disks
        if cut_topology_panels_are_disks is None
        else cut_topology_panels_are_disks
    )
    if not panels_all_disks:
        panel_unwrap_blockers.append("panels_not_disks")
    panel_unwrap_blockers = list(dict.fromkeys(panel_unwrap_blockers))
    corrected_residuals, correction_payload_hash, correction_blockers = _load_corrected_residuals(
        corrections_path,
        expected_hash=solver_receipt.correction_payload_hash,
        panel_count=len(per_panel_distortion),
    )
    metric_correction_receipt_hash: str | None = None
    metric_residual_gate = float(distortion_threshold)
    if metric_correction_receipt_path is not None:
        metric_receipt = load_metric_correction_receipt(metric_correction_receipt_path)
        metric_residual_gate = float(metric_receipt.residual_gate)
        metric_correction_receipt_hash = _sha256_file(metric_correction_receipt_path)
        if not can_consume_metric_correction_receipt(metric_receipt, "panel_unwrap"):
            correction_blockers.append("metric_correction_receipt_not_promoted")
        if metric_receipt.solver_receipt_hash != _sha256_file(solver_receipt_path):
            correction_blockers.append("metric_solver_receipt_hash_mismatch")
        if cut_topology_receipt_hash is None:
            correction_blockers.append("missingPanelUnwrapCompatibility")
        elif metric_receipt.cut_topology_receipt_hash != cut_topology_receipt_hash:
            correction_blockers.append("metric_cut_topology_receipt_hash_mismatch")
        if metric_receipt.seam_edges_hash != _sha256_file(seam_edges_path):
            correction_blockers.append("metric_seam_edges_hash_mismatch")
        correction_blockers.extend(metric_receipt.metric_correction_blockers)
        if any(entry.delta_metric_meaning == "proxy" for entry in metric_receipt.corrections):
            correction_blockers.append("missingDeltaMetricMeaning")
        incompatible_types = [
            entry.correction_type
            for entry in metric_receipt.corrections
            if entry.correction_type not in UNWRAP_COMPATIBLE_CORRECTION_TYPES
        ]
        if incompatible_types:
            correction_blockers.append("missingPanelUnwrapCompatibility")
        incompatible_states = [
            entry.result_state
            for entry in metric_receipt.corrections
            if entry.result_state not in UNWRAP_COMPATIBLE_CORRECTION_STATES
        ]
        if incompatible_states:
            correction_blockers.append("metric_correction_result_not_accepted")
        if metric_receipt.corrected_residual_total > metric_receipt.residual_gate:
            correction_blockers.append("metric_corrected_residual_exceeds_gate")
        by_panel = {
            entry.panel_label: entry.corrected_residual for entry in metric_receipt.corrections
        }
        if any(value > metric_receipt.residual_gate for value in by_panel.values()):
            correction_blockers.append("metric_corrected_residual_exceeds_gate")
        if len(by_panel) > len(per_panel_distortion) or any(
            panel_label >= len(per_panel_distortion) for panel_label in by_panel
        ):
            correction_blockers.append("correction_panel_count_mismatch")
        elif by_panel:
            corrected_residuals = [
                float(by_panel.get(idx, per_panel_distortion[idx]))
                for idx in range(len(per_panel_distortion))
            ]
            correction_payload_hash = (
                metric_receipt.correction_payload_hash or metric_correction_receipt_hash
            )
    elif cut_topology_typed_operator_count > 0:
        correction_blockers.extend(
            [
                "missingShapingIntentReceipt",
                "missingDeltaMetricMeaning",
                "missingPanelUnwrapCompatibility",
            ]
        )
    panel_unwrap_blockers.extend(correction_blockers)
    panel_unwrap_blockers = list(dict.fromkeys(panel_unwrap_blockers))
    if corrected_residuals is not None and len(corrected_residuals) != len(per_panel_distortion):
        corrected_residuals = [float(value) for value in per_panel_distortion]
    worst_corrected = float(max(corrected_residuals)) if corrected_residuals is not None else None
    mean_corrected = (
        float(sum(corrected_residuals) / len(corrected_residuals))
        if corrected_residuals is not None
        else None
    )
    fabric_metric = None
    correction_operator_scoring = None
    realized_correction_operator = None
    correction_tree_materialization = None
    if fabric_profile_path is not None:
        fabric_metric = _fabric_metric_receipt(
            fabric_profile_path=fabric_profile_path,
            distortions=per_panel_distortion,
            corrected_residuals=corrected_residuals,
            grain_directions=per_panel_grain,
        )
        if correction_tree["branch_count"]:
            fabric_profile = load_fabric_profile(fabric_profile_path)
            residual_before = (
                float(worst_corrected) if worst_corrected is not None else float(worst)
            )
            correction_operator_scoring = price_correction_operator_tree(
                seam_edges=seam_edges,
                vertices=vertices,
                fabric_profile=fabric_profile,
                residual_before=residual_before,
                fabric_violation_before=_json_float(fabric_metric["worst_fabric_violation"]),
                typed_operator_count=cut_topology_typed_operator_count,
            )
            correction_tree = _correction_tree_receipt(
                seam_edges,
                typed_operator_count=cut_topology_typed_operator_count,
                operator_scoring_receipt=correction_operator_scoring,
            )
            fabric_metric["after_operator_tree"] = {
                "estimated_worst_residual_after": correction_operator_scoring[
                    "estimated_worst_residual_after"
                ],
                "estimated_worst_fabric_violation_after": correction_operator_scoring[
                    "estimated_worst_fabric_violation_after"
                ],
                "operator_scoring_promotion": correction_operator_scoring["promotion"],
            }
            if correction_operator_scoring["promotion"] == 1:
                panel_unwrap_blockers = [
                    blocker
                    for blocker in panel_unwrap_blockers
                    if blocker != "unpriced_correction_tree_node"
                ]
                realized_correction_operator = _realized_correction_operator_receipt(
                    operator_scoring_receipt=correction_operator_scoring,
                    fabric_metric_receipt=fabric_metric,
                    residual_gate=metric_residual_gate,
                )
                next_correction_tree_hash = _sha256_json(correction_tree)
                correction_tree_materialization = _correction_tree_materialization_receipt(
                    correction_tree_hash=next_correction_tree_hash or "correction-tree-sha256",
                    correction_tree_receipt_hash=next_correction_tree_hash
                    or "correction-tree-receipt-sha256",
                    correction_operator_scoring_receipt_hash=_sha256_json(
                        correction_operator_scoring
                    ),
                    realized_correction_operator_receipt=realized_correction_operator,
                    panels=per_panel_patches,
                )
                fabric_metric["realized_correction_operator"] = {
                    "realization_promotion": realized_correction_operator["promotion"],
                    "realized_worst_fabric_violation_after": realized_correction_operator[
                        "realized_worst_fabric_violation_after"
                    ],
                    "realized_worst_residual_after": realized_correction_operator[
                        "realized_worst_residual_after"
                    ],
                    "estimate_realization_delta": realized_correction_operator[
                        "estimate_realization_delta"
                    ],
                }
        fabric_violation_after = (
            _json_float(
                cast(Mapping[str, object], fabric_metric["realized_correction_operator"])[
                    "realized_worst_fabric_violation_after"
                ]
            )
            if isinstance(fabric_metric.get("realized_correction_operator"), Mapping)
            else _json_float(
                cast(Mapping[str, object], fabric_metric["after_operator_tree"])[
                    "estimated_worst_fabric_violation_after"
                ]
            )
            if isinstance(fabric_metric.get("after_operator_tree"), Mapping)
            else _json_float(fabric_metric["worst_fabric_violation"])
        )
        fabric_violation_gate = (
            0.02
            if realized_correction_operator is not None
            and realized_correction_operator["promotion"] == 1
            else 1e-12
        )
        realized_residual_after = (
            _json_float(realized_correction_operator["realized_worst_residual_after"])
            if realized_correction_operator is not None
            else None
        )
        if realized_residual_after is not None and realized_residual_after <= metric_residual_gate:
            panel_unwrap_blockers = [
                blocker
                for blocker in panel_unwrap_blockers
                if blocker != "metric_corrected_residual_exceeds_gate"
            ]
        fabric_metric["fabric_violation_gate"] = fabric_violation_gate
        fabric_metric["promotion"] = 1 if fabric_violation_after <= fabric_violation_gate else 0
        fabric_metric["blockers"] = (
            ["fabric_metric_violation_exceeds_profile"]
            if fabric_violation_after > fabric_violation_gate
            else []
        )
        panel_unwrap_blockers.extend(
            str(blocker) for blocker in _json_list(fabric_metric["blockers"])
        )
        if correction_operator_scoring is not None:
            panel_unwrap_blockers.extend(
                str(blocker) for blocker in _json_list(correction_operator_scoring["blockers"])
            )
        if realized_correction_operator is not None:
            panel_unwrap_blockers.extend(
                str(blocker) for blocker in _json_list(realized_correction_operator["blockers"])
            )
        if correction_tree_materialization is not None:
            panel_unwrap_blockers.extend(
                str(blocker) for blocker in _json_list(correction_tree_materialization["blockers"])
            )
            if int(correction_tree_materialization.get("promotion", 0)) != 1:
                panel_unwrap_blockers.append("unmaterialized_correction_operator")
        panel_unwrap_blockers = list(dict.fromkeys(panel_unwrap_blockers))
    if (
        correction_tree_materialization is not None
        and int(correction_tree_materialization.get("promotion", 0)) == 1
    ):
        progress("materialization-aware backend competition starting")
        source_panel_count = len(per_panel_patches)
        parent_serialization_failure_fields = _serialization_failure_fields_for_panels(
            vertices=vertices,
            panels=per_panel_patches,
            uv_by_panel=per_panel_uv,
            selected_backends=selected_backend_per_panel,
            seam_edges=seam_edges,
            distortion_threshold=distortion_threshold,
        )
        (
            vertices,
            chart_panels,
            chart_parent_indices,
            chart_materialization_kinds,
            chart_variant_ids,
        ) = _materialized_chart_panels(
            vertices=vertices,
            panels=per_panel_patches,
            materialization_receipt=correction_tree_materialization,
            failure_fields=parent_serialization_failure_fields,
        )
        original_uv = dict(per_panel_uv)
        original_distortions = list(per_panel_distortion)
        original_candidates = list(per_panel_candidate_receipts)
        original_selected_backends = list(selected_backend_per_panel)
        original_grain = list(per_panel_grain)
        original_patches = list(per_panel_patches)
        next_distortions: list[float] = []
        next_uv: dict[str, np.ndarray] = {}
        next_grain: list[str] = []
        next_candidates: list[list[object]] = []
        next_selected_backends: list[str] = []
        next_patches: list[PanelPatch] = []
        accepted_parent_indices: list[int] = []
        chart_domain_decisions: list[dict[str, object]] = []
        operator_basis_search_receipts: list[dict[str, object]] = []
        for parent_idx, original_panel in enumerate(original_patches):
            parent_variant_ids = sorted(
                {
                    variant_id
                    for source_idx, variant_id in zip(
                        chart_parent_indices,
                        chart_variant_ids,
                        strict=False,
                    )
                    if source_idx == parent_idx
                }
            )
            original_score = _selected_candidate_score(
                original_candidates[parent_idx],
                original_selected_backends[parent_idx],
            )
            original_distortion = _selected_candidate_distortion(
                original_candidates[parent_idx],
                original_selected_backends[parent_idx],
            )
            original_metrics = {
                "score": float(original_score),
                "worst_distortion": float(original_distortion),
                "foldovers": int(
                    _selected_candidate_metric(
                        original_candidates[parent_idx],
                        original_selected_backends[parent_idx],
                        "foldovers",
                    )
                ),
                "boundary_deviation": float(
                    _selected_candidate_metric(
                        original_candidates[parent_idx],
                        original_selected_backends[parent_idx],
                        "boundary_deviation",
                    )
                ),
                "chart_count": 1,
                "exportable": bool(
                    _selected_candidate_metric(
                        original_candidates[parent_idx],
                        original_selected_backends[parent_idx],
                        "exportable",
                        default=False,
                    )
                ),
            }
            variant_results: list[dict[str, object]] = []
            for variant_id in parent_variant_ids:
                split_entries = [
                    (idx, chart_panels[idx], chart_materialization_kinds[idx])
                    for idx, (source_idx, source_variant_id) in enumerate(
                        zip(chart_parent_indices, chart_variant_ids, strict=False)
                    )
                    if source_idx == parent_idx and source_variant_id == variant_id
                ]
                group_results: list[
                    tuple[PanelPatch, np.ndarray, float, list[object], str, str]
                ] = []
                for split_position, (split_idx, panel, materialization_kind) in enumerate(
                    split_entries
                ):
                    constraints = _panel_materialization_constraints(
                        correction_tree_materialization,
                        parent_idx,
                        chart_panel_id=f"P{len(next_distortions) + split_position}",
                    )
                    if constraints is not None:
                        constraints["source_panel_id"] = f"P{parent_idx}"
                        constraints["chart_materialization_kind"] = materialization_kind
                        constraints["chart_surgery_variant"] = variant_id
                    uv, distortion, candidates, selected_backend = compete_panel(
                        panel,
                        panel_label=f"P{parent_idx}.{variant_id}.{split_position}",
                        depth=0,
                        correction_tree_for_competition=correction_tree,
                        materialization_constraints=constraints,
                    )
                    group_results.append(
                        (
                            panel,
                            uv,
                            float(distortion),
                            candidates,
                            selected_backend,
                            materialization_kind,
                        )
                    )
                split_score = sum(
                    _selected_candidate_score(candidates, selected_backend)
                    for _panel, _uv, _distortion, candidates, selected_backend, _kind in group_results
                )
                split_worst_distortion = max(
                    (
                        _selected_candidate_distortion(candidates, selected_backend)
                        for _panel, _uv, _distortion, candidates, selected_backend, _kind in group_results
                    ),
                    default=float("inf"),
                )
                split_valid = _chart_group_backend_serializable(
                    vertices,
                    [
                        panel
                        for panel, _uv, _distortion, _candidates, _backend, _kind in group_results
                    ],
                )
                variant_metrics = _variant_metrics_from_group_results(group_results)
                split_requested = any(
                    kind != "original_chart"
                    for _panel, _uv, _distortion, _candidates, _backend, kind in group_results
                )
                acceptable = bool(
                    split_requested
                    and split_valid
                    and split_score < original_score
                    and split_worst_distortion <= original_distortion + 1e-12
                )
                variant_results.append(
                    {
                        "variant_id": variant_id,
                        "acceptable": acceptable,
                        "split_requested": split_requested,
                        "split_valid": split_valid,
                        "split_score": split_score,
                        "split_worst_distortion": split_worst_distortion,
                        "split_chart_count": len(group_results),
                        "group_results": group_results,
                        "candidate_id": f"P{parent_idx}.{variant_id}",
                        "metrics": variant_metrics,
                        "metric_comparison": _metric_comparison_receipt(
                            original_metrics,
                            variant_metrics,
                        ),
                    }
                )
            pareto_receipt = _variant_pareto_receipt(
                original_candidate_id=f"P{parent_idx}.original",
                original_metrics=original_metrics,
                variant_candidates=[
                    {
                        "candidate_id": str(result["candidate_id"]),
                        "variant_id": str(result["variant_id"]),
                        "metrics": result["metrics"],
                    }
                    for result in variant_results
                ],
            )
            operator_basis_search_receipt = _operator_basis_search_receipt(
                panel_id=f"P{parent_idx}",
                original_metrics=original_metrics,
                variant_candidates=[
                    {
                        "candidate_id": str(result["candidate_id"]),
                        "variant_id": str(result["variant_id"]),
                        "metrics": result["metrics"],
                    }
                    for result in variant_results
                ],
            )
            operator_basis_search_receipts.append(operator_basis_search_receipt)
            frontier_candidate_ids = set(
                _json_list(pareto_receipt.get("frontier_candidate_ids", []))
            )
            profile_best_candidate_ids = cast(
                Mapping[str, object],
                pareto_receipt.get("profile_best_candidate_ids", {}),
            )
            pareto_candidates = cast(Sequence[Mapping[str, object]], pareto_receipt["candidates"])
            variant_profile_scores = {
                str(candidate["candidate_id"]): cast(
                    Mapping[str, object], candidate["profile_scores"]
                )
                for candidate in pareto_candidates
            }
            variant_default_profile_selection = {
                str(candidate["candidate_id"]): cast(
                    Mapping[str, object], candidate["default_profile_selection"]
                )
                for candidate in pareto_candidates
            }
            for result in variant_results:
                candidate_id = str(result["candidate_id"])
                result["pareto_frontier_member"] = candidate_id in frontier_candidate_ids
                result["pareto_useful"] = bool(result["metric_comparison"]["pareto_useful"])
                result["pareto_status"] = (
                    "frontier" if result["pareto_frontier_member"] else "dominated"
                )
                result["profile_scores"] = {
                    profile: float(score)
                    for profile, score in variant_profile_scores.get(candidate_id, {}).items()
                }
                result["selected_by_default_profile"] = candidate_id == str(
                    profile_best_candidate_ids.get("default")
                )
                result["default_profile_selection"] = dict(
                    variant_default_profile_selection.get(candidate_id, {})
                )
            acceptable_variants = [
                result for result in variant_results if bool(result["acceptable"])
            ]
            accepted_variant = (
                min(acceptable_variants, key=lambda result: float(result["split_score"]))
                if acceptable_variants
                else None
            )
            accept_split = accepted_variant is not None
            chart_domain_decisions.append(
                {
                    "source_panel_id": f"P{parent_idx}",
                    "accepted_variant": None
                    if accepted_variant is None
                    else str(accepted_variant["variant_id"]),
                    "split_requested": any(
                        bool(result["split_requested"]) for result in variant_results
                    ),
                    "accepted": accept_split,
                    "original_score": original_score,
                    "original_metrics": original_metrics,
                    "split_score": None
                    if accepted_variant is None
                    else float(accepted_variant["split_score"]),
                    "original_distortion": original_distortion,
                    "split_worst_distortion": None
                    if accepted_variant is None
                    else float(accepted_variant["split_worst_distortion"]),
                    "split_chart_count": None
                    if accepted_variant is None
                    else int(accepted_variant["split_chart_count"]),
                    "split_backend_serializable": None
                    if accepted_variant is None
                    else bool(accepted_variant["split_valid"]),
                    "pareto_receipt": pareto_receipt,
                    "operator_basis_search_receipt": operator_basis_search_receipt,
                    "profile_best_candidate_ids": profile_best_candidate_ids,
                    "variants": [
                        {
                            "variant_id": str(result["variant_id"]),
                            "acceptable": bool(result["acceptable"]),
                            "split_requested": bool(result["split_requested"]),
                            "split_valid": bool(result["split_valid"]),
                            "split_score": float(result["split_score"]),
                            "split_worst_distortion": float(result["split_worst_distortion"]),
                            "split_chart_count": int(result["split_chart_count"]),
                            "candidate_id": str(result["candidate_id"]),
                            "metrics": result["metrics"],
                            "metric_comparison": result["metric_comparison"],
                            "pareto_frontier_member": bool(result["pareto_frontier_member"]),
                            "pareto_useful": bool(result["pareto_useful"]),
                            "pareto_status": str(result["pareto_status"]),
                            "profile_scores": result["profile_scores"],
                            "selected_by_default_profile": bool(
                                result["selected_by_default_profile"]
                            ),
                            "default_profile_selection": result["default_profile_selection"],
                        }
                        for result in variant_results
                    ],
                }
            )
            if accept_split:
                accepted_parent_indices.append(parent_idx)
                accepted_group_results = cast(
                    list[tuple[PanelPatch, np.ndarray, float, list[object], str, str]],
                    accepted_variant["group_results"],
                )
                for (
                    panel,
                    uv,
                    distortion,
                    candidates,
                    selected_backend,
                    _kind,
                ) in accepted_group_results:
                    chart_idx = len(next_distortions)
                    next_uv[f"panel_{chart_idx}"] = uv
                    next_distortions.append(float(distortion))
                    next_candidates.append(candidates)
                    next_selected_backends.append(selected_backend)
                    next_grain.append(_infer_grain_direction(vertices, panel, uv))
                    next_patches.append(panel)
            else:
                chart_idx = len(next_distortions)
                next_uv[f"panel_{chart_idx}"] = original_uv[f"panel_{parent_idx}"]
                next_distortions.append(float(original_distortions[parent_idx]))
                next_candidates.append(original_candidates[parent_idx])
                next_selected_backends.append(original_selected_backends[parent_idx])
                next_grain.append(original_grain[parent_idx])
                next_patches.append(original_panel)
        per_panel_uv = next_uv
        per_panel_distortion = next_distortions
        per_panel_candidate_receipts = next_candidates
        selected_backend_per_panel = next_selected_backends
        per_panel_grain = next_grain
        per_panel_patches = next_patches
        corrected_residuals = _expanded_corrected_residuals(
            corrected_residuals,
            chart_parent_indices,
            per_panel_distortion,
        )
        if correction_tree_materialization is not None:
            corrected_residuals = None
        panels_all_disks = panels_all_disks and all(
            bool(panel_chart_diagnostics(vertices, panel)["backend_serializable"])
            for panel in per_panel_patches
        )
        if fabric_metric is not None:
            fabric_metric["materialized_chart_domains"] = {
                "source_panel_count": source_panel_count,
                "chart_panel_count": len(per_panel_patches),
                "accepted_parent_indices": accepted_parent_indices,
                "parent_serialization_failure_fields": parent_serialization_failure_fields,
                "operator_basis_search_receipts": operator_basis_search_receipts,
                "chart_domain_decisions": chart_domain_decisions,
            }
        worst = float(max(per_panel_distortion))
        mean = float(sum(per_panel_distortion) / len(per_panel_distortion))
        distortion_margin = float(distortion_threshold - worst)
        if distortion_margin >= 0.0:
            panel_unwrap_blockers = [
                blocker
                for blocker in panel_unwrap_blockers
                if blocker != "distortion_exceeds_threshold"
            ]
        elif "distortion_exceeds_threshold" not in panel_unwrap_blockers:
            panel_unwrap_blockers.append("distortion_exceeds_threshold")
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(uv_path, **per_panel_uv)  # type: ignore[arg-type]
        uv_hash = _sha256_file(uv_path)
    correction_tree_hash = _sha256_json(correction_tree)
    serialization_failure_fields = _serialization_failure_fields_for_panels(
        vertices=vertices,
        panels=per_panel_patches,
        uv_by_panel=per_panel_uv,
        selected_backends=selected_backend_per_panel,
        seam_edges=seam_edges,
        distortion_threshold=distortion_threshold,
    )
    panel_serialization_competitions = [
        build_panel_serialization_competition_receipt(
            panel_id=f"P{idx}",
            correction_tree_hash=correction_tree_hash,
            correction_tree=correction_tree,
            candidates=cast(Any, candidates),
            selected_backend=selected_backend_per_panel[idx],
        )
        for idx, candidates in enumerate(per_panel_candidate_receipts)
    ]
    serialization_promoted = all(
        int(competition["promotion"]) == 1 for competition in panel_serialization_competitions
    )
    serialization_competition_receipt = {
        "schema_version": "smii.panel_serialization_competition.v1",
        "claim_boundary": "serialization_is_not_morphology_authority",
        "correction_tree_hash": correction_tree_hash,
        "panel_count": len(panel_serialization_competitions),
        "panels": panel_serialization_competitions,
        "failure_fields": serialization_failure_fields,
        "selected_backend_per_panel": list(selected_backend_per_panel),
        "promotion": 1 if serialization_promoted else 0,
        "blockers": (
            []
            if serialization_promoted
            else list(
                dict.fromkeys(
                    str(blocker)
                    for competition in panel_serialization_competitions
                    for blocker in _json_list(competition.get("blockers"))
                )
            )
        ),
    }
    if not serialization_promoted:
        panel_unwrap_blockers.extend(
            str(blocker) for blocker in _json_list(serialization_competition_receipt["blockers"])
        )
        if "backend_skipped_invalid_chart_domain" in _json_list(
            serialization_competition_receipt["blockers"]
        ):
            panel_unwrap_blockers.append("chart_domain_not_backend_serializable")
        if (
            correction_tree_materialization is not None
            and int(correction_tree_materialization.get("promotion", 0)) == 1
        ):
            panel_unwrap_blockers.append("operator_materialized_but_serialization_failed")
        panel_unwrap_blockers.append("panel_serialization_blocked")
        panel_unwrap_blockers = list(dict.fromkeys(panel_unwrap_blockers))
    promotion = 1 if not panel_unwrap_blockers else 0
    selected_backends = set(selected_backend_per_panel)
    receipt_backend = next(iter(selected_backends)) if len(selected_backends) == 1 else None
    receipt = PanelUnwrapReceipt(
        solver_receipt_hash=_sha256_file(solver_receipt_path),
        panel_count=len(per_panel_distortion),
        panels_all_disks=panels_all_disks,
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
        cut_topology_receipt_hash=cut_topology_receipt_hash,
        unwrap_backend=receipt_backend,
        backend_is_bootstrap=(
            None if receipt_backend is None else receipt_backend == BOOTSTRAP_BACKEND
        ),
        distortion_margin=distortion_margin,
        panel_unwrap_blockers=panel_unwrap_blockers,
        per_panel_corrected_residual=corrected_residuals,
        worst_corrected_residual=worst_corrected,
        mean_corrected_residual=mean_corrected,
        correction_payload_hash=correction_payload_hash,
        metric_correction_receipt_hash=metric_correction_receipt_hash,
        fabric_metric_receipt=fabric_metric,
        correction_tree_receipt=correction_tree,
        correction_operator_scoring_receipt=correction_operator_scoring,
        realized_correction_operator_receipt=realized_correction_operator,
        correction_tree_materialization_receipt=correction_tree_materialization,
        serialization_competition_receipt=serialization_competition_receipt,
        selected_backend_per_panel=selected_backend_per_panel,
        serialization_promoted=serialization_promoted,
    )
    target_receipt_path = receipt_path or (output_dir / "panel_unwrap_receipt.json")
    receipt.to_json(target_receipt_path)
    print(f"Wrote panel UVs to {uv_path}")
    print(f"Wrote panel unwrap receipt to {target_receipt_path}")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-receipt", type=Path, required=True)
    parser.add_argument("--seam-edges", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True, help="Mesh NPZ with vertices/faces.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--cut-topology-receipt",
        type=Path,
        default=None,
        help="Promoted cut topology receipt required by new P3 chains.",
    )
    parser.add_argument(
        "--out-panel-receipt",
        type=Path,
        default=None,
        help="Output panel receipt path (default: <out-dir>/panel_unwrap_receipt.json).",
    )
    parser.add_argument("--distortion-threshold", type=float, default=0.05)
    parser.add_argument("--max-subdivisions", type=int, default=3)
    parser.add_argument(
        "--solver",
        choices=list(UNWRAP_BACKENDS),
        default=BOOTSTRAP_BACKEND,
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        default=None,
        help="Optional metric corrections JSON emitted by metric_panelization.",
    )
    parser.add_argument(
        "--metric-correction-receipt",
        type=Path,
        default=None,
        help="Promoted MetricCorrectionReceipt required for typed correction operators.",
    )
    parser.add_argument(
        "--fabric-profile",
        type=Path,
        default=None,
        help="Optional fabric YAML/JSON used for fabric-relative panel metric gating.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    unwrap_panels(
        solver_receipt_path=args.solver_receipt,
        seam_edges_path=args.seam_edges,
        mesh_path=args.mesh,
        output_dir=args.out_dir,
        cut_topology_receipt_path=args.cut_topology_receipt,
        distortion_threshold=args.distortion_threshold,
        max_subdivisions=args.max_subdivisions,
        solver=args.solver,
        corrections_path=args.corrections,
        metric_correction_receipt_path=args.metric_correction_receipt,
        fabric_profile_path=args.fabric_profile,
        receipt_path=args.out_panel_receipt,
    )


if __name__ == "__main__":
    main()
