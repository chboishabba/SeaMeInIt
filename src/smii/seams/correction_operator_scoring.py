"""Deterministic pricing for seam-branch correction operators.

This is deliberately a receipt-level estimator, not a cloth simulator.  It
prices unresolved correction-tree branch nodes so Gate 6 can distinguish
"untyped morphology" from "priced operators that still do not absorb enough
metric/fabric residual."
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence, cast

import numpy as np

from .fabric_kernels import FabricProfile

Edge = tuple[int, int]

OPERATOR_FAMILIES = (
    "stretch_zone",
    "dart_apex",
    "gusset_corner",
    "ease_convergence",
    "grain_rotation",
    "seam_junction",
    "diagnostic_carry",
)

__all__ = [
    "OPERATOR_FAMILIES",
    "price_correction_operator_tree",
]


def _as_float(value: object) -> float:
    return float(cast(Any, value))


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


def _branch_vertices(edges: Sequence[Edge]) -> list[tuple[int, set[int]]]:
    graph = _adjacency(edges)
    return [
        (vertex, neighbors) for vertex, neighbors in sorted(graph.items()) if len(neighbors) > 2
    ]


def _fabric_compliance(profile: FabricProfile | None) -> tuple[float, float, float, float]:
    if profile is None:
        return 1.0, 1.0, 1.0, 1.0
    comp = profile.compliance
    values = (
        max(0.0, float(comp.s_parallel)),
        max(0.0, float(comp.s_perp)),
        max(0.0, float(comp.s_shear)),
    )
    max_value = max(values + (1e-6,))
    return values[0], values[1], values[2], max_value


def _branch_axis_coherence(vertices: np.ndarray, vertex: int, neighbors: set[int]) -> float:
    if len(neighbors) < 2:
        return 0.0
    if vertex < 0 or vertex >= len(vertices):
        return 0.0
    center = np.asarray(vertices[vertex], dtype=float)
    vectors: list[np.ndarray] = []
    for neighbor in sorted(neighbors):
        if neighbor < 0 or neighbor >= len(vertices):
            continue
        vector = np.asarray(vertices[neighbor], dtype=float) - center
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            vectors.append(vector / norm)
    if len(vectors) < 2:
        return 0.0
    matrix = np.asarray(vectors, dtype=float)
    try:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0
    total = float(np.sum(singular_values))
    if total <= 1e-12:
        return 0.0
    return float(np.clip(float(singular_values[0]) / total, 0.0, 1.0))


def _candidate_estimate(
    *,
    operator: str,
    residual_before: float,
    fabric_violation_before: float,
    coherence: float,
    incident_degree: int,
    profile: FabricProfile | None,
) -> dict[str, object]:
    s_parallel, s_perp, s_shear, max_compliance = _fabric_compliance(profile)
    stretch_strength = float(
        np.clip((0.45 * s_parallel + 0.35 * s_perp + 0.20 * s_shear) / max_compliance, 0.0, 1.0)
    )
    low_stretch = 1.0 - min(max_compliance, 1.0)
    branch_factor = float(np.clip((incident_degree - 2) / 3.0, 0.0, 1.0))
    bias_allowed = bool(profile.constraints.allow_bias) if profile is not None else True
    modifiers: Mapping[str, float] = profile.mdl_modifiers if profile is not None else {}
    seam_modifier = float(modifiers.get("seam_count", 1.0))
    panel_modifier = float(modifiers.get("panel_count", 1.0))

    fabrication_allowed = True
    if operator == "stretch_zone":
        absorption = 0.24 + 0.48 * stretch_strength
        fabric_absorption = 0.38 + 0.45 * stretch_strength
        sewing_complexity = 0.018 + 0.012 * (1.0 - stretch_strength)
        manufacturing_cost = 0.020 if max_compliance < 0.4 else 0.006
        style_cost = 0.010
    elif operator == "dart_apex":
        absorption = 0.28 + 0.36 * coherence + 0.12 * low_stretch
        fabric_absorption = 0.24 + 0.24 * coherence
        sewing_complexity = 0.052 * seam_modifier
        manufacturing_cost = 0.020
        style_cost = 0.022
    elif operator == "gusset_corner":
        absorption = 0.32 + 0.30 * branch_factor + 0.18 * stretch_strength
        fabric_absorption = 0.30 + 0.28 * branch_factor
        sewing_complexity = 0.074 * panel_modifier
        manufacturing_cost = 0.044
        style_cost = 0.020
    elif operator == "ease_convergence":
        absorption = 0.20 + 0.28 * stretch_strength + 0.14 * (1.0 - coherence)
        fabric_absorption = 0.22 + 0.25 * stretch_strength
        sewing_complexity = 0.034 * seam_modifier
        manufacturing_cost = 0.016
        style_cost = 0.030
    elif operator == "grain_rotation":
        fabrication_allowed = bias_allowed
        absorption = 0.18 + 0.26 * coherence
        fabric_absorption = 0.32 + 0.40 * coherence
        sewing_complexity = 0.020
        manufacturing_cost = 0.018
        style_cost = 0.030
    elif operator == "seam_junction":
        absorption = 0.18 + 0.22 * branch_factor
        fabric_absorption = 0.18 + 0.16 * branch_factor
        sewing_complexity = 0.090 * seam_modifier
        manufacturing_cost = 0.062
        style_cost = 0.034
    else:
        absorption = 0.0
        fabric_absorption = 0.0
        sewing_complexity = 0.0
        manufacturing_cost = 0.0
        style_cost = 0.0

    if not fabrication_allowed:
        absorption = 0.0
        fabric_absorption = 0.0

    estimated_residual_after = residual_before * (1.0 - float(np.clip(absorption, 0.0, 0.95)))
    estimated_fabric_violation_after = fabric_violation_before * (
        1.0 - float(np.clip(fabric_absorption, 0.0, 0.95))
    )
    pressure_shear_after = 0.18 * estimated_fabric_violation_after + 0.04 * branch_factor
    score = (
        1.0 * estimated_residual_after
        + 1.0 * estimated_fabric_violation_after
        + 0.45 * pressure_shear_after
        + 1.0 * sewing_complexity
        + 0.7 * manufacturing_cost
        + 0.35 * style_cost
    )
    return {
        "operator": operator,
        "fabrication_allowed": fabrication_allowed,
        "estimated_residual_after": float(estimated_residual_after),
        "estimated_fabric_violation_after": float(estimated_fabric_violation_after),
        "estimated_pressure_shear_after": float(pressure_shear_after),
        "sewing_complexity": float(sewing_complexity),
        "manufacturing_cost": float(manufacturing_cost),
        "style_cost": float(style_cost),
        "score": float(score),
        "promotion": "fallback" if operator == "diagnostic_carry" else "candidate",
    }


def price_correction_operator_tree(
    *,
    seam_edges: Sequence[tuple[int, int]],
    vertices: np.ndarray,
    fabric_profile: FabricProfile | None,
    residual_before: float,
    fabric_violation_before: float,
    typed_operator_count: int = 0,
    margin: float = 1e-6,
) -> dict[str, object]:
    """Price correction operators for each branch node in a seam graph."""

    normalized_edges = tuple(_normalize_edge(a, b) for a, b in seam_edges)
    branch_nodes = _branch_vertices(normalized_edges)
    priced_nodes: list[dict[str, object]] = []
    typed_count = 0
    diagnostic_count = 0
    residual_afters: list[float] = []
    fabric_afters: list[float] = []

    for branch_idx, (vertex, neighbors) in enumerate(branch_nodes):
        coherence = _branch_axis_coherence(vertices, vertex, neighbors)
        residual_before_node = float(residual_before)
        fabric_before_node = float(fabric_violation_before)
        candidates = [
            _candidate_estimate(
                operator=operator,
                residual_before=residual_before_node,
                fabric_violation_before=fabric_before_node,
                coherence=coherence,
                incident_degree=len(neighbors),
                profile=fabric_profile,
            )
            for operator in OPERATOR_FAMILIES
        ]
        diagnostic = next(
            candidate for candidate in candidates if candidate["operator"] == "diagnostic_carry"
        )
        admissible = [
            candidate
            for candidate in candidates
            if candidate["fabrication_allowed"] and candidate["operator"] != "diagnostic_carry"
        ]
        best = (
            min(admissible, key=lambda candidate: _as_float(candidate["score"]))
            if admissible
            else diagnostic
        )
        promoted = branch_idx < int(typed_operator_count) or _as_float(best["score"]) + float(
            margin
        ) < _as_float(diagnostic["score"])
        selected = best if promoted else diagnostic
        if promoted:
            typed_count += 1
        else:
            diagnostic_count += 1
        residual_afters.append(_as_float(selected["estimated_residual_after"]))
        fabric_afters.append(_as_float(selected["estimated_fabric_violation_after"]))
        priced_nodes.append(
            {
                "branch_id": f"branch_{branch_idx:03d}",
                "branch_vertex": int(vertex),
                "incident_degree": int(len(neighbors)),
                "residual_signature": {
                    "local_distortion": residual_before_node,
                    "metric_corrected_residual": residual_before_node,
                    "fabric_violation": fabric_before_node,
                    "principal_axis_coherence": coherence,
                    "rom_pressure_p95": 0.0,
                    "rom_shear_p95": 0.0,
                },
                "candidates": candidates,
                "selected_operator": str(selected["operator"]),
                "selection_reason": "lowest_total_cost_under_declared_fabric"
                if promoted
                else "diagnostic_carry_not_beaten",
                "promoted": bool(promoted),
                "blockers": [] if promoted else ["unpriced_correction_tree_node"],
            }
        )

    worst_residual_after = max(residual_afters) if residual_afters else float(residual_before)
    worst_fabric_after = max(fabric_afters) if fabric_afters else float(fabric_violation_before)
    blocker = diagnostic_count > 0
    return {
        "schema_version": "smii.correction_operator_scoring.v1",
        "claim_boundary": "operators_are_deterministic_estimators_not_cloth_simulation",
        "fabric_profile": fabric_profile.fabric_id if fabric_profile is not None else None,
        "branch_count": len(branch_nodes),
        "typed_branch_count": typed_count,
        "diagnostic_branch_count": diagnostic_count,
        "residual_before": float(residual_before),
        "fabric_violation_before": float(fabric_violation_before),
        "estimated_worst_residual_after": float(worst_residual_after),
        "estimated_worst_fabric_violation_after": float(worst_fabric_after),
        "nodes": priced_nodes,
        "promotion": 0 if blocker else 1,
        "blockers": ["unpriced_correction_tree_node"] if blocker else [],
    }
