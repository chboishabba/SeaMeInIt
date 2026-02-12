"""Deterministic seam solver built on ROM-derived costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping

import numpy as np

from smii.rom.constraints import ConstraintRegistry
from smii.rom.seam_costs import SeamCostField

from .edge_costs import EdgeAggregationMode, EdgeCostResult, build_edge_costs
from .kernels import EdgeKernel, KernelWeights, edge_energy

try:  # Optional import to avoid heavy deps at module import time
    from suit.seam_generator import SeamGraph
except Exception:  # pragma: no cover
    SeamGraph = None  # type: ignore[assignment]


SolverMode = str


@dataclass(frozen=True, slots=True)
class PanelSolution:
    """Selected seams for a single panel."""

    panel: str
    edges: tuple[tuple[int, int], ...]
    vertices: tuple[int, ...]
    cost: float
    cost_breakdown: Mapping[str, float]
    warnings: tuple[str, ...]
    fabric_id: str | None = None
    grain_rotation_deg: float = 0.0


@dataclass(frozen=True, slots=True)
class SeamSolution:
    """Seam selections across all panels."""

    solver: SolverMode
    panel_solutions: Mapping[str, PanelSolution]
    total_cost: float
    warnings: tuple[str, ...]
    metadata: Mapping[str, float]


def _component_roots(vertices: Iterable[int]) -> dict[int, int]:
    return {vertex: vertex for vertex in vertices}


def _find(parent: dict[int, int], item: int) -> int:
    root = parent[item]
    while parent[root] != root:
        root = parent[root]
    # Path compression
    while parent[item] != root:
        item, parent[item] = parent[item], root
    return root


def _union(parent: dict[int, int], a: int, b: int) -> bool:
    root_a = _find(parent, a)
    root_b = _find(parent, b)
    if root_a == root_b:
        return False
    parent[root_b] = root_a
    return True


def _symmetry_penalty(
    edges: Iterable[tuple[int, int]],
    constraint_registry: ConstraintRegistry | None,
    penalty_weight: float,
) -> float:
    if constraint_registry is None or penalty_weight <= 0.0:
        return 0.0

    penalty = 0.0
    edge_set = {tuple(sorted(edge)) for edge in edges}
    for edge in edge_set:
        partner = tuple(
            sorted(
                (
                    constraint_registry.symmetry_partner(edge[0]) or edge[0],
                    constraint_registry.symmetry_partner(edge[1]) or edge[1],
                )
            )
        )
        if partner not in edge_set:
            penalty += penalty_weight
    return penalty


def solve_seams(
    seam_graph: "SeamGraph",
    cost_field: SeamCostField,
    *,
    constraints: ConstraintRegistry | None = None,
    kernels: Mapping[tuple[int, int], EdgeKernel] | None = None,
    kernel_weights: KernelWeights | None = None,
    edge_costs: EdgeCostResult | None = None,
    edge_mode: EdgeAggregationMode = "mean",
    vertices: np.ndarray | None = None,
    solver: SolverMode = "mst",
    vertex_weight: float = 1.0,
    edge_weight: float = 1.0,
    symmetry_penalty: float = 0.0,
    forbidden_policy: str = "error",
) -> SeamSolution:
    """Compute deterministic seam selections per panel.

    This baseline solver builds a minimal-cost spanning structure across each panel's
    seam vertices, enforcing hard forbidden-vertex constraints and optional symmetry penalties.
    When ``kernels`` are provided, edge costs are derived from ``edge_energy`` (including
    fabric terms) instead of ROM vertex aggregation.
    """

    if SeamGraph is None:
        raise ImportError("suit.seam_generator.SeamGraph is required to solve seams.")
    if solver not in ("mst", "minimum_spanning"):
        raise ValueError(f"Solver '{solver}' is not implemented. Use 'mst'.")

    edge_mode_used: str = str(edge_mode)
    if kernels is not None:
        weights = kernel_weights or KernelWeights()
        edge_lookup = {
            (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1]))): edge_energy(kernel, weights)
            for edge, kernel in kernels.items()
        }
        edge_mode_used = "kernel"
    else:
        edge_costs = edge_costs or build_edge_costs(
            cost_field,
            seam_graph,
            vertices=vertices,
            mode=edge_mode,
        )
        edge_lookup = edge_costs.as_mapping()

    panel_solutions: MutableMapping[str, PanelSolution] = {}
    warnings_accum: list[str] = []

    for panel in seam_graph.panels:
        panel_vertices = tuple(int(idx) for idx in panel.seam_vertices)
        panel_vertex_set = set(panel_vertices)
        panel_warnings: list[str] = []

        if not panel_vertices:
            panel_solutions[panel.name] = PanelSolution(
                panel=panel.name,
                edges=tuple(),
                vertices=panel_vertices,
                cost=0.0,
                cost_breakdown={"edge_cost": 0.0, "vertex_cost": 0.0, "symmetry_penalty": 0.0},
                warnings=("panel has no seam vertices",),
            )
            continue

        if constraints:
            forbidden = [v for v in panel_vertices if constraints.is_vertex_forbidden(v)]
            if forbidden:
                message = f"Panel '{panel.name}' contains forbidden vertices: {sorted(forbidden)}"
                if forbidden_policy == "error":
                    raise ValueError(message)
                panel_warnings.append(message)
                panel_vertex_set = {v for v in panel_vertex_set if v not in forbidden}

        candidate_edges = [
            (edge, cost)
            for edge, cost in edge_lookup.items()
            if edge[0] in panel_vertex_set and edge[1] in panel_vertex_set
        ]
        if not candidate_edges:
            panel_warnings.append("no candidate edges intersect seam vertices")
            panel_solutions[panel.name] = PanelSolution(
                panel=panel.name,
                edges=tuple(),
                vertices=tuple(panel_vertex_set),
                cost=0.0,
                cost_breakdown={"edge_cost": 0.0, "vertex_cost": 0.0, "symmetry_penalty": 0.0},
                warnings=tuple(panel_warnings),
            )
            warnings_accum.extend(panel_warnings)
            continue

        # Minimum spanning selection for connectivity
        parent = _component_roots(panel_vertex_set)
        selected_edges: list[tuple[int, int]] = []
        edge_cost_sum = 0.0
        for edge, cost in sorted(candidate_edges, key=lambda item: (item[1], item[0])):
            if _union(parent, edge[0], edge[1]):
                selected_edges.append(edge)
                edge_cost_sum += float(cost) * edge_weight

        components = {_find(parent, vertex) for vertex in panel_vertex_set}
        if len(components) > 1:
            panel_warnings.append(f"panel remains disconnected after solve (components={len(components)})")

        vertex_cost_term = float(np.nansum(np.asarray(cost_field.vertex_costs, dtype=float)[list(panel_vertex_set)]))
        vertex_cost_term *= vertex_weight

        symmetry_term = _symmetry_penalty(selected_edges, constraints, symmetry_penalty)

        total_cost = edge_cost_sum + vertex_cost_term + symmetry_term
        breakdown = {
            "edge_cost": edge_cost_sum,
            "vertex_cost": vertex_cost_term,
            "symmetry_penalty": symmetry_term,
        }

        panel_solution = PanelSolution(
            panel=panel.name,
            edges=tuple(sorted(selected_edges)),
            vertices=tuple(sorted(panel_vertex_set)),
            cost=total_cost,
            cost_breakdown=breakdown,
            warnings=tuple(panel_warnings),
        )
        panel_solutions[panel.name] = panel_solution
        warnings_accum.extend(panel_warnings)

    total = sum(solution.cost for solution in panel_solutions.values())
    metadata = {
        "edge_mode": edge_mode_used,
        "vertex_weight": float(vertex_weight),
        "edge_weight": float(edge_weight),
        "symmetry_penalty": float(symmetry_penalty),
        "solver": solver,
    }
    return SeamSolution(
        solver=solver,
        panel_solutions=panel_solutions,
        total_cost=total,
        warnings=tuple(warnings_accum),
        metadata=metadata,
    )
