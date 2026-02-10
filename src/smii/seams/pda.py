"""PDA-style seam optimization controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping

import numpy as np

from smii.rom.constraints import ConstraintRegistry
from smii.rom.seam_costs import SeamCostField
from smii.seams.fabric_kernels import FabricAssignment, FabricProfile, fabric_penalty, rotate_grain
from smii.seams.kernels import EdgeKernel, KernelWeights
from smii.seams.mdl import MDLPrior, mdl_cost
from smii.seams.moves import merge_panels, reroute_edge, shorten_seam, split_panel
from smii.seams.solver import PanelSolution, SeamSolution

try:  # Optional heavy import
    from suit.seam_generator import SeamGraph
except Exception:  # pragma: no cover
    SeamGraph = None  # type: ignore[assignment]

__all__ = ["solve_seams_pda"]


SolverMode = str


def _component_roots(vertices: Iterable[int]) -> dict[int, int]:
    return {vertex: vertex for vertex in vertices}


def _find(parent: dict[int, int], item: int) -> int:
    root = parent[item]
    while parent[root] != root:
        root = parent[root]
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


def _apply_rotation_limit(rotation_deg: float, profile: FabricProfile | None) -> float:
    if profile is None or profile.constraints.max_grain_rotation_deg is None:
        return rotation_deg
    return float(np.clip(rotation_deg, -profile.constraints.max_grain_rotation_deg, profile.constraints.max_grain_rotation_deg))


def _fabric_profile_for(fabric_id: str | None, catalog: Mapping[str, FabricProfile] | None) -> FabricProfile | None:
    if fabric_id is None or catalog is None:
        return None
    return catalog.get(fabric_id)


def _edge_cost_with_fabric(
    edge: tuple[int, int],
    kernel: EdgeKernel,
    weights: KernelWeights,
    *,
    fabric_profile: FabricProfile | None,
    fabric_grain: np.ndarray | None,
    grain_rotation_deg: float,
    vertices: np.ndarray | None,
) -> tuple[float, float]:
    base_cost = (
        weights.rom_mean * kernel.rom_mean
        + weights.rom_max * kernel.rom_max
        + weights.rom_grad * kernel.rom_grad
        + weights.curvature * kernel.curvature
    )
    fabric_cost = weights.fabric * kernel.fabric_misalignment
    if fabric_profile is not None and fabric_grain is not None and vertices is not None:
        rotated_grain = rotate_grain(fabric_grain, _apply_rotation_limit(grain_rotation_deg, fabric_profile))
        a, b = edge
        if max(a, b) < len(vertices):
            edge_vec = np.asarray(vertices[b] - vertices[a], dtype=float)
            fabric_cost = weights.fabric * fabric_penalty(edge_vec, rotated_grain, fabric_profile)
    return base_cost + fabric_cost, fabric_cost


def _edge_cost_maps(
    panel_edges: Iterable[tuple[int, int]],
    kernels: Mapping[tuple[int, int], EdgeKernel],
    weights: KernelWeights,
    vertices: np.ndarray,
    *,
    fabric_profile: FabricProfile | None,
    fabric_grain: np.ndarray | None,
    grain_rotation_deg: float,
) -> tuple[Mapping[tuple[int, int], float], Mapping[tuple[int, int], float]]:
    edge_costs: MutableMapping[tuple[int, int], float] = {}
    fabric_costs: MutableMapping[tuple[int, int], float] = {}
    for edge in panel_edges:
        kernel = kernels.get(edge)
        if kernel is None:
            continue
        total, fabric_component = _edge_cost_with_fabric(
            edge,
            kernel,
            weights,
            fabric_profile=fabric_profile,
            fabric_grain=fabric_grain,
            grain_rotation_deg=grain_rotation_deg,
            vertices=vertices,
        )
        edge_costs[edge] = total
        fabric_costs[edge] = fabric_component
    return edge_costs, fabric_costs


def _edges_for_panel(panel, kernels: Mapping[tuple[int, int], EdgeKernel]) -> list[tuple[int, int]]:
    seam_vertices = {int(idx) for idx in panel.seam_vertices}
    return [
        edge for edge in kernels.keys() if edge[0] in seam_vertices and edge[1] in seam_vertices
    ]


def _panel_solution_from_edges(
    panel,
    edges: Iterable[tuple[int, int]],
    *,
    vertex_costs: np.ndarray,
    edge_costs: Mapping[tuple[int, int], float],
    symmetry_penalty_weight: float,
    constraints: ConstraintRegistry | None,
    vertex_weight: float,
    fabric_costs: Mapping[tuple[int, int], float] | None = None,
    fabric_assignment: FabricAssignment | None = None,
) -> PanelSolution:
    panel_vertices = tuple(int(idx) for idx in panel.seam_vertices)
    selected_edges = tuple(sorted({tuple(sorted(edge)) for edge in edges}))
    parent = _component_roots(panel_vertices)

    edge_cost_sum = 0.0
    fabric_sum = 0.0
    for edge in selected_edges:
        if edge[0] not in panel_vertices or edge[1] not in panel_vertices:
            continue
        _union(parent, edge[0], edge[1])
        edge_cost_sum += float(edge_costs.get(edge, 0.0))
        if fabric_costs is not None:
            fabric_sum += float(fabric_costs.get(edge, 0.0))

    components = {_find(parent, vertex) for vertex in panel_vertices} if panel_vertices else set()
    connectivity_penalty = 0.0
    if len(components) > 1:
        connectivity_penalty = 1000.0 * float(len(components) - 1)

    vertex_term = float(np.nansum(np.asarray(vertex_costs, dtype=float)[list(panel_vertices)])) * float(vertex_weight)
    symmetry_term = _symmetry_penalty(selected_edges, constraints, symmetry_penalty_weight)

    total_cost = edge_cost_sum + vertex_term + symmetry_term + connectivity_penalty
    warnings = []
    if not selected_edges:
        warnings.append("panel has no seam edges")
    if connectivity_penalty > 0.0:
        warnings.append("panel is disconnected")
    breakdown = {
        "edge_energy": edge_cost_sum,
        "vertex_cost": vertex_term,
        "symmetry_penalty": symmetry_term,
        "connectivity_penalty": connectivity_penalty,
    }
    if fabric_costs is not None:
        breakdown["fabric_cost"] = fabric_sum
    return PanelSolution(
        panel=panel.name,
        edges=selected_edges,
        vertices=panel_vertices,
        cost=total_cost,
        cost_breakdown=breakdown,
        warnings=tuple(warnings),
        fabric_id=fabric_assignment.fabric_id if fabric_assignment else None,
        grain_rotation_deg=fabric_assignment.grain_rotation_deg if fabric_assignment else 0.0,
    )


def _assemble_solution(
    seam_graph: SeamGraph,
    panel_solutions: Mapping[str, PanelSolution],
    *,
    solver: SolverMode,
    data_cost: float,
    mdl_cost_value: float,
    metadata: Mapping[str, float],
    warnings: list[str],
) -> SeamSolution:
    return SeamSolution(
        solver=solver,
        panel_solutions=dict(panel_solutions),
        total_cost=data_cost + mdl_cost_value,
        warnings=tuple(warnings),
        metadata=metadata,
    )


def _data_cost_from_panels(panel_solutions: Mapping[str, PanelSolution]) -> float:
    total = 0.0
    for panel in panel_solutions.values():
        total += float(panel.cost)
    return total


def _initial_solution(
    seam_graph: SeamGraph,
    kernels: Mapping[tuple[int, int], EdgeKernel],
    weights: KernelWeights,
    cost_field: SeamCostField,
    constraints: ConstraintRegistry | None,
    vertices: np.ndarray,
    vertex_weight: float,
    symmetry_penalty_weight: float,
    *,
    fabric_catalog: Mapping[str, FabricProfile] | None,
    fabric_grain: np.ndarray | None,
    panel_fabrics: Mapping[str, FabricAssignment] | None,
    default_fabric_id: str | None,
) -> Mapping[str, PanelSolution]:
    solutions: MutableMapping[str, PanelSolution] = {}
    for panel in seam_graph.panels:
        seam_vertices = tuple(int(idx) for idx in panel.seam_vertices)
        panel_edges = [
            edge for edge in kernels.keys() if edge[0] in seam_vertices and edge[1] in seam_vertices
        ]
        assignment = panel_fabrics.get(panel.name) if panel_fabrics else None
        if assignment is None:
            assignment = FabricAssignment(default_fabric_id, 0.0)
        fabric_profile = _fabric_profile_for(assignment.fabric_id, fabric_catalog)
        edge_costs, fabric_costs = _edge_cost_maps(
            panel_edges,
            kernels,
            weights,
            vertices=vertices,
            fabric_profile=fabric_profile,
            fabric_grain=fabric_grain,
            grain_rotation_deg=assignment.grain_rotation_deg,
        )
        parent = _component_roots(seam_vertices)
        chosen: list[tuple[int, int]] = []
        for edge in sorted(panel_edges, key=lambda edge: (edge_costs.get(edge, 0.0), edge)):
            if _union(parent, edge[0], edge[1]):
                chosen.append(edge)
        solutions[panel.name] = _panel_solution_from_edges(
            panel,
            chosen,
            vertex_costs=cost_field.vertex_costs,
            edge_costs=edge_costs,
            symmetry_penalty_weight=symmetry_penalty_weight,
            constraints=constraints,
            vertex_weight=vertex_weight,
            fabric_costs=fabric_costs,
            fabric_assignment=assignment,
        )
    return solutions


def _witness_stable(
    solution: SeamSolution,
    seam_graph: SeamGraph,
    kernels: Mapping[tuple[int, int], EdgeKernel],
    cost_field: SeamCostField,
    constraints: ConstraintRegistry | None,
    mdl_prior: MDLPrior,
    vertices: np.ndarray,
    *,
    fabric_catalog: Mapping[str, FabricProfile] | None,
    fabric_grain: np.ndarray | None,
    weights: KernelWeights,
    vertex_weight: float,
    symmetry_penalty_weight: float,
    noise: float,
) -> bool:
    default_fabric_id = next(iter(fabric_catalog.keys()), None) if fabric_catalog else None
    fabric_assignments = {
        name: FabricAssignment(panel.fabric_id, panel.grain_rotation_deg) for name, panel in solution.panel_solutions.items()
    }
    for factor in (1.0 + noise, 1.0 - noise):
        scaled_weights = KernelWeights(
            rom_mean=weights.rom_mean * factor,
            rom_max=weights.rom_max * factor,
            rom_grad=weights.rom_grad * factor,
            curvature=weights.curvature * factor,
            fabric=weights.fabric * factor,
        )
        baseline = _initial_solution(
            seam_graph,
            kernels,
            scaled_weights,
            cost_field,
            constraints,
            vertices,
            vertex_weight,
            symmetry_penalty_weight,
            fabric_catalog=fabric_catalog,
            fabric_grain=fabric_grain,
            panel_fabrics=fabric_assignments,
            default_fabric_id=default_fabric_id,
        )
        for panel_name, panel_solution in baseline.items():
            candidate_edges = solution.panel_solutions[panel_name].edges
            if set(panel_solution.edges) != set(candidate_edges):
                return False
        baseline_solution = _assemble_solution(
            seam_graph,
            baseline,
            solver="pda_baseline",
            data_cost=_data_cost_from_panels(baseline),
            mdl_cost_value=0.0,
            metadata={},
            warnings=[],
        )
        baseline_mdl, _ = mdl_cost(baseline_solution, mdl_prior, vertices=vertices)
        candidate_mdl, _ = mdl_cost(solution, mdl_prior, vertices=vertices)
        if baseline_mdl < candidate_mdl * 0.5:
            return False
    return True


def _panel_proposals(
    panel_solution: PanelSolution,
    panel_edges: list[tuple[int, int]],
    edge_costs: Mapping[tuple[int, int], float],
) -> list[tuple[tuple[int, int], str, tuple[tuple[int, int], ...]]]:
    proposals = []
    reroute = reroute_edge(panel_solution.edges, panel_edges, edge_costs)
    proposals.append((reroute.edges, reroute.description, reroute.changed_edges))

    shorten = shorten_seam(panel_solution.edges, edge_costs)
    proposals.append((shorten.edges, shorten.description, shorten.changed_edges))

    split = split_panel(panel_solution.edges, panel_edges, edge_costs)
    proposals.append((split.edges, split.description, split.changed_edges))

    merged = merge_panels(panel_solution.edges, tuple())  # no-op merge for compatibility
    proposals.append((merged.edges, merged.description, merged.changed_edges))
    return proposals


def solve_seams_pda(
    seam_graph: "SeamGraph",
    kernels: Mapping[tuple[int, int], EdgeKernel],
    weights: KernelWeights,
    mdl_prior: MDLPrior,
    constraints: ConstraintRegistry | None,
    budget: int,
    *,
    cost_field: SeamCostField,
    vertices: np.ndarray,
    vertex_weight: float = 1.0,
    symmetry_penalty_weight: float = 0.5,
    witness_noise: float = 0.05,
    fabric_catalog: Mapping[str, FabricProfile] | None = None,
    fabric_grain: np.ndarray | None = None,
    panel_fabrics: Mapping[str, str] | None = None,
    grain_rotation_step: float = 10.0,
) -> SeamSolution:
    """PDA seam solver that optimizes ROM kernels under MDL prior."""

    if SeamGraph is None:
        raise ImportError("suit.seam_generator.SeamGraph is required to run the PDA solver.")
    if budget < 0:
        raise ValueError("budget must be non-negative")

    default_fabric_id = next(iter(fabric_catalog.keys()), None) if fabric_catalog else None
    fabric_grain_arr = np.asarray(fabric_grain, dtype=float) if fabric_grain is not None else None
    fabric_assignments = {name: FabricAssignment(fabric_id, 0.0) for name, fabric_id in (panel_fabrics or {}).items()}
    panel_edges = {panel.name: _edges_for_panel(panel, kernels) for panel in seam_graph.panels}

    current_panels = _initial_solution(
        seam_graph,
        kernels,
        weights,
        cost_field,
        constraints,
        vertices,
        vertex_weight,
        symmetry_penalty_weight,
        fabric_catalog=fabric_catalog,
        fabric_grain=fabric_grain_arr,
        panel_fabrics=fabric_assignments,
        default_fabric_id=default_fabric_id,
    )
    current_data_cost = _data_cost_from_panels(current_panels)
    current_temp = SeamSolution("pda", current_panels, current_data_cost, (), {})  # type: ignore[arg-type]
    mdl_value, mdl_breakdown = mdl_cost(current_temp, mdl_prior, vertices=vertices)
    current_solution = SeamSolution(
        solver="pda",
        panel_solutions=current_panels,
        total_cost=current_data_cost + mdl_value,
        warnings=tuple(),
        metadata={
            "data_cost": current_data_cost,
            "mdl_cost": mdl_value,
            "iterations": 0,
            "symmetry_penalty": symmetry_penalty_weight,
            "vertex_weight": vertex_weight,
            "witness_noise": witness_noise,
            **mdl_breakdown,
        },
    )

    for iteration in range(budget):
        improved = False
        for panel in seam_graph.panels:
            panel_solution = current_solution.panel_solutions.get(panel.name)
            if panel_solution is None:
                continue

            current_assignment = FabricAssignment(
                panel_solution.fabric_id or default_fabric_id,
                panel_solution.grain_rotation_deg,
            )
            fabric_profile = _fabric_profile_for(current_assignment.fabric_id, fabric_catalog)
            panel_edge_list = panel_edges.get(panel.name, [])
            edge_costs_current, fabric_costs_current = _edge_cost_maps(
                panel_edge_list,
                kernels,
                weights,
                vertices=vertices,
                fabric_profile=fabric_profile,
                fabric_grain=fabric_grain_arr,
                grain_rotation_deg=current_assignment.grain_rotation_deg,
            )
            proposals = _panel_proposals(panel_solution, panel_edge_list, edge_costs_current)
            best_candidate = None
            best_cost = current_solution.total_cost
            best_metadata: MutableMapping[str, float] = {}

            def _evaluate_candidate(candidate_panel: PanelSolution, description: str, candidate_metadata: Mapping[str, float]) -> None:
                nonlocal best_candidate, best_cost, best_metadata
                candidate_panels = dict(current_solution.panel_solutions)
                candidate_panels[panel.name] = candidate_panel
                candidate_data = _data_cost_from_panels(candidate_panels)
                temp_solution = SeamSolution("pda", candidate_panels, candidate_data, (), current_solution.metadata)
                candidate_mdl, candidate_mdl_breakdown = mdl_cost(temp_solution, mdl_prior, vertices=vertices)
                candidate_total = candidate_data + candidate_mdl
                candidate_solution = SeamSolution(
                    solver="pda",
                    panel_solutions=candidate_panels,
                    total_cost=candidate_total,
                    warnings=tuple(list(current_solution.warnings) + [description]),
                    metadata={
                        **current_solution.metadata,
                        "data_cost": candidate_data,
                        "mdl_cost": candidate_mdl,
                        **candidate_mdl_breakdown,
                        **candidate_metadata,
                    },
                )
                if candidate_total < best_cost and _witness_stable(
                    candidate_solution,
                    seam_graph,
                    kernels,
                    cost_field,
                    constraints,
                    mdl_prior,
                    vertices,
                    fabric_catalog=fabric_catalog,
                    fabric_grain=fabric_grain_arr,
                    weights=weights,
                    vertex_weight=vertex_weight,
                    symmetry_penalty_weight=symmetry_penalty_weight,
                    noise=witness_noise,
                ):
                    best_candidate = candidate_solution
                    best_cost = candidate_total
                    best_metadata = candidate_solution.metadata

            for candidate_edges, description, changed in proposals:
                if not changed:
                    continue
                candidate_panel = _panel_solution_from_edges(
                    panel,
                    candidate_edges,
                    vertex_costs=cost_field.vertex_costs,
                    edge_costs=edge_costs_current,
                    symmetry_penalty_weight=symmetry_penalty_weight,
                    constraints=constraints,
                    vertex_weight=vertex_weight,
                    fabric_costs=fabric_costs_current,
                    fabric_assignment=current_assignment,
                )
                _evaluate_candidate(candidate_panel, description, {})

            if fabric_catalog:
                for fabric_id in fabric_catalog:
                    if fabric_id == current_assignment.fabric_id:
                        continue
                    assignment = FabricAssignment(fabric_id, 0.0)
                    alt_profile = _fabric_profile_for(assignment.fabric_id, fabric_catalog)
                    alt_edge_costs, alt_fabric_costs = _edge_cost_maps(
                        panel_edge_list,
                        kernels,
                        weights,
                        vertices=vertices,
                        fabric_profile=alt_profile,
                        fabric_grain=fabric_grain_arr,
                        grain_rotation_deg=assignment.grain_rotation_deg,
                    )
                    candidate_panel = _panel_solution_from_edges(
                        panel,
                        panel_solution.edges,
                        vertex_costs=cost_field.vertex_costs,
                        edge_costs=alt_edge_costs,
                        symmetry_penalty_weight=symmetry_penalty_weight,
                        constraints=constraints,
                        vertex_weight=vertex_weight,
                        fabric_costs=alt_fabric_costs,
                        fabric_assignment=assignment,
                    )
                    _evaluate_candidate(candidate_panel, f"switch_fabric:{fabric_id}", {"fabric_id": fabric_id})

                if current_assignment.fabric_id is not None:
                    rotation_targets = (
                        _apply_rotation_limit(current_assignment.grain_rotation_deg + grain_rotation_step, fabric_profile),
                        _apply_rotation_limit(current_assignment.grain_rotation_deg - grain_rotation_step, fabric_profile),
                    )
                    for target_rotation in rotation_targets:
                        if target_rotation == current_assignment.grain_rotation_deg:
                            continue
                        assignment = FabricAssignment(current_assignment.fabric_id, target_rotation)
                        rot_edge_costs, rot_fabric_costs = _edge_cost_maps(
                            panel_edge_list,
                            kernels,
                            weights,
                            vertices=vertices,
                            fabric_profile=fabric_profile,
                            fabric_grain=fabric_grain_arr,
                            grain_rotation_deg=assignment.grain_rotation_deg,
                        )
                        candidate_panel = _panel_solution_from_edges(
                            panel,
                            panel_solution.edges,
                            vertex_costs=cost_field.vertex_costs,
                            edge_costs=rot_edge_costs,
                            symmetry_penalty_weight=symmetry_penalty_weight,
                            constraints=constraints,
                            vertex_weight=vertex_weight,
                            fabric_costs=rot_fabric_costs,
                            fabric_assignment=assignment,
                        )
                        _evaluate_candidate(candidate_panel, "rotate_grain", {"grain_rotation_deg": target_rotation})

            if best_candidate is not None:
                current_solution = SeamSolution(
                    solver=best_candidate.solver,
                    panel_solutions=best_candidate.panel_solutions,
                    total_cost=best_cost,
                    warnings=best_candidate.warnings,
                    metadata={
                        **best_metadata,
                        "iterations": iteration + 1,
                    },
                )
                improved = True
        if not improved:
            break

    # Final MDL recalc for reporting consistency
    final_data = _data_cost_from_panels(current_solution.panel_solutions)
    temp_final = SeamSolution("pda", current_solution.panel_solutions, final_data, (), current_solution.metadata)
    final_mdl, final_breakdown = mdl_cost(temp_final, mdl_prior, vertices=vertices)
    return SeamSolution(
        solver=current_solution.solver,
        panel_solutions=current_solution.panel_solutions,
        total_cost=final_data + final_mdl,
        warnings=current_solution.warnings,
        metadata={
            **current_solution.metadata,
            "data_cost": final_data,
            "mdl_cost": final_mdl,
            **final_breakdown,
        },
    )
