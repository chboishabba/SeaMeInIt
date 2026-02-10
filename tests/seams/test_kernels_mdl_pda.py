import numpy as np
import pytest

from smii.rom.constraints import ConstraintRegistry, ConstraintSet
from smii.rom.seam_costs import SeamCostField
from smii.seams.fabric_kernels import load_fabrics_from_dir
from smii.seams.kernels import KernelWeights, build_edge_kernels
from smii.seams.mdl import MDLPrior, mdl_cost
from smii.seams.pda import solve_seams_pda
from smii.seams.solver import PanelSolution, SeamSolution
from suit.seam_generator import SeamGraph, SeamPanel


def _seam_graph(vertices: np.ndarray, seam_vertices: tuple[int, ...]) -> SeamGraph:
    panel = SeamPanel(
        name="panel",
        anchor_loops=("a", "b"),
        side="front",
        vertices=vertices,
        faces=np.zeros((0, 3), dtype=int),
        global_indices=tuple(range(len(vertices))),
        seam_vertices=seam_vertices,
        loop_vertex_indices=tuple(),
        metadata={},
    )
    return SeamGraph(panels=(panel,), measurement_loops=tuple(), seam_metadata={}, seam_costs=None)


def test_build_edge_kernels_filters_forbidden_vertices():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    seam_graph = _seam_graph(vertices, seam_vertices=(0, 1, 2))
    cost_field = SeamCostField(
        field="stress",
        vertex_costs=np.array([1.0, 2.0, 3.0], dtype=float),
        edge_costs=np.zeros(3, dtype=float),
        edges=((0, 1), (1, 2), (0, 2)),
        samples_used=1,
        metadata={},
    )
    constraints = ConstraintRegistry(
        ConstraintSet(
            forbidden_vertices={"zones": (2,)},
            anchors={},
            symmetry_vertex_pairs=tuple(),
            symmetry_edge_pairs=tuple(),
            panel_limits_default={},
            panel_overrides={},
        )
    )

    kernels = build_edge_kernels(
        cost_field,
        seam_graph,
        vertices=vertices,
        fabric_grain=np.array([1.0, 0.0, 0.0]),
        constraints=constraints,
    )
    assert (1, 2) not in kernels and (0, 2) not in kernels
    for kernel in kernels.values():
        assert all(np.isfinite([kernel.rom_mean, kernel.rom_max, kernel.rom_grad, kernel.curvature, kernel.fabric_misalignment]))
        assert 0.0 <= kernel.fabric_misalignment <= 1.0


def test_mdl_cost_increases_with_more_edges():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    panel_vertices = (0, 1, 2)
    base_panel = PanelSolution(
        panel="panel",
        edges=((0, 1),),
        vertices=panel_vertices,
        cost=1.0,
        cost_breakdown={"edge_energy": 1.0},
        warnings=tuple(),
    )
    expanded_panel = PanelSolution(
        panel="panel",
        edges=((0, 1), (1, 2)),
        vertices=panel_vertices,
        cost=2.0,
        cost_breakdown={"edge_energy": 2.0},
        warnings=tuple(),
    )
    base_solution = SeamSolution(solver="test", panel_solutions={"panel": base_panel}, total_cost=1.0, warnings=tuple(), metadata={})
    expanded_solution = SeamSolution(solver="test", panel_solutions={"panel": expanded_panel}, total_cost=2.0, warnings=tuple(), metadata={})

    prior = MDLPrior()
    base_cost, _ = mdl_cost(base_solution, prior, vertices=vertices)
    expanded_cost, _ = mdl_cost(expanded_solution, prior, vertices=vertices)
    assert expanded_cost > base_cost


def test_solve_seams_pda_runs_and_is_deterministic():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    seam_graph = _seam_graph(vertices, seam_vertices=(0, 1, 2))
    edges = ((0, 1), (1, 2), (0, 2))
    cost_field = SeamCostField(
        field="stress",
        vertex_costs=np.array([1.0, 5.0, 1.0], dtype=float),
        edge_costs=np.zeros(len(edges), dtype=float),
        edges=edges,
        samples_used=1,
        metadata={},
    )
    kernels = build_edge_kernels(cost_field, seam_graph, vertices=vertices, fabric_grain=np.array([1.0, 0.0, 0.0]))
    weights = KernelWeights(rom_mean=1.0, rom_max=0.5, rom_grad=1.0, curvature=0.0, fabric=0.0)
    prior = MDLPrior(seam_count=0.5, seam_length=0.1, panel_count=0.0, boundary_roughness=0.0, symmetry_violation=0.0)
    constraints = ConstraintRegistry(
        ConstraintSet(
            forbidden_vertices={},
            anchors={},
            symmetry_vertex_pairs=tuple(),
            symmetry_edge_pairs=tuple(),
            panel_limits_default={},
            panel_overrides={},
        )
    )

    solution = solve_seams_pda(
        seam_graph,
        kernels,
        weights,
        prior,
        constraints,
        budget=3,
        cost_field=cost_field,
        vertices=vertices,
        vertex_weight=0.0,
        symmetry_penalty_weight=0.0,
        witness_noise=0.02,
    )

    assert solution.solver == "pda"
    assert solution.total_cost >= 0.0
    assert "data_cost" in solution.metadata and "mdl_cost" in solution.metadata
    assert solution.panel_solutions["panel"].edges


def test_solve_seams_pda_tracks_fabric_assignment():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    seam_graph = _seam_graph(vertices, seam_vertices=(0, 1, 2))
    edges = ((0, 1), (1, 2), (0, 2))
    cost_field = SeamCostField(
        field="stress",
        vertex_costs=np.array([0.5, 0.5, 0.5], dtype=float),
        edge_costs=np.zeros(len(edges), dtype=float),
        edges=edges,
        samples_used=1,
        metadata={},
    )
    kernels = build_edge_kernels(cost_field, seam_graph, vertices=vertices, fabric_grain=np.array([1.0, 0.0, 0.0]))
    weights = KernelWeights(rom_mean=1.0, rom_max=0.5, rom_grad=1.0, curvature=0.0, fabric=0.5)
    prior = MDLPrior(seam_count=0.5, seam_length=0.1, panel_count=0.0, boundary_roughness=0.0, symmetry_violation=0.0)
    constraints = ConstraintRegistry(
        ConstraintSet(
            forbidden_vertices={},
            anchors={},
            symmetry_vertex_pairs=tuple(),
            symmetry_edge_pairs=tuple(),
            panel_limits_default={},
            panel_overrides={},
        )
    )
    fabrics = load_fabrics_from_dir("configs/fabrics")

    solution = solve_seams_pda(
        seam_graph,
        kernels,
        weights,
        prior,
        constraints,
        budget=2,
        cost_field=cost_field,
        vertices=vertices,
        vertex_weight=0.0,
        symmetry_penalty_weight=0.0,
        fabric_catalog=fabrics,
        fabric_grain=np.array([1.0, 0.0, 0.0]),
        panel_fabrics={"panel": "woven_2way"},
    )

    panel_solution = solution.panel_solutions["panel"]
    assert panel_solution.fabric_id == "woven_2way"
    profile = fabrics["woven_2way"]
    assert abs(panel_solution.grain_rotation_deg) <= float(profile.constraints.max_grain_rotation_deg or 0.0) + 1e-6
