import numpy as np
import pytest

from smii.rom.constraints import ConstraintRegistry, ConstraintSet
from smii.rom.seam_costs import SeamCostField
from smii.seams.edge_costs import build_edge_costs
from smii.seams.kernels import EdgeKernel, KernelWeights
from smii.seams.solver import solve_seams
from suit.seam_generator import SeamGraph, SeamPanel


def _panel(vertices: np.ndarray, seam_vertices: tuple[int, ...]) -> SeamGraph:
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


def test_build_edge_costs_mean_and_integral():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    seam_graph = _panel(vertices, seam_vertices=(0, 1, 2, 3))
    edges = ((0, 1), (1, 2), (2, 3))
    cost_field = SeamCostField(
        field="phi",
        vertex_costs=np.array([1.0, 3.0, 5.0, 7.0], dtype=float),
        edge_costs=np.zeros(len(edges), dtype=float),
        edges=edges,
        samples_used=1,
        metadata={},
    )

    mean_costs = build_edge_costs(cost_field, seam_graph, vertices=vertices, mode="mean")
    integral_costs = build_edge_costs(cost_field, seam_graph, vertices=vertices, mode="integral")

    np.testing.assert_allclose(mean_costs.costs, np.array([2.0, 4.0, 6.0]))
    np.testing.assert_allclose(integral_costs.costs, np.array([2.0, 8.0, 18.0]))


def test_solve_seams_respects_constraints_and_costs():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    seam_graph = _panel(vertices, seam_vertices=(0, 1, 2))
    edges = ((0, 1), (1, 2))
    cost_field = SeamCostField(
        field="phi",
        vertex_costs=np.array([1.0, 10.0, 1.0], dtype=float),
        edge_costs=np.zeros(len(edges), dtype=float),
        edges=edges,
        samples_used=1,
        metadata={},
    )
    edge_costs = build_edge_costs(cost_field, seam_graph, mode="mean")

    constraint_set = ConstraintSet(
        forbidden_vertices={"zones": (2,)},
        anchors={},
        symmetry_vertex_pairs=tuple(),
        symmetry_edge_pairs=tuple(),
        panel_limits_default={},
        panel_overrides={},
    )
    constraints = ConstraintRegistry(constraint_set)
    with pytest.raises(ValueError):
        solve_seams(seam_graph, cost_field, constraints=constraints, edge_costs=edge_costs)

    safe_constraints = ConstraintRegistry(
        ConstraintSet(
            forbidden_vertices={},
            anchors={},
            symmetry_vertex_pairs=tuple(),
            symmetry_edge_pairs=tuple(),
            panel_limits_default={},
            panel_overrides={},
        )
    )
    solution = solve_seams(seam_graph, cost_field, constraints=safe_constraints, edge_costs=edge_costs, vertex_weight=0.0)
    panel_solution = solution.panel_solutions["panel"]

    assert panel_solution.edges == edges
    assert panel_solution.cost_breakdown["edge_cost"] > 0.0
    assert not panel_solution.warnings


def test_solve_seams_applies_symmetry_penalty():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    seam_graph = _panel(vertices, seam_vertices=(0, 1, 2, 3))
    edges = ((0, 1),)
    cost_field = SeamCostField(
        field="phi",
        vertex_costs=np.ones(len(vertices), dtype=float),
        edge_costs=np.zeros(len(edges), dtype=float),
        edges=edges,
        samples_used=1,
        metadata={},
    )
    edge_costs = build_edge_costs(cost_field, seam_graph, mode="mean")
    constraints = ConstraintRegistry(
        ConstraintSet(
            forbidden_vertices={},
            anchors={},
            symmetry_vertex_pairs=((0, 2), (1, 3)),
            symmetry_edge_pairs=tuple(),
            panel_limits_default={},
            panel_overrides={},
        )
    )

    solution = solve_seams(
        seam_graph,
        cost_field,
        constraints=constraints,
        edge_costs=edge_costs,
        vertex_weight=0.0,
        symmetry_penalty=0.5,
    )
    panel_solution = solution.panel_solutions["panel"]
    assert panel_solution.cost_breakdown["symmetry_penalty"] == pytest.approx(0.5)
    assert panel_solution.cost > panel_solution.cost_breakdown["edge_cost"]


def test_solve_seams_can_use_kernel_edge_costs() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    seam_graph = _panel(vertices, seam_vertices=(0, 1, 2, 3))
    edges = ((0, 1), (1, 2), (2, 3), (0, 3))
    cost_field = SeamCostField(
        field="phi",
        vertex_costs=np.ones(len(vertices), dtype=float),
        edge_costs=np.zeros(len(edges), dtype=float),
        edges=edges,
        samples_used=1,
        metadata={},
    )

    baseline = solve_seams(seam_graph, cost_field, vertex_weight=0.0)
    baseline_edges = baseline.panel_solutions["panel"].edges
    assert (0, 3) in baseline_edges

    kernels = {
        (0, 1): EdgeKernel(rom_mean=0.0, rom_max=0.0, rom_grad=0.0, curvature=0.0, fabric_misalignment=0.0),
        (1, 2): EdgeKernel(rom_mean=0.0, rom_max=0.0, rom_grad=0.0, curvature=0.0, fabric_misalignment=0.0),
        (2, 3): EdgeKernel(rom_mean=0.0, rom_max=0.0, rom_grad=0.0, curvature=0.0, fabric_misalignment=0.0),
        (0, 3): EdgeKernel(rom_mean=0.0, rom_max=0.0, rom_grad=0.0, curvature=0.0, fabric_misalignment=10.0),
    }
    weights = KernelWeights(rom_mean=0.0, rom_max=0.0, rom_grad=0.0, curvature=0.0, fabric=1.0)
    solution = solve_seams(
        seam_graph,
        cost_field,
        kernels=kernels,
        kernel_weights=weights,
        vertex_weight=0.0,
    )

    panel_solution = solution.panel_solutions["panel"]
    assert panel_solution.edges == ((0, 1), (1, 2), (2, 3))
    assert solution.metadata["edge_mode"] == "kernel"
