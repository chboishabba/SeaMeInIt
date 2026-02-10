import numpy as np

from smii.rom.seam_costs import SeamCostField
from smii.seams.edge_costs import build_edge_costs
from smii.seams.kernels import KernelWeights, build_edge_kernels
from smii.seams.mdl import MDLPrior
from smii.seams.pda import solve_seams_pda
from smii.seams.solver import solve_seams
from smii.seams.solvers_mincut import solve_seams_mincut
from smii.seams.solvers_sp import AnchorPair, solve_seams_shortest_path
from suit.seam_generator import SeamGraph, SeamPanel


def _starburst_fixture() -> tuple[SeamGraph, np.ndarray, SeamCostField]:
    # Vertex 0 is a hub connected to a ring (1..5). Cheap hub costs can produce
    # radial starbursts unless solver topology is regularized.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.309, 0.951, 0.0],
            [-0.809, 0.588, 0.0],
            [-0.809, -0.588, 0.0],
            [0.309, -0.951, 0.0],
        ],
        dtype=float,
    )
    panel = SeamPanel(
        name="panel",
        anchor_loops=("a", "b"),
        side="front",
        vertices=vertices,
        faces=np.zeros((0, 3), dtype=int),
        global_indices=tuple(range(len(vertices))),
        seam_vertices=tuple(range(len(vertices))),
        loop_vertex_indices=tuple(),
        metadata={},
    )
    seam_graph = SeamGraph(panels=(panel,), measurement_loops=tuple(), seam_metadata={}, seam_costs=None)

    ring_edges = ((1, 2), (2, 3), (3, 4), (4, 5), (1, 5))
    hub_edges = tuple((0, idx) for idx in range(1, 6))
    edges = tuple(sorted({tuple(sorted(edge)) for edge in ring_edges + hub_edges}))
    cost_field = SeamCostField(
        field="stress",
        vertex_costs=np.array([0.01, 4.0, 4.0, 4.0, 4.0, 4.0], dtype=float),
        edge_costs=np.zeros(len(edges), dtype=float),
        edges=edges,
        samples_used=1,
        metadata={},
    )
    return seam_graph, vertices, cost_field


def _max_degree(edges: tuple[tuple[int, int], ...]) -> int:
    if not edges:
        return 0
    degree: dict[int, int] = {}
    for a, b in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    return max(degree.values())


def test_mst_solver_avoids_starburst_hubs() -> None:
    seam_graph, _, cost_field = _starburst_fixture()
    edge_costs = build_edge_costs(cost_field, seam_graph, mode="mean")

    solution = solve_seams(
        seam_graph,
        cost_field,
        edge_costs=edge_costs,
        vertex_weight=0.0,
        max_branch_degree=2,
        branch_penalty_weight=3.0,
    )

    panel_solution = solution.panel_solutions["panel"]
    assert _max_degree(panel_solution.edges) <= 2
    assert len(panel_solution.edges) == len(panel_solution.vertices) - 1


def test_shortest_path_solver_stays_non_starburst() -> None:
    seam_graph, vertices, cost_field = _starburst_fixture()
    kernels = build_edge_kernels(cost_field, seam_graph, vertices=vertices, fabric_grain=np.array([1.0, 0.0, 0.0]))
    weights = KernelWeights(rom_mean=1.0, rom_max=1.0, rom_grad=0.0, curvature=0.0, fabric=0.0)
    prior = MDLPrior(seam_count=0.0, seam_length=0.0, panel_count=0.0, boundary_roughness=0.0, symmetry_violation=0.0)

    solution = solve_seams_shortest_path(
        seam_graph,
        kernels,
        weights,
        prior,
        constraints=None,
        cost_field=cost_field,
        vertices=vertices,
        anchors={"panel": AnchorPair(source=1, target=4)},
    )

    assert _max_degree(solution.panel_solutions["panel"].edges) <= 2


def test_mincut_solver_avoids_starburst_hubs() -> None:
    seam_graph, vertices, cost_field = _starburst_fixture()
    kernels = build_edge_kernels(cost_field, seam_graph, vertices=vertices, fabric_grain=np.array([1.0, 0.0, 0.0]))
    weights = KernelWeights(rom_mean=1.0, rom_max=1.0, rom_grad=0.0, curvature=0.0, fabric=0.0)
    prior = MDLPrior(seam_count=0.0, seam_length=0.0, panel_count=0.0, boundary_roughness=0.0, symmetry_violation=0.0)

    solution = solve_seams_mincut(
        seam_graph,
        kernels,
        weights,
        prior,
        constraints=None,
        cost_field=cost_field,
        vertices=vertices,
        max_branch_degree=2,
        branch_penalty_weight=3.0,
    )

    panel_solution = solution.panel_solutions["panel"]
    assert panel_solution.edges
    assert _max_degree(panel_solution.edges) <= 2


def test_pda_solver_avoids_starburst_hubs() -> None:
    seam_graph, vertices, cost_field = _starburst_fixture()
    kernels = build_edge_kernels(cost_field, seam_graph, vertices=vertices, fabric_grain=np.array([1.0, 0.0, 0.0]))
    weights = KernelWeights(rom_mean=1.0, rom_max=1.0, rom_grad=0.0, curvature=0.0, fabric=0.0)
    prior = MDLPrior(seam_count=0.5, seam_length=0.1, panel_count=0.0, boundary_roughness=0.0, symmetry_violation=0.0)

    solution = solve_seams_pda(
        seam_graph,
        kernels,
        weights,
        prior,
        constraints=None,
        budget=3,
        cost_field=cost_field,
        vertices=vertices,
        vertex_weight=0.0,
        symmetry_penalty_weight=0.0,
        max_branch_degree=2,
        branch_penalty_weight=3.0,
        witness_noise=0.0,
    )

    panel_solution = solution.panel_solutions["panel"]
    assert panel_solution.edges
    assert _max_degree(panel_solution.edges) <= 2
