import numpy as np

from smii.rom.constraints import ConstraintRegistry, ConstraintSet
from smii.rom.seam_costs import SeamCostField
from smii.seams.diagnostics import build_diagnostics_report
from smii.seams.kernels import EdgeKernel, KernelWeights
from smii.seams.mdl import MDLPrior
from smii.seams.solvers_mincut import solve_seams_mincut
from smii.seams.solvers_sp import AnchorPair, solve_seams_shortest_path
from suit.seam_generator import SeamGraph, SeamPanel


def _seam_graph(
    vertices: np.ndarray,
    seam_vertices: tuple[int, ...],
    *,
    loop_vertices: tuple[tuple[str, tuple[int, ...]], ...] = tuple(),
) -> SeamGraph:
    panel = SeamPanel(
        name="panel",
        anchor_loops=("a", "b"),
        side="front",
        vertices=vertices,
        faces=np.zeros((0, 3), dtype=int),
        global_indices=tuple(range(len(vertices))),
        seam_vertices=seam_vertices,
        loop_vertex_indices=loop_vertices,
        metadata={},
    )
    return SeamGraph(panels=(panel,), measurement_loops=tuple(), seam_metadata={}, seam_costs=None)


def _is_cycle(edges: tuple[tuple[int, int], ...]) -> bool:
    if not edges:
        return False
    degree: dict[int, int] = {}
    for a, b in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    return all(value == 2 for value in degree.values())


def test_shortest_path_solver_follows_anchors():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    seam_graph = _seam_graph(vertices, seam_vertices=(0, 1, 2))
    edges = ((0, 1), (1, 2), (0, 2))
    kernels = {edge: EdgeKernel(1.0, 1.0, 0.0, 0.0, 0.0) for edge in edges}
    weights = KernelWeights()
    prior = MDLPrior(
        seam_count=0.1,
        seam_length=0.1,
        panel_count=0.0,
        boundary_roughness=0.0,
        symmetry_violation=0.0,
    )
    cost_field = SeamCostField(
        field="stress",
        vertex_costs=np.ones(len(vertices), dtype=float),
        edge_costs=np.zeros(len(edges), dtype=float),
        edges=edges,
        samples_used=1,
        metadata={},
    )
    anchors = {"panel": AnchorPair(source=0, target=2)}
    solution = solve_seams_shortest_path(
        seam_graph,
        kernels,
        weights,
        prior,
        constraints=None,
        cost_field=cost_field,
        vertices=vertices,
        anchors=anchors,
    )

    panel_solution = solution.panel_solutions["panel"]
    assert panel_solution.edges == ((0, 1), (1, 2))
    assert solution.total_cost > 0.0


def test_shortest_path_loop_mode_builds_cycle_from_two_routes():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    seam_graph = _seam_graph(vertices, seam_vertices=(0, 1, 2, 3))
    edges = ((0, 1), (1, 2), (0, 3), (2, 3))
    kernels = {edge: EdgeKernel(1.0, 1.0, 0.0, 0.0, 0.0) for edge in edges}
    weights = KernelWeights()
    prior = MDLPrior(
        seam_count=0.0,
        seam_length=0.0,
        panel_count=0.0,
        boundary_roughness=0.0,
        symmetry_violation=0.0,
    )
    cost_field = SeamCostField(
        field="stress",
        vertex_costs=np.ones(len(vertices), dtype=float),
        edge_costs=np.zeros(len(edges), dtype=float),
        edges=edges,
        samples_used=1,
        metadata={},
    )
    solution = solve_seams_shortest_path(
        seam_graph,
        kernels,
        weights,
        prior,
        constraints=None,
        cost_field=cost_field,
        vertices=vertices,
        anchors={"panel": AnchorPair(source=0, target=2)},
        require_loop=True,
    )
    panel_solution = solution.panel_solutions["panel"]
    assert _is_cycle(panel_solution.edges)
    assert len(panel_solution.edges) == 4


def test_shortest_path_loop_mode_warns_when_only_single_route_available():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    seam_graph = _seam_graph(vertices, seam_vertices=(0, 1, 2))
    edges = ((0, 1), (1, 2))
    kernels = {edge: EdgeKernel(1.0, 1.0, 0.0, 0.0, 0.0) for edge in edges}
    weights = KernelWeights()
    prior = MDLPrior()
    cost_field = SeamCostField(
        field="stress",
        vertex_costs=np.ones(len(vertices), dtype=float),
        edge_costs=np.zeros(len(edges), dtype=float),
        edges=edges,
        samples_used=1,
        metadata={},
    )
    solution = solve_seams_shortest_path(
        seam_graph,
        kernels,
        weights,
        prior,
        constraints=None,
        cost_field=cost_field,
        vertices=vertices,
        anchors={"panel": AnchorPair(source=0, target=2)},
        require_loop=True,
    )
    panel_solution = solution.panel_solutions["panel"]
    assert "loop closure unavailable" in " ".join(panel_solution.warnings)
    assert panel_solution.edges == ((0, 1), (1, 2))


def test_shortest_path_loop_mode_strict_drops_non_cycles():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    seam_graph = _seam_graph(vertices, seam_vertices=(0, 1, 2))
    edges = ((0, 1), (1, 2))
    kernels = {edge: EdgeKernel(1.0, 1.0, 0.0, 0.0, 0.0) for edge in edges}
    weights = KernelWeights()
    prior = MDLPrior()
    cost_field = SeamCostField(
        field="stress",
        vertex_costs=np.ones(len(vertices), dtype=float),
        edge_costs=np.zeros(len(edges), dtype=float),
        edges=edges,
        samples_used=1,
        metadata={},
    )
    solution = solve_seams_shortest_path(
        seam_graph,
        kernels,
        weights,
        prior,
        constraints=None,
        cost_field=cost_field,
        vertices=vertices,
        anchors={"panel": AnchorPair(source=0, target=2)},
        require_loop=True,
        strict_loop=True,
    )
    panel_solution = solution.panel_solutions["panel"]
    assert panel_solution.edges == tuple()
    assert "loop closure unavailable" in " ".join(panel_solution.warnings)


def test_shortest_path_loop_mode_supports_multiple_disjoint_loops():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # loop 1
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],  # loop 2
            [4.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    seam_graph = _seam_graph(
        vertices,
        seam_vertices=tuple(range(8)),
        loop_vertices=(
            ("a", (0, 4)),
            ("b", (2, 6)),
        ),
    )
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
        (4, 5),
        (5, 6),
        (6, 7),
        (4, 7),
    )
    kernels = {edge: EdgeKernel(1.0, 1.0, 0.0, 0.0, 0.0) for edge in edges}
    weights = KernelWeights()
    prior = MDLPrior(
        seam_count=0.0,
        seam_length=0.0,
        panel_count=0.0,
        boundary_roughness=0.0,
        symmetry_violation=0.0,
    )
    cost_field = SeamCostField(
        field="stress",
        vertex_costs=np.ones(len(vertices), dtype=float),
        edge_costs=np.zeros(len(edges), dtype=float),
        edges=edges,
        samples_used=1,
        metadata={},
    )
    solution = solve_seams_shortest_path(
        seam_graph,
        kernels,
        weights,
        prior,
        constraints=None,
        cost_field=cost_field,
        vertices=vertices,
        anchors={"panel": AnchorPair(source=0, target=2)},
        require_loop=True,
        strict_loop=True,
        loop_count=2,
        loop_waypoint_count=2,
    )
    panel_solution = solution.panel_solutions["panel"]
    assert len(panel_solution.edges) == 8
    assert set(panel_solution.edges) == set(edges)


def test_shortest_path_symmetry_penalty_added_to_cost() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    seam_graph = _seam_graph(vertices, seam_vertices=(0, 1, 2, 3))
    edges = ((0, 1), (1, 2), (2, 3))
    kernels = {edge: EdgeKernel(1.0, 1.0, 0.0, 0.0, 0.0) for edge in edges}
    weights = KernelWeights()
    prior = MDLPrior(
        seam_count=0.0,
        seam_length=0.0,
        panel_count=0.0,
        boundary_roughness=0.0,
        symmetry_violation=0.0,
    )
    cost_field = SeamCostField(
        field="stress",
        vertex_costs=np.ones(len(vertices), dtype=float),
        edge_costs=np.zeros(len(edges), dtype=float),
        edges=edges,
        samples_used=1,
        metadata={},
    )
    constraints = ConstraintRegistry(
        ConstraintSet(
            forbidden_vertices={},
            anchors={},
            symmetry_vertex_pairs=((0, 3), (1, 2)),
            symmetry_edge_pairs=tuple(),
            panel_limits_default={},
            panel_overrides={},
        )
    )
    solution = solve_seams_shortest_path(
        seam_graph,
        kernels,
        weights,
        prior,
        constraints=constraints,
        cost_field=cost_field,
        vertices=vertices,
        anchors={"panel": AnchorPair(source=0, target=2)},
        symmetry_penalty_weight=2.0,
    )
    panel_solution = solution.panel_solutions["panel"]
    assert panel_solution.cost_breakdown["symmetry_penalty"] > 0.0


def test_mincut_solver_separates_partitions():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    seam_graph = _seam_graph(vertices, seam_vertices=(0, 1, 2, 3))
    edges = ((0, 1), (1, 3), (2, 3), (0, 2), (1, 2))
    kernels = {edge: EdgeKernel(1.0, 0.0, 0.0, 0.0, 0.0) for edge in edges}
    weights = KernelWeights()
    prior = MDLPrior()
    cost_field = SeamCostField(
        field="stress",
        vertex_costs=np.ones(len(vertices), dtype=float),
        edge_costs=np.zeros(len(edges), dtype=float),
        edges=edges,
        samples_used=1,
        metadata={},
    )
    solution = solve_seams_mincut(
        seam_graph,
        kernels,
        weights,
        prior,
        constraints=None,
        cost_field=cost_field,
        vertices=vertices,
    )
    panel_solution = solution.panel_solutions["panel"]
    assert panel_solution.edges


def test_diagnostics_report_contains_panels():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    seam_graph = _seam_graph(vertices, seam_vertices=(0, 1))
    edges = ((0, 1),)
    kernels = {edge: EdgeKernel(1.0, 1.0, 1.0, 0.0, 0.0) for edge in edges}
    weights = KernelWeights()
    prior = MDLPrior()
    cost_field = SeamCostField(
        field="stress",
        vertex_costs=np.ones(len(vertices), dtype=float),
        edge_costs=np.zeros(len(edges), dtype=float),
        edges=edges,
        samples_used=1,
        metadata={},
    )
    solution = solve_seams_shortest_path(
        seam_graph,
        kernels,
        weights,
        prior,
        constraints=None,
        cost_field=cost_field,
        vertices=vertices,
    )
    report = build_diagnostics_report(
        seam_graph,
        solution,
        kernels,
        weights,
        prior,
        vertices=vertices,
        cost_field=cost_field,
    )
    assert "panels" in report and "panel" in report["panels"]
