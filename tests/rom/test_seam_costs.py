import numpy as np

from suit.seam_generator import MeasurementLoop, SeamGenerator

from smii.rom.aggregation import RomSample, aggregate_fields
from smii.rom.basis import KernelBasis, KernelProjector
from smii.rom.seam_costs import annotate_seam_graph_with_costs, build_seam_cost_field


def test_build_seam_cost_field_maps_edge_weights():
    basis = KernelBasis.from_arrays(np.eye(3))
    projector = KernelProjector(basis)
    edges = [(0, 1), (1, 2)]
    samples = [
        RomSample(pose_id="base", coeffs={"T": np.array([0.0, 0.0, 0.0])}),
        RomSample(pose_id="stressed", coeffs={"T": np.array([2.0, 0.0, 0.0])}),
    ]

    aggregation = aggregate_fields(samples, projector, edges=edges)
    cost_field = build_seam_cost_field(
        aggregation,
        field="T",
        variance_weight=1.0,
        maximum_weight=0.0,
        minimum_cost=1e-3,
    )

    assert cost_field.vertex_costs.shape == (3,)
    np.testing.assert_allclose(cost_field.vertex_costs[0], 1.0)
    assert cost_field.vertex_costs[1] >= 1e-3
    np.testing.assert_allclose(cost_field.edge_costs, np.array([1.0, 1e-3]), rtol=1e-5)
    weights = cost_field.to_edge_weights()
    assert weights[edges[0]] == cost_field.edge_costs[0]
    assert weights[edges[1]] == cost_field.edge_costs[1]


def test_build_seam_cost_field_without_edges():
    basis = KernelBasis.from_arrays(np.eye(2))
    projector = KernelProjector(basis)
    samples = [RomSample(pose_id="solo", coeffs={"phi": np.array([0.5, 1.0])})]

    aggregation = aggregate_fields(samples, projector)
    cost_field = build_seam_cost_field(aggregation, field="phi")

    assert cost_field.edge_costs.size == 0
    assert cost_field.edges == tuple()
    assert cost_field.samples_used == aggregation.per_field["phi"].sample_count


def test_save_and_load_seam_cost_field(tmp_path):
    basis = KernelBasis.from_arrays(np.eye(2))
    projector = KernelProjector(basis)
    samples = [RomSample(pose_id="solo", coeffs={"phi": np.array([1.0, 0.0])})]
    aggregation = aggregate_fields(samples, projector)
    cost_field = build_seam_cost_field(aggregation, field="phi")

    out = tmp_path / "costs.npz"
    from smii.rom.seam_costs import load_seam_cost_field, save_seam_cost_field

    save_seam_cost_field(cost_field, out)
    loaded = load_seam_cost_field(out)
    np.testing.assert_allclose(loaded.vertex_costs, cost_field.vertex_costs)
    assert loaded.field == cost_field.field


def _cylindrical_mesh(levels: dict[str, float], radius: float, segments: int) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, ...]]]:
    vertices: list[list[float]] = []
    loops: dict[str, tuple[int, ...]] = {}
    sorted_levels = sorted(levels.items(), key=lambda item: item[1])
    for name, coord in sorted_levels:
        level_vertices: list[int] = []
        for segment in range(segments):
            theta = (2.0 * np.pi * segment) / segments
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            level_vertices.append(len(vertices))
            vertices.append([x, y, coord])
        loops[name] = tuple(level_vertices)

    faces: list[list[int]] = []
    for lower_index in range(len(sorted_levels) - 1):
        lower_loop = loops[sorted_levels[lower_index][0]]
        upper_loop = loops[sorted_levels[lower_index + 1][0]]
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a = lower_loop[segment]
            b = lower_loop[next_segment]
            c = upper_loop[segment]
            d = upper_loop[next_segment]
            faces.append([a, b, c])
            faces.append([b, d, c])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int), loops


def _face_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        a, b, c = map(int, face)
        for start, end in ((a, b), (b, c), (c, a)):
            edge = (min(start, end), max(start, end))
            edges.add(edge)
    return sorted(edges)


def test_annotate_seam_graph_with_costs_maps_real_seams():
    vertices, faces, loops = _cylindrical_mesh(
        levels={"neck_circumference": 0.5, "waist_circumference": 0.0},
        radius=0.2,
        segments=8,
    )
    axis_map = {
        "longitudinal": np.array([0.0, 0.0, 1.0], dtype=float),
        "lateral": np.array([1.0, 0.0, 0.0], dtype=float),
        "anterior": np.array([0.0, 1.0, 0.0], dtype=float),
    }
    generator = SeamGenerator(base_allowance=0.01, seam_bias=0.0, seam_tolerance=5e-3)
    loop_objects = [
        MeasurementLoop(name, tuple(indices), float(np.mean(vertices[list(indices)] @ axis_map["longitudinal"])))
        for name, indices in loops.items()
    ]
    seam_graph = generator.generate(vertices, faces, loop_objects, axis_map)

    basis = KernelBasis.from_arrays(np.eye(vertices.shape[0]))
    projector = KernelProjector(basis)
    coeffs = np.zeros(vertices.shape[0], dtype=float)
    target_panel = seam_graph.panels[0]
    for idx in target_panel.seam_vertices:
        coeffs[int(idx)] = 10.0

    edges = _face_edges(faces)
    aggregation = aggregate_fields(
        [RomSample(pose_id="rom", coeffs={"stress": coeffs})],
        projector,
        edges=edges,
        field_keys=("stress",),
    )
    cost_field = build_seam_cost_field(
        aggregation,
        field="stress",
        variance_weight=0.0,
        maximum_weight=1.0,
        minimum_cost=1e-6,
    )

    annotated = annotate_seam_graph_with_costs(cost_field, seam_graph)
    assert annotated.seam_costs is not None
    costs = annotated.seam_costs[target_panel.name]
    assert costs["vertex_cost_max"] == 1.0
    assert costs["edge_cost_max"] >= costs["edge_cost_mean"] >= 0.0
    assert costs["samples_used"] == 1
    assert annotated.seam_metadata == seam_graph.seam_metadata
