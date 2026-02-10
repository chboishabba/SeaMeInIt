import numpy as np

from smii.rom.seam_costs import SeamCostField
from smii.seams.fabric_kernels import fabric_penalty, load_fabric_profile, load_fabrics_from_dir
from smii.seams.kernels import build_edge_kernels
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


def test_load_fabric_profile_reads_yaml():
    profile = load_fabric_profile("configs/fabrics/knit_4way_light.yaml")
    assert profile.fabric_id == "knit_4way_light"
    assert profile.compliance.s_parallel > profile.compliance.s_perp
    assert profile.constraints.allow_bias is True


def test_fabric_penalty_increases_with_misalignment():
    profile = load_fabric_profile("configs/fabrics/woven_2way.yaml")
    grain = np.array([1.0, 0.0, 0.0], dtype=float)
    along_grain = fabric_penalty(np.array([1.0, 0.0, 0.0]), grain, profile)
    forty_five = fabric_penalty(np.array([1.0, 1.0, 0.0]), grain, profile)
    perpendicular = fabric_penalty(np.array([0.0, 1.0, 0.0]), grain, profile)
    assert along_grain <= forty_five <= perpendicular + 1e-6
    assert along_grain == 0.0
    assert perpendicular >= forty_five


def test_build_edge_kernels_respects_fabric_rotation():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    seam_graph = _seam_graph(vertices, seam_vertices=(0, 1))
    cost_field = SeamCostField(
        field="stress",
        vertex_costs=np.array([1.0, 1.0], dtype=float),
        edge_costs=np.zeros(1, dtype=float),
        edges=((0, 1),),
        samples_used=1,
        metadata={},
    )
    fabrics = load_fabrics_from_dir("configs/fabrics")
    profile = fabrics["woven_2way"]

    kernels_aligned = build_edge_kernels(
        cost_field,
        seam_graph,
        vertices=vertices,
        fabric_grain=np.array([1.0, 0.0, 0.0]),
        fabric_profile=profile,
    )
    kernels_rotated = build_edge_kernels(
        cost_field,
        seam_graph,
        vertices=vertices,
        fabric_grain=np.array([1.0, 0.0, 0.0]),
        fabric_profile=profile,
        grain_rotation_deg=90.0,
    )

    aligned_penalty = next(iter(kernels_aligned.values())).fabric_misalignment
    rotated_penalty = next(iter(kernels_rotated.values())).fabric_misalignment
    assert aligned_penalty == 0.0
    assert rotated_penalty > aligned_penalty
