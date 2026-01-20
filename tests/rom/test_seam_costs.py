import numpy as np

from smii.rom.aggregation import RomSample, aggregate_fields
from smii.rom.basis import KernelBasis, KernelProjector
from smii.rom.seam_costs import build_seam_cost_field


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
