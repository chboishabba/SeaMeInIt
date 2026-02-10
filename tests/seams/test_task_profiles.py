import numpy as np

from smii.rom.basis import KernelBasis, KernelProjector
from smii.rom.aggregation import RomSample
from smii.seams.task_profiles import aggregate_rom_for_task, load_task_profile


def test_load_task_profile_normalizes_weights():
    profile = load_task_profile("configs/tasks/reach_overhead_v1.yaml")
    weights = profile.normalized_weights
    assert profile.task_id == "reach_overhead_v1"
    assert set(weights.keys()) >= {"keyposes", "sweeps"}
    np.testing.assert_allclose(sum(weights.values()), 1.0)


def test_aggregate_rom_for_task_applies_weights():
    basis = KernelBasis.from_arrays(np.eye(1))
    projector = KernelProjector(basis)
    samples = [
        RomSample(pose_id="a", coeffs={"stress": np.array([1.0])}, observations={"task_component": "keyposes"}),
        RomSample(pose_id="b", coeffs={"stress": np.array([3.0])}, observations={"task_component": "sweeps"}),
    ]
    profile = load_task_profile("configs/tasks/reach_overhead_v1.yaml")
    aggregation = aggregate_rom_for_task(samples, projector, profile, field_keys=("stress",))

    mean_value = float(aggregation.per_field["stress"].mean[0])
    assert aggregation.per_field["stress"].sample_count == 2
    assert mean_value > 2.0  # task mixture downweights the first sample
