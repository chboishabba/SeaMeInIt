from pathlib import Path

import numpy as np

from smii.rom.pose_schedule import load_schedule_samples
from smii.rom.pose_legality import LegalityConfig
from smii.rom.sampler_real import ParameterLayout, _build_joint_map


def test_schedule_expansion_counts_and_determinism():
    schedule_path = Path("data/rom/sweep_schedule.yaml")
    layout = ParameterLayout.from_shapes({"global_orient": (1, 3), "body_pose": (1, 63)})
    base_pose = {"global_orient": np.zeros(3, dtype=float), "body_pose": np.zeros(63, dtype=float)}
    joint_map = _build_joint_map()

    neutral, samples, meta = load_schedule_samples(
        schedule_path,
        layout_blocks=[block.name for block in layout.blocks],
        joint_map=joint_map,
        base_pose=base_pose,
        seed_override=123,
    )

    assert neutral["body_pose"].shape[0] == 63
    assert meta["counts"]["l0"] > 0 and meta["counts"]["l1"] > 0 and meta["counts"]["l2"] > 0
    assert len(samples) == meta["counts"]["l0"] + meta["counts"]["l1"] + meta["counts"]["l2"]

    # Deterministic under same seed.
    _, samples_again, meta_again = load_schedule_samples(
        schedule_path,
        layout_blocks=[block.name for block in layout.blocks],
        joint_map=joint_map,
        base_pose=base_pose,
        seed_override=123,
    )
    for s1, s2 in zip(samples[:10], samples_again[:10]):
        assert s1.pose_id == s2.pose_id
        for key in s1.parameters:
            np.testing.assert_allclose(s1.parameters[key], s2.parameters[key])
    assert meta_again["seed_used"] == 123


def test_spine_angles_are_distributed():
    schedule_path = Path("data/rom/sweep_schedule.yaml")
    layout = ParameterLayout.from_shapes({"global_orient": (1, 3), "body_pose": (1, 63)})
    base_pose = {"global_orient": np.zeros(3, dtype=float), "body_pose": np.zeros(63, dtype=float)}
    joint_map = _build_joint_map()

    _, samples, _ = load_schedule_samples(
        schedule_path,
        layout_blocks=[block.name for block in layout.blocks],
        joint_map=joint_map,
        base_pose=base_pose,
        seed_override=5,
    )
    spine_sample = next(sample for sample in samples if sample.pose_id.startswith("L0-spine-bend"))
    pose = spine_sample.parameters["body_pose"]
    offsets = [joint_map["spine1"], joint_map["spine2"], joint_map["spine3"]]
    non_zero = []
    for offset in offsets:
        non_zero.append(np.linalg.norm(pose[offset : offset + 3]) > 0)
    assert all(non_zero)


def test_legality_filter_and_weight_are_deterministic():
    schedule_path = Path("data/rom/sweep_schedule.yaml")
    layout = ParameterLayout.from_shapes({"global_orient": (1, 3), "body_pose": (1, 63)})
    base_pose = {"global_orient": np.zeros(3, dtype=float), "body_pose": np.zeros(63, dtype=float)}
    joint_map = _build_joint_map()

    _, samples_with_flags, _ = load_schedule_samples(
        schedule_path,
        layout_blocks=[block.name for block in layout.blocks],
        joint_map=joint_map,
        base_pose=base_pose,
        seed_override=7,
        legality_config=LegalityConfig(alpha=1.0),
        filter_illegal=True,
        legality_weight=0.5,
    )
    assert samples_with_flags  # still expanded
    assert any(s.metadata and s.metadata.get("filter_illegal") for s in samples_with_flags)
    assert any(s.metadata and "legality_weight" in s.metadata for s in samples_with_flags)
    # No deterministic weight changes applied at schedule time; weights remain 1.0
    np.testing.assert_allclose([s.weight for s in samples_with_flags], np.ones(len(samples_with_flags)))
    # Active axes metadata should exist for chain-aware weighting downstream.
    sample = next(s for s in samples_with_flags if s.metadata and s.metadata.get("active_axes"))
    assert sample.metadata["active_axes"]
