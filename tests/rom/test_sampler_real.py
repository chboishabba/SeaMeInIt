import numpy as np
import pytest

from smii.rom.sampler_real import (
    ParameterBlock,
    ParameterLayout,
    PoseSample,
    _load_smplx_parameter_payload,
    _expand_weight_block,
    _nearest_neighbor_map,
    _remap_costs,
    _save_vertex_correspondence,
    _stream_diagonal_costs,
)


class DummyBackend:
    """Deterministic linear backend used for FD smoke tests."""

    def __init__(self) -> None:
        self.base = np.zeros((2, 3), dtype=float)
        self.axes = (
            np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=float),
        )

    def evaluate(self, pose: dict[str, np.ndarray]) -> np.ndarray:
        body_pose = np.asarray(pose["body_pose"], dtype=float).reshape(-1)
        result = self.base.copy()
        for idx, basis in enumerate(self.axes):
            result += float(body_pose[idx]) * basis
        return result

    def evaluate_with_joints(
        self, pose: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        vertices = self.evaluate(pose)
        return vertices, {"pelvis": np.zeros(3, dtype=float)}


def test_stream_diagonal_costs_linear_backend():
    layout = ParameterLayout(blocks=(ParameterBlock("body_pose", 3, 0),), total=3)
    backend = DummyBackend()
    neutral = {"body_pose": np.zeros(3, dtype=float)}
    samples = [
        PoseSample(
            pose_id="pose_a", parameters={"body_pose": np.array([1.0, 0.0, 0.0])}, weight=1.0
        )
    ]
    weights = np.array([1.0, 0.0, 0.0], dtype=float)

    costs, call_counts, pose_stats, _, neutral_vertices = _stream_diagonal_costs(
        backend,
        neutral_pose=neutral,
        samples=samples,
        layout=layout,
        weights=weights,
        fd_step=1e-3,
        epsilon=1e-6,
    )

    assert costs.shape == (2,)
    np.testing.assert_allclose(costs, np.ones(2), rtol=1e-5, atol=1e-6)
    assert call_counts["total"] == 4  # neutral + pose + central diff calls
    assert pose_stats[0]["pose_id"] == "pose_a"
    assert pose_stats[0]["q_max"] == pytest.approx(1.0, rel=1e-5)
    np.testing.assert_allclose(neutral_vertices, np.zeros((2, 3)))


def test_expand_weight_block_overrides():
    joint_map = {"left_shoulder": 0, "right_shoulder": 3}
    weights = _expand_weight_block(
        {"default": 1.0, "overrides": {"right_shoulder": 2.0}},
        length=6,
        joint_map=joint_map,
    )
    expected = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(weights, expected)


def test_nearest_neighbor_map_round_trip():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    target = np.array([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=float)
    mapping, max_dist, mean_dist = _nearest_neighbor_map(source, target, batch_size=1)
    np.testing.assert_array_equal(mapping, np.array([0, 1], dtype=int))
    assert max_dist <= 0.1
    assert mean_dist <= 0.1


def test_remap_costs_errors_on_policy_error_mismatch():
    costs = np.array([1.0, 2.0])
    neutral = np.zeros((2, 3), dtype=float)
    target = np.zeros((3, 3), dtype=float)
    with pytest.raises(ValueError):
        _remap_costs(costs, neutral, target, policy="error", max_distance=0.1)


def test_remap_costs_warns_on_large_distance():
    costs = np.array([1.0, 0.5])
    neutral = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    target = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float)
    with pytest.warns(RuntimeWarning):
        remapped, info = _remap_costs(costs, neutral, target, policy="nearest", max_distance=0.5)
    assert remapped.shape[0] == target.shape[0]
    assert info["mode"] == "nearest_neighbor"
    assert info["max_distance"] >= 9.0


def test_save_vertex_correspondence_exports_bidirectional_map(tmp_path):
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    target = np.array([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    out = tmp_path / "correspondence.npz"

    meta = _save_vertex_correspondence(
        out,
        source_vertices=source,
        target_vertices=target,
        source_label="source",
        target_label="target",
        policy="nearest",
        max_distance=0.5,
    )

    payload = np.load(out, allow_pickle=True)
    source_to_target = payload["source_to_target_indices"]
    target_to_source = payload["target_to_source_indices"]
    assert source_to_target.shape[0] == source.shape[0]
    assert target_to_source.shape[0] == target.shape[0]
    assert meta["source_vertex_count"] == source.shape[0]
    assert meta["target_vertex_count"] == target.shape[0]


def test_load_smplx_parameter_payload_decodes_parameter_list():
    payload = {
        "model_type": "smplx",
        "gender": "neutral",
        "scale": 1.23,
        "parameters": ["betas", "body_pose"],
        "betas": [0.1, 0.2, 0.3],
        "body_pose": [[0.0, 0.1, 0.2]],
    }
    params, scale, model_type, gender = _load_smplx_parameter_payload(payload)
    assert scale == pytest.approx(1.23)
    assert model_type == "smplx"
    assert gender == "neutral"
    assert tuple(params.keys()) == ("betas", "body_pose")
    assert params["betas"].shape == (1, 3)
    assert params["body_pose"].shape == (1, 3)
