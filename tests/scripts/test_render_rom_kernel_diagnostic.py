import importlib.util
from pathlib import Path

import numpy as np

from smii.rom.sampler_real import ParameterBlock, ParameterLayout, PoseSample


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyBackend:
    def evaluate(self, pose):
        body_pose = np.asarray(pose["body_pose"], dtype=float).reshape(-1)
        base = np.zeros((2, 3), dtype=float)
        base += body_pose[0] * np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        base += body_pose[1] * np.array([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
        return base


def test_compute_pose_kernel_fields_matches_expected_directional_sensitivity():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "render_rom_kernel_diagnostic.py"
    module = _load_script_module(script_path, "render_rom_kernel_diagnostic")

    layout = ParameterLayout(blocks=(ParameterBlock("body_pose", 2, 0),), total=2)
    neutral = {"body_pose": np.zeros(2, dtype=float)}
    sample = PoseSample(
        pose_id="pose_a",
        parameters={"body_pose": np.array([1.0, 0.0], dtype=float)},
        weight=1.0,
    )
    weights = np.array([1.0, 3.0], dtype=float)

    fields = module.compute_pose_kernel_fields(
        DummyBackend(),
        neutral_pose=neutral,
        sample=sample,
        layout=layout,
        weights=weights,
        fd_step=1e-3,
        epsilon=1e-6,
    )

    np.testing.assert_allclose(fields["displacement_magnitude"], np.array([1.0, 1.0]), atol=1e-6)
    np.testing.assert_allclose(fields["derivative_magnitude"], np.array([13.0, 13.0]), atol=1e-4)
    np.testing.assert_allclose(fields["seam_sensitivity"], np.array([1.0, 1.0]), atol=1e-4)
