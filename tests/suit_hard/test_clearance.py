from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from suit_hard.clearance import Mesh, analyze_clearance, interpolate_poses
from smii.pipelines.analyze_clearance import run_clearance


def _box_mesh(size: float) -> Mesh:
    half = size / 2.0
    vertices = np.array(
        [
            (-half, -half, -half),
            (half, -half, -half),
            (half, half, -half),
            (-half, half, -half),
            (-half, -half, half),
            (half, -half, half),
            (half, half, half),
            (-half, half, half),
        ],
        dtype=float,
    )
    faces = np.array(
        [
            (0, 1, 2),
            (0, 2, 3),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ],
        dtype=int,
    )
    return Mesh(vertices=vertices, faces=faces)


def _translation(tx: float, ty: float, tz: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = (tx, ty, tz)
    return transform


def test_analyze_clearance_detects_penetration() -> None:
    shell = _box_mesh(2.0)
    target = _box_mesh(1.0)
    result = analyze_clearance(shell, target, [np.eye(4)])
    assert pytest.approx(result.worst_penetration, rel=1e-5) == 0.5
    pose = result.poses[0]
    assert pose.contacts, "Expected contacts for enclosed target"
    assert pytest.approx(pose.min_clearance, rel=1e-5) == -0.5


def test_interpolate_poses_inserts_samples() -> None:
    identity = np.eye(4)
    shifted = _translation(1.0, 0.0, 0.0)
    samples = interpolate_poses([identity, shifted], samples_per_segment=2)
    assert len(samples) == 4
    mid = samples[1]
    assert np.isclose(mid[0, 3], 1.0 / 3.0)


def test_run_clearance_produces_reports(tmp_path: Path) -> None:
    shell = _box_mesh(2.0)
    target = _box_mesh(1.0)
    shell_path = tmp_path / "shell.npz"
    target_path = tmp_path / "target.npz"
    np.savez(shell_path, vertices=shell.vertices, faces=shell.faces)
    np.savez(target_path, vertices=target.vertices, faces=target.faces)

    pose_path = tmp_path / "poses.json"
    transforms = [np.eye(4).tolist(), _translation(2.0, 0.0, 0.0).tolist()]
    with pose_path.open("w", encoding="utf-8") as stream:
        json.dump(transforms, stream)

    output_dir = tmp_path / "reports"
    result_dir = run_clearance(
        shell_path,
        target_path,
        pose_path=pose_path,
        output_dir=output_dir,
        samples_per_segment=1,
    )

    assert result_dir == output_dir
    report_path = output_dir / "report.json"
    table_path = output_dir / "poses.csv"
    text_path = output_dir / "report.txt"
    for path in (report_path, table_path, text_path):
        assert path.exists()

    with report_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    assert payload["worst_penetration"] > 0
    assert payload["poses"][0]["max_penetration"] > 0
    assert payload["poses"][-1]["min_clearance"] > 0
    assert len(payload["poses"]) == 3  # two key poses plus one interpolated sample
