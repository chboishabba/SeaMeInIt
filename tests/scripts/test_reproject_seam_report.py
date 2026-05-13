from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np


def test_reproject_seam_report_emits_rejected_correspondence_receipt(tmp_path) -> None:
    source_mesh = tmp_path / "source.npz"
    target_mesh = tmp_path / "target.npz"
    vertex_map = tmp_path / "vertex_map.npz"
    report_path = tmp_path / "seam_report.json"
    out_report = tmp_path / "reprojected.json"
    out_receipt = tmp_path / "correspondence_receipt.json"

    np.savez(
        source_mesh,
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )
    np.savez(
        target_mesh,
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )
    np.savez(
        vertex_map,
        source_to_target_indices=np.array([0, 0, 0, 1], dtype=np.int64),
        source_to_target_distances=np.array([0.49, 0.49, 0.49, 0.494], dtype=float),
        meta={
            "source_to_target_collision_ratio": 0.9842,
            "source_to_target_mean_distance": 0.491,
        },
    )
    report_path.write_text(
        json.dumps({"panels": {"front": {"edges": [[0, 1], [2, 3]]}}}),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/reproject_seam_report.py",
            "--report",
            str(report_path),
            "--source-mesh",
            str(source_mesh),
            "--target-mesh",
            str(target_mesh),
            "--vertex-map-file",
            str(vertex_map),
            "--out",
            str(out_report),
            "--out-correspondence-receipt",
            str(out_receipt),
        ],
        check=True,
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
    )

    receipt = json.loads(out_receipt.read_text(encoding="utf-8"))
    reprojection = json.loads(out_report.read_text(encoding="utf-8"))["reprojection"]

    assert "Wrote correspondence receipt" in result.stdout
    assert receipt["promotion"] == -1
    assert receipt["collision_ratio"] == 0.9842
    assert receipt["seam_transfer_collapse"] == reprojection["target_vertex_collision_ratio"]
    assert receipt["mean_distance"] == 0.491
    assert "solver_promotion" in receipt["blocked_consumers"]
    assert "Strategy B diagnostic only" in receipt["notes"][0]

