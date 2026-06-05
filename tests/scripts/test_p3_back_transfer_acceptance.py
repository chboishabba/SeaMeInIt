from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_p3_back_transfer_receipt_labels_approximate_transfer(tmp_path: Path) -> None:
    source_mesh = tmp_path / "source_b_v4.npz"
    target_mesh = tmp_path / "target_a_v4.npz"
    body_receipt = tmp_path / "body_carrier_receipt.json"
    out_receipt = tmp_path / "back_transfer_receipt.json"
    out_map = tmp_path / "vertex_map.npz"

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
        ],
        dtype=int,
    )
    np.savez(source_mesh, vertices=vertices, faces=faces)
    np.savez(target_mesh, vertices=vertices, faces=faces)
    body_receipt.write_text(
        json.dumps(
            {
                "source_hash": "0" * 64,
                "raw_reprojection_hash": "1" * 64,
                "refined_pre_repair_hash": "2" * 64,
                "repaired_export_hash": _sha256_file(target_mesh),
                "vertex_count": 4,
                "face_count": 2,
                "topology_label": "A_v4",
                "landmark_residuals": {"measurement_fit_residual": 0.0},
                "skull_rigidity_residual": 0.0,
                "body_fit_confidence": 1.0,
                "promotion": 1,
                "blocked_consumers": [],
            }
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/p3_back_transfer_acceptance.py",
            "--source-mesh",
            str(source_mesh),
            "--target-mesh",
            str(target_mesh),
            "--target-body-receipt",
            str(body_receipt),
            "--out-receipt",
            str(out_receipt),
            "--out-map",
            str(out_map),
            "--source-topology-label",
            "B_v4",
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    receipt = json.loads(out_receipt.read_text(encoding="utf-8"))
    assert "Wrote P3 back-transfer receipt" in result.stdout
    assert receipt["promotion"] == 1
    assert receipt["source_topology_label"] == "B_v4"
    assert receipt["target_topology_label"] == "A_v4"
    assert receipt["forward_object_hash"] == _sha256_file(source_mesh)
    assert receipt["target_body_receipt_hash"] == _sha256_file(body_receipt)
    assert receipt["transfer_mode"] == "approximate_correspondence"
    assert receipt["approximate_transfer"] is True
    assert "not a geometric inverse" in receipt["notes"][1]
    assert out_map.exists()
