from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_cone_mesh(path: Path) -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 1.4],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [5, 2, 1],
            [5, 3, 2],
            [5, 4, 3],
            [5, 1, 4],
        ],
        dtype=int,
    )
    np.savez(path, vertices=vertices, faces=faces)


def _run_candidates(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, "scripts/propose_dart_relief_cuts.py", *args],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_propose_dart_relief_cuts_finds_cone_apex(tmp_path: Path) -> None:
    mesh_path = tmp_path / "cone.npz"
    out_json = tmp_path / "candidates.json"
    _write_cone_mesh(mesh_path)

    result = _run_candidates(
        "--mesh",
        str(mesh_path),
        "--out-json",
        str(out_json),
        "--max-candidates",
        "3",
        "--percentile",
        "80",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["diagnostic_only"] is True
    assert payload["curvature_metric"] == "absolute_angle_deficit"
    assert payload["candidate_count"] >= 1
    assert any(candidate["apex_vertex"] == 0 for candidate in payload["candidates"])
    assert {candidate["candidate_type"] for candidate in payload["candidates"]} <= {
        "dart",
        "relief_cut",
    }


def test_propose_dart_relief_cuts_types_path_to_existing_seam_as_dart(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "cone.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    out_json = tmp_path / "candidates.json"
    _write_cone_mesh(mesh_path)
    np.savez_compressed(seam_edges_path, seam_edges=np.array([[1, 2]], dtype=int))

    result = _run_candidates(
        "--mesh",
        str(mesh_path),
        "--seam-edges",
        str(seam_edges_path),
        "--out-json",
        str(out_json),
        "--max-candidates",
        "1",
        "--percentile",
        "80",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["candidate_count"] == 1
    candidate = payload["candidates"][0]
    assert candidate["candidate_type"] == "dart"
    assert candidate["endpoint_class"] == "existing_seam"
    assert candidate["path_edges"]
