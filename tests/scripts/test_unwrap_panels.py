from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from smii.seams import SolverPromotionReceipt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_mesh(path: Path) -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    np.savez(path, vertices=vertices, faces=faces)


def _write_seam_edges(path: Path) -> None:
    np.savez_compressed(path, seam_edges=np.empty((0, 2), dtype=int))


def _write_solver_receipt(
    path: Path,
    seam_edges_path: Path,
    *,
    promotion: int = 1,
    panels_are_disks: bool = True,
) -> None:
    SolverPromotionReceipt(
        seam_cost_receipt_hash="seam-cost-receipt-sha256",
        solver_mode="shortest_path",
        anchor_count=2,
        anchor_source="field_minima",
        connected_component_count=1,
        anchor_fallback_used=False,
        seam_edge_count=0,
        seam_vertex_count=0,
        total_seam_cost=0.0,
        panel_count=2,
        panels_are_disks=panels_are_disks,
        seam_hash=_sha256_file(seam_edges_path),
        promotion=promotion,
        blocked_consumers=[],
    ).to_json(path)


def _run_unwrap(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, "scripts/unwrap_panels.py", *args],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_unwrap_panels_emits_promoted_receipt_from_promoted_solver(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    out_dir = tmp_path / "out"
    panel_receipt_path = out_dir / "panel_unwrap_receipt.json"

    _write_mesh(mesh_path)
    _write_seam_edges(seam_edges_path)
    _write_solver_receipt(solver_receipt_path, seam_edges_path)

    result = _run_unwrap(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-panel-receipt",
        str(panel_receipt_path),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(panel_receipt_path.read_text(encoding="utf-8"))
    assert receipt["solver_receipt_hash"] == _sha256_file(solver_receipt_path)
    assert receipt["panel_count"] == 2
    assert receipt["panels_all_disks"]
    assert receipt["promotion"] == 1
    assert receipt["seam_topology_hash"] == _sha256_file(seam_edges_path)

    uv_path = out_dir / "panel_uvs.npz"
    assert receipt["uv_hash"] == _sha256_file(uv_path)
    payload = np.load(uv_path)
    assert set(payload.files) == {"panel_0", "panel_1"}


def test_unwrap_panels_blocks_unpromoted_solver_receipt(tmp_path: Path) -> None:
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"

    _write_mesh(mesh_path)
    _write_seam_edges(seam_edges_path)
    _write_solver_receipt(solver_receipt_path, seam_edges_path, promotion=0)

    result = _run_unwrap(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(tmp_path / "out"),
    )

    assert result.returncode != 0
    assert "SolverPromotionReceipt not promoted" in result.stderr


def test_unwrap_panels_reports_topology_not_unwrapper_when_not_disks(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"

    _write_mesh(mesh_path)
    _write_seam_edges(seam_edges_path)
    _write_solver_receipt(
        solver_receipt_path,
        seam_edges_path,
        panels_are_disks=False,
    )

    result = _run_unwrap(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(tmp_path / "out"),
    )

    assert result.returncode != 0
    assert "seam topology is incomplete" in result.stderr
    assert "unwrapper is not the problem" in result.stderr
