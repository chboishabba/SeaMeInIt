from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from smii.rom import SeamCostField, save_seam_cost_field
from smii.seams import SeamCostReceipt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_mesh(path: Path) -> tuple[tuple[int, int], ...]:
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
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=int,
    )
    np.savez(path, vertices=vertices, faces=faces)
    return ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _write_costs(path: Path, edges: tuple[tuple[int, int], ...]) -> None:
    save_seam_cost_field(
        SeamCostField(
            field="combined",
            vertex_costs=np.array([0.1, 0.2, 1.0, 0.3], dtype=float),
            edge_costs=np.array([0.1, 0.3, 0.5, 0.8, 0.4, 0.7], dtype=float),
            edges=edges,
            samples_used=2,
            metadata={"cost_uniformity": 0.4},
        ),
        path,
    )


def _write_seam_cost_receipt(
    path: Path,
    costs_path: Path,
    *,
    promotion: int = 1,
) -> None:
    SeamCostReceipt(
        rom_field_receipt_hash="rom-field-receipt-sha256",
        body_receipt_hash="body-receipt-sha256",
        correspondence_receipt_hash=None,
        solve_domain="A_v3240",
        vertex_count=4,
        edge_count=6,
        finite_cost_coverage=1.0,
        cost_uniformity=0.4,
        peak_cost=0.8,
        mean_cost=0.45,
        weight_vector={"w_P": 1.0},
        costs_hash=_sha256_file(costs_path),
        promotion=promotion,
        blocked_consumers=[],
    ).to_json(path)


def _run_solve(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, "scripts/solve_seams.py", *args],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_solve_seams_emits_promoted_receipt_from_promoted_costs(tmp_path: Path) -> None:
    mesh_path = tmp_path / "body.npz"
    costs_path = tmp_path / "seam_costs.npz"
    receipt_path = tmp_path / "seam_cost_receipt.json"
    out_dir = tmp_path / "out"
    solver_receipt_path = out_dir / "solver_promotion_receipt.json"

    edges = _write_mesh(mesh_path)
    _write_costs(costs_path, edges)
    _write_seam_cost_receipt(receipt_path, costs_path)

    result = _run_solve(
        "--seam-cost-receipt",
        str(receipt_path),
        "--costs",
        str(costs_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-solver-receipt",
        str(solver_receipt_path),
        "--anchor-count",
        "3",
        "--min-geodesic-separation",
        "0",
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(solver_receipt_path.read_text(encoding="utf-8"))
    assert receipt["seam_cost_receipt_hash"] == _sha256_file(receipt_path)
    assert receipt["solver_mode"] == "shortest_path"
    assert receipt["anchor_source"] == "field_minima"
    assert receipt["seam_edge_count"] > 0
    assert receipt["seam_vertex_count"] > 0
    assert receipt["panels_are_disks"]
    assert receipt["promotion"] == 1

    seam_edges_path = out_dir / "seam_edges.npz"
    assert receipt["seam_hash"] == _sha256_file(seam_edges_path)
    assert "seam_edges" in np.load(seam_edges_path)


def test_solve_seams_blocks_unpromoted_seam_cost_receipt(tmp_path: Path) -> None:
    mesh_path = tmp_path / "body.npz"
    costs_path = tmp_path / "seam_costs.npz"
    receipt_path = tmp_path / "seam_cost_receipt.json"
    out_dir = tmp_path / "out"

    edges = _write_mesh(mesh_path)
    _write_costs(costs_path, edges)
    _write_seam_cost_receipt(receipt_path, costs_path, promotion=0)

    result = _run_solve(
        "--seam-cost-receipt",
        str(receipt_path),
        "--costs",
        str(costs_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
    )

    assert result.returncode != 0
    assert "SeamCostReceipt not promoted" in result.stderr
    assert not out_dir.exists()


def test_solve_seams_rejects_cost_hash_mismatch(tmp_path: Path) -> None:
    mesh_path = tmp_path / "body.npz"
    costs_path = tmp_path / "seam_costs.npz"
    receipt_path = tmp_path / "seam_cost_receipt.json"

    edges = _write_mesh(mesh_path)
    _write_costs(costs_path, edges)
    _write_seam_cost_receipt(receipt_path, costs_path)
    _write_costs(costs_path, tuple(reversed(edges)))

    result = _run_solve(
        "--seam-cost-receipt",
        str(receipt_path),
        "--costs",
        str(costs_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(tmp_path / "out"),
    )

    assert result.returncode != 0
    assert "Seam costs hash does not match" in result.stderr
