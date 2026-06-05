from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from smii.seams import CutTopologyReceipt, SolverPromotionReceipt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_seam_edges(path: Path) -> None:
    np.savez_compressed(path, seam_edges=np.array([[0, 1], [1, 2], [1, 3]], dtype=int))


def _write_solver_receipt(path: Path, seam_edges_path: Path, *, correction_hash: str | None = None) -> None:
    SolverPromotionReceipt(
        seam_cost_receipt_hash="seam-cost-receipt-sha256",
        solver_mode="metric_panelization",
        anchor_count=2,
        anchor_source="field_minima",
        connected_component_count=1,
        anchor_fallback_used=False,
        seam_edge_count=3,
        seam_vertex_count=4,
        total_seam_cost=1.0,
        panel_count=2,
        panels_are_disks=True,
        seam_hash=_sha256_file(seam_edges_path),
        promotion=1,
        blocked_consumers=[],
        correction_payload_hash=correction_hash,
    ).to_json(path)


def _write_cut_topology_receipt(
    path: Path,
    *,
    solver_receipt_path: Path,
    seam_edges_path: Path,
    typed_operator_count: int,
) -> None:
    CutTopologyReceipt(
        solver_receipt_hash=_sha256_file(solver_receipt_path),
        mesh_hash="mesh-sha256",
        seam_edges_hash=_sha256_file(seam_edges_path),
        seam_edge_segment_count=3,
        seam_vertex_count=4,
        seam_connected_component_count=1,
        seam_endpoint_count=3,
        seam_branch_vertex_count=1,
        panel_count=2,
        panel_face_counts=[1, 1],
        panel_boundary_edge_counts=[3, 3],
        panels_are_disks=True,
        typed_dart_count=typed_operator_count,
        typed_gusset_count=0,
        promotion=1,
        blocked_consumers=[],
        cut_topology_blockers=[],
        typed_operator_count=typed_operator_count,
        seam_graph_classifications=["typed_correction_operator"] if typed_operator_count else [],
    ).to_json(path)


def _write_corrections(path: Path, *, corrected_total: float = 0.01) -> None:
    path.write_text(
        json.dumps(
            {
                "energy": {
                    "raw_residual_total": 0.04,
                    "corrected_residual_total": corrected_total,
                },
                "selected_corrections": [
                    {
                        "family": "dart",
                        "panel_label": 0,
                        "reason": "local first-fundamental-form relaxation",
                        "raw_residual": 0.04,
                        "corrected_residual": corrected_total,
                        "correction_cost": 0.01,
                        "complexity_penalty": 0.0,
                        "manufacture_penalty": 0.0,
                        "gain": 0.03,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_emit(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, "scripts/emit_metric_correction_receipt.py", *args],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_emit_metric_correction_receipt_promotes_typed_operator(tmp_path: Path) -> None:
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    cut_topology_receipt_path = tmp_path / "cut_topology_receipt.json"
    corrections_path = tmp_path / "corrections.json"
    receipt_path = tmp_path / "metric_correction_receipt.json"

    _write_seam_edges(seam_edges_path)
    _write_corrections(corrections_path)
    _write_solver_receipt(
        solver_receipt_path,
        seam_edges_path,
        correction_hash=_sha256_file(corrections_path),
    )
    _write_cut_topology_receipt(
        cut_topology_receipt_path,
        solver_receipt_path=solver_receipt_path,
        seam_edges_path=seam_edges_path,
        typed_operator_count=1,
    )

    result = _run_emit(
        "--solver-receipt",
        str(solver_receipt_path),
        "--cut-topology-receipt",
        str(cut_topology_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--corrections",
        str(corrections_path),
        "--out-metric-correction-receipt",
        str(receipt_path),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["promotion"] == 1
    assert receipt["metric_correction_blockers"] == []
    assert receipt["correction_payload_hash"] == _sha256_file(corrections_path)


def test_emit_metric_correction_receipt_blocks_missing_typed_correction(tmp_path: Path) -> None:
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    cut_topology_receipt_path = tmp_path / "cut_topology_receipt.json"
    receipt_path = tmp_path / "metric_correction_receipt.json"

    _write_seam_edges(seam_edges_path)
    _write_solver_receipt(solver_receipt_path, seam_edges_path)
    _write_cut_topology_receipt(
        cut_topology_receipt_path,
        solver_receipt_path=solver_receipt_path,
        seam_edges_path=seam_edges_path,
        typed_operator_count=1,
    )

    result = _run_emit(
        "--solver-receipt",
        str(solver_receipt_path),
        "--cut-topology-receipt",
        str(cut_topology_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--out-metric-correction-receipt",
        str(receipt_path),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["promotion"] == 0
    assert "missingShapingIntentReceipt" in receipt["metric_correction_blockers"]
    assert "panel_unwrap" in receipt["blocked_consumers"]


def test_emit_metric_correction_receipt_blocks_residual_over_gate(tmp_path: Path) -> None:
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    cut_topology_receipt_path = tmp_path / "cut_topology_receipt.json"
    corrections_path = tmp_path / "corrections.json"
    receipt_path = tmp_path / "metric_correction_receipt.json"

    _write_seam_edges(seam_edges_path)
    _write_corrections(corrections_path, corrected_total=0.2)
    _write_solver_receipt(
        solver_receipt_path,
        seam_edges_path,
        correction_hash=_sha256_file(corrections_path),
    )
    _write_cut_topology_receipt(
        cut_topology_receipt_path,
        solver_receipt_path=solver_receipt_path,
        seam_edges_path=seam_edges_path,
        typed_operator_count=1,
    )

    result = _run_emit(
        "--solver-receipt",
        str(solver_receipt_path),
        "--cut-topology-receipt",
        str(cut_topology_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--corrections",
        str(corrections_path),
        "--out-metric-correction-receipt",
        str(receipt_path),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["promotion"] == 0
    assert "correctionResidualExceedsGate" in receipt["metric_correction_blockers"]
