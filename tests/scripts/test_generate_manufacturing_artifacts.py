from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from smii.seams import PanelUnwrapReceipt
from smii.seams.cut_topology_receipt import CutTopologyReceipt
from smii.seams.metric_correction_receipt import MetricCorrectionEntry, MetricCorrectionReceipt
from smii.seams.seam_cost_receipt import SeamCostReceipt
from smii.seams.solver_promotion_receipt import SolverPromotionReceipt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_panel_uvs(path: Path) -> None:
    np.savez_compressed(
        path,
        panel_0=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=float),
        panel_1=np.array([[0.0, 0.0], [0.8, 0.0], [0.4, 0.6]], dtype=float),
    )


def _write_rom_fields(path: Path) -> None:
    np.savez_compressed(
        path,
        pressure_peak=np.array([0.0, 0.1, 0.6, 1.6, 3.2, 5.0], dtype=float),
        shear_peak=np.array([0.0, 0.4, 0.5, 1.0, 2.0, 2.2], dtype=float),
    )


def _write_panel_receipt(
    path: Path,
    panel_uvs_path: Path,
    *,
    promotion: int = 1,
) -> None:
    PanelUnwrapReceipt(
        solver_receipt_hash="solver-receipt-sha256",
        panel_count=2,
        panels_all_disks=True,
        per_panel_distortion=[0.01, 0.02],
        worst_panel_distortion=0.02,
        mean_panel_distortion=0.015,
        distortion_threshold=0.05,
        subdivision_iterations=0,
        grain_directions=["warp", "bias"],
        uv_hash=_sha256_file(panel_uvs_path),
        seam_topology_hash="seam-sha256",
        promotion=promotion,
        blocked_consumers=[],
    ).to_json(path)


def _write_upstream_receipts(tmp_path: Path) -> dict[str, Path]:
    seam_cost_path = tmp_path / "seam_cost_receipt.json"
    solver_path = tmp_path / "solver_promotion_receipt.json"
    cut_topology_path = tmp_path / "cut_topology_receipt.json"
    metric_correction_path = tmp_path / "metric_correction_receipt.json"

    SeamCostReceipt(
        rom_field_receipt_hash="rom-field-sha256",
        body_receipt_hash="body-sha256",
        correspondence_receipt_hash=None,
        solve_domain="A_v3240",
        vertex_count=4,
        edge_count=5,
        finite_cost_coverage=1.0,
        cost_uniformity=0.4,
        peak_cost=2.0,
        mean_cost=1.0,
        weight_vector={"w_P": 1.0, "w_S": 0.8},
        costs_hash="costs-sha256",
        promotion=1,
        blocked_consumers=[],
    ).to_json(seam_cost_path)
    SolverPromotionReceipt(
        seam_cost_receipt_hash=_sha256_file(seam_cost_path),
        solver_mode="shortest_path",
        anchor_count=4,
        anchor_source="field_minima",
        connected_component_count=1,
        anchor_fallback_used=False,
        seam_edge_count=3,
        seam_vertex_count=4,
        total_seam_cost=2.5,
        panel_count=2,
        panels_are_disks=True,
        seam_hash="seam-sha256",
        promotion=1,
        blocked_consumers=[],
    ).to_json(solver_path)
    CutTopologyReceipt(
        solver_receipt_hash=_sha256_file(solver_path),
        mesh_hash="mesh-sha256",
        seam_edges_hash="seam-edges-sha256",
        seam_edge_segment_count=8,
        seam_vertex_count=8,
        seam_connected_component_count=1,
        seam_endpoint_count=0,
        seam_branch_vertex_count=0,
        panel_count=2,
        panel_face_counts=[12, 12],
        panel_boundary_edge_counts=[4, 4],
        panels_are_disks=True,
        typed_dart_count=0,
        typed_gusset_count=0,
        promotion=1,
        blocked_consumers=[],
        cut_topology_blockers=[],
    ).to_json(cut_topology_path)
    MetricCorrectionReceipt(
        solver_receipt_hash=_sha256_file(solver_path),
        cut_topology_receipt_hash=_sha256_file(cut_topology_path),
        seam_edges_hash="seam-edges-sha256",
        panels_requiring_correction=[0],
        corrections=[
            MetricCorrectionEntry(
                panel_label=0,
                correction_type="dart",
                delta_metric_meaning="local first-fundamental-form relaxation",
                raw_residual=0.08,
                corrected_residual=0.02,
                energy_terms={"shape": 0.02},
                result_state="correctionOk",
                blockers=[],
            )
        ],
        raw_residual_total=0.08,
        corrected_residual_total=0.02,
        residual_gate=0.05,
        promotion=1,
        blocked_consumers=[],
        metric_correction_blockers=[],
        correction_payload_hash="metric-correction-sha256",
    ).to_json(metric_correction_path)
    return {
        "seam_cost": seam_cost_path,
        "solver": solver_path,
        "cut_topology": cut_topology_path,
        "metric_correction": metric_correction_path,
    }


def _run_manufacture(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, "scripts/generate_manufacturing_artifacts.py", *args],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_generate_manufacturing_artifacts_emits_promoted_receipt(
    tmp_path: Path,
) -> None:
    panel_uvs_path = tmp_path / "panel_uvs.npz"
    rom_fields_path = tmp_path / "rom_fields.npz"
    panel_receipt_path = tmp_path / "panel_unwrap_receipt.json"
    out_dir = tmp_path / "out"
    manufacturing_receipt_path = out_dir / "manufacturing_receipt.json"

    _write_panel_uvs(panel_uvs_path)
    _write_rom_fields(rom_fields_path)
    _write_panel_receipt(panel_receipt_path, panel_uvs_path)

    result = _run_manufacture(
        "--panel-receipt",
        str(panel_receipt_path),
        "--panel-uvs",
        str(panel_uvs_path),
        "--rom-fields",
        str(rom_fields_path),
        "--out-dir",
        str(out_dir),
        "--out-manufacturing-receipt",
        str(manufacturing_receipt_path),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(manufacturing_receipt_path.read_text(encoding="utf-8"))
    assert receipt["panel_unwrap_receipt_hash"] == _sha256_file(panel_receipt_path)
    assert receipt["panel_count"] == 2
    assert receipt["manufacturing_method"] == "home_sewing"
    assert receipt["accessibility_level"] == "consumer"
    assert receipt["allowance_varies"]
    assert receipt["promotion"] == 1
    assert receipt["notches_present"]
    assert receipt["labels_present"]
    assert receipt["cutting_artifacts_hash"] == _sha256_file(out_dir / "cutting_layout.svg")
    assert receipt["seam_allowance_hash"] == _sha256_file(out_dir / "seam_allowance.npz")

    allowance = np.load(out_dir / "seam_allowance.npz")["allowance"]
    assert float(allowance.std()) > 1e-4


def test_generate_manufacturing_artifacts_emits_finished_seam_receipt(
    tmp_path: Path,
) -> None:
    panel_uvs_path = tmp_path / "panel_uvs.npz"
    rom_fields_path = tmp_path / "rom_fields.npz"
    panel_receipt_path = tmp_path / "panel_unwrap_receipt.json"
    out_dir = tmp_path / "out"
    manufacturing_receipt_path = out_dir / "manufacturing_receipt.json"
    finished_receipt_path = out_dir / "finished_seam_receipt.json"

    _write_panel_uvs(panel_uvs_path)
    _write_rom_fields(rom_fields_path)
    _write_panel_receipt(panel_receipt_path, panel_uvs_path)
    upstream = _write_upstream_receipts(tmp_path)

    result = _run_manufacture(
        "--panel-receipt",
        str(panel_receipt_path),
        "--panel-uvs",
        str(panel_uvs_path),
        "--rom-fields",
        str(rom_fields_path),
        "--out-dir",
        str(out_dir),
        "--out-manufacturing-receipt",
        str(manufacturing_receipt_path),
        "--out-finished-seam-receipt",
        str(finished_receipt_path),
        "--body-receipt-hash",
        "body-sha256",
        "--rom-receipt-hash",
        "rom-sha256",
        "--fabric-receipt-hash",
        "fabric-sha256",
        "--basis-receipt-hash",
        "basis-sha256",
        "--seam-cost-receipt",
        str(upstream["seam_cost"]),
        "--solver-receipt",
        str(upstream["solver"]),
        "--cut-topology-receipt",
        str(upstream["cut_topology"]),
        "--metric-correction-receipt",
        str(upstream["metric_correction"]),
    )

    assert result.returncode == 0, result.stderr
    assert manufacturing_receipt_path.exists()
    assert finished_receipt_path.exists()
    receipt = json.loads(finished_receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "smii.finished_seam_receipt.v1"
    assert receipt["promotion"] == 1
    assert receipt["body_gate"]["receipt_hash"] == "body-sha256"
    assert receipt["rom_gate"]["receipt_hash"] == "rom-sha256"
    assert receipt["fabric_gate"]["receipt_hash"] == "fabric-sha256"
    assert receipt["basis_gate"]["receipt_hash"] == "basis-sha256"
    assert receipt["seam_atlas"]["selected_seam_count"] == 3
    assert receipt["panel_atlas"]["panel_count"] == 2
    assert receipt["flattening"]["panel_unwrap_hash"] == _sha256_file(panel_uvs_path)
    assert {"type": "dart", "count": 1} in receipt["correction_ops"]
    assert receipt["allowance_fields"]["policy"] == "variable_boundary_field"
    assert receipt["manufacturing_exports"]["hash"] == _sha256_file(out_dir / "cutting_layout.svg")
    assert receipt["claim_boundary"]["export_is_geometry_truth"] is False
    assert receipt["claim_boundary"]["claims_true_inverse"] is False


def test_constant_allowance_is_named_diagnostic(
    tmp_path: Path,
) -> None:
    panel_uvs_path = tmp_path / "panel_uvs.npz"
    rom_fields_path = tmp_path / "rom_fields.npz"
    panel_receipt_path = tmp_path / "panel_unwrap_receipt.json"
    out_dir = tmp_path / "out"

    _write_panel_uvs(panel_uvs_path)
    _write_rom_fields(rom_fields_path)
    _write_panel_receipt(panel_receipt_path, panel_uvs_path)

    result = _run_manufacture(
        "--panel-receipt",
        str(panel_receipt_path),
        "--panel-uvs",
        str(panel_uvs_path),
        "--rom-fields",
        str(rom_fields_path),
        "--out-dir",
        str(out_dir),
        "--constant-allowance",
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads((out_dir / "manufacturing_receipt.json").read_text("utf-8"))
    assert not receipt["allowance_varies"]
    assert receipt["promotion"] == 0
    assert "allowance_varies=False" in receipt["notes"]


def test_generate_manufacturing_artifacts_blocks_unpromoted_panel_receipt(
    tmp_path: Path,
) -> None:
    panel_uvs_path = tmp_path / "panel_uvs.npz"
    rom_fields_path = tmp_path / "rom_fields.npz"
    panel_receipt_path = tmp_path / "panel_unwrap_receipt.json"

    _write_panel_uvs(panel_uvs_path)
    _write_rom_fields(rom_fields_path)
    _write_panel_receipt(panel_receipt_path, panel_uvs_path, promotion=0)

    result = _run_manufacture(
        "--panel-receipt",
        str(panel_receipt_path),
        "--panel-uvs",
        str(panel_uvs_path),
        "--rom-fields",
        str(rom_fields_path),
        "--out-dir",
        str(tmp_path / "out"),
    )

    assert result.returncode != 0
    assert "PanelUnwrapReceipt not promoted" in result.stderr


def test_generate_manufacturing_artifacts_rejects_uv_hash_mismatch(
    tmp_path: Path,
) -> None:
    panel_uvs_path = tmp_path / "panel_uvs.npz"
    rom_fields_path = tmp_path / "rom_fields.npz"
    panel_receipt_path = tmp_path / "panel_unwrap_receipt.json"

    _write_panel_uvs(panel_uvs_path)
    _write_rom_fields(rom_fields_path)
    _write_panel_receipt(panel_receipt_path, panel_uvs_path)
    np.savez_compressed(panel_uvs_path, panel_0=np.zeros((3, 2)), panel_1=np.ones((3, 2)))

    result = _run_manufacture(
        "--panel-receipt",
        str(panel_receipt_path),
        "--panel-uvs",
        str(panel_uvs_path),
        "--rom-fields",
        str(rom_fields_path),
        "--out-dir",
        str(tmp_path / "out"),
    )

    assert result.returncode != 0
    assert "Panel UV hash does not match" in result.stderr
