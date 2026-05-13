from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from smii.meshing.body_carrier_receipt import BodyCarrierReceipt
from smii.meshing.correspondence_receipt import CorrespondenceReceipt
from smii.rom.rom_field_receipt import ROMFieldReceipt


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


def _write_body_receipt(path: Path, *, promotion: int = 1) -> None:
    BodyCarrierReceipt(
        source_hash="source",
        raw_reprojection_hash="raw",
        refined_pre_repair_hash="refined",
        repaired_export_hash="repaired",
        vertex_count=4,
        face_count=4,
        topology_label="A_v3240",
        landmark_residuals={"nose": 0.1},
        skull_rigidity_residual=0.1,
        body_fit_confidence=0.9,
        promotion=promotion,
        blocked_consumers=[],
    ).to_json(path)


def _write_rom_fields(path: Path, *, flat: bool = False) -> None:
    pressure = np.ones(4, dtype=float) if flat else np.array([0.0, 2.0, 0.2, 0.4])
    np.savez_compressed(
        path,
        pressure_peak=pressure,
        shear_peak=np.array([0.0, 0.4, 0.1, 0.2]),
        tension_peak=np.array([0.0, 0.1, 0.7, 0.2]),
        thermal_peak=np.array([0.3, 0.1, 0.4, 0.0]),
        cooling_peak=np.array([0.0, 0.2, 0.8, 0.1]),
    )


def _write_rom_receipt(path: Path, fields_path: Path, *, promotion: int = 1) -> None:
    ROMFieldReceipt(
        basis_receipt_hash="basis",
        samples_hash="samples",
        aggregation_summary_hash="summary",
        fields_hash=_sha256_file(fields_path),
        pose_count=2,
        total_samples=2,
        pose_source="rom_corpus_aggregated",
        fields_computed=["pressure", "shear", "tension", "thermal", "cooling"],
        vertex_count=4,
        peak_pressure_max=2.0,
        peak_pressure_percentile95=1.8,
        field_uniformity=0.4,
        synthetic=False,
        promotion=promotion,
        blocked_consumers=[],
    ).to_json(path)


def _write_correspondence(path: Path, *, promotion: int = 1) -> None:
    CorrespondenceReceipt(
        source_mesh_hash="source-mesh",
        target_mesh_hash="target-mesh",
        transform_type="barycentric",
        mean_distance=0.001,
        max_distance=0.002,
        collision_ratio=0.1,
        seam_transfer_collapse=0.1,
        retention_ratio=0.99,
        unique_targets_used=4,
        total_target_vertices=4,
        edge_retention_ratio=0.98,
        promotion=promotion,
        notes=[],
        blocked_consumers=[],
    ).to_json(path)


def _run_compute(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, "scripts/compute_seam_costs.py", *args],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_compute_seam_costs_emits_promoted_receipt_for_native_domain(tmp_path: Path) -> None:
    mesh_path = tmp_path / "body.npz"
    fields_path = tmp_path / "rom_fields.npz"
    body_receipt_path = tmp_path / "body_carrier_receipt.json"
    rom_receipt_path = tmp_path / "rom_field_receipt.json"
    costs_path = tmp_path / "seam_costs.npz"
    receipt_path = tmp_path / "seam_cost_receipt.json"

    _write_mesh(mesh_path)
    _write_rom_fields(fields_path)
    _write_body_receipt(body_receipt_path)
    _write_rom_receipt(rom_receipt_path, fields_path)

    result = _run_compute(
        "--body-receipt",
        str(body_receipt_path),
        "--rom-field-receipt",
        str(rom_receipt_path),
        "--rom-fields",
        str(fields_path),
        "--mesh",
        str(mesh_path),
        "--out-costs",
        str(costs_path),
        "--out-seam-cost-receipt",
        str(receipt_path),
        "--solve-domain",
        "A_v3240",
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["rom_field_receipt_hash"] == _sha256_file(rom_receipt_path)
    assert receipt["body_receipt_hash"] == _sha256_file(body_receipt_path)
    assert receipt["correspondence_receipt_hash"] is None
    assert receipt["solve_domain"] == "A_v3240"
    assert receipt["finite_cost_coverage"] == 1.0
    assert receipt["cost_uniformity"] < 0.95
    assert receipt["costs_hash"] == _sha256_file(costs_path)
    assert receipt["promotion"] == 1

    costs = np.load(costs_path, allow_pickle=True)
    assert "edge_costs" in costs
    assert len(costs["edge_costs"]) == receipt["edge_count"]


def test_compute_seam_costs_blocks_unpromoted_rom_field(tmp_path: Path) -> None:
    mesh_path = tmp_path / "body.npz"
    fields_path = tmp_path / "rom_fields.npz"
    body_receipt_path = tmp_path / "body_carrier_receipt.json"
    rom_receipt_path = tmp_path / "rom_field_receipt.json"
    output_dir = tmp_path / "out"

    _write_mesh(mesh_path)
    _write_rom_fields(fields_path)
    _write_body_receipt(body_receipt_path)
    _write_rom_receipt(rom_receipt_path, fields_path, promotion=0)

    result = _run_compute(
        "--body-receipt",
        str(body_receipt_path),
        "--rom-field-receipt",
        str(rom_receipt_path),
        "--rom-fields",
        str(fields_path),
        "--mesh",
        str(mesh_path),
        "--out-costs",
        str(output_dir / "seam_costs.npz"),
        "--out-seam-cost-receipt",
        str(output_dir / "seam_cost_receipt.json"),
    )

    assert result.returncode != 0
    assert "ROMFieldReceipt not promoted" in result.stderr
    assert not output_dir.exists()


def test_compute_seam_costs_requires_promoted_correspondence_for_transfer(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "body.npz"
    fields_path = tmp_path / "rom_fields.npz"
    body_receipt_path = tmp_path / "body_carrier_receipt.json"
    rom_receipt_path = tmp_path / "rom_field_receipt.json"
    corr_receipt_path = tmp_path / "correspondence_receipt.json"

    _write_mesh(mesh_path)
    _write_rom_fields(fields_path)
    _write_body_receipt(body_receipt_path)
    _write_rom_receipt(rom_receipt_path, fields_path)
    _write_correspondence(corr_receipt_path, promotion=0)

    result = _run_compute(
        "--body-receipt",
        str(body_receipt_path),
        "--rom-field-receipt",
        str(rom_receipt_path),
        "--rom-fields",
        str(fields_path),
        "--mesh",
        str(mesh_path),
        "--correspondence-receipt",
        str(corr_receipt_path),
        "--solve-domain",
        "B_v9438",
    )

    assert result.returncode != 0
    assert "CorrespondenceReceipt not promoted" in result.stderr
