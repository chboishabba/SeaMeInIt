from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from smii.rom.basis_receipt import BasisReceipt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_basis(path: Path) -> None:
    basis = np.eye(4, dtype=float)
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    np.savez(path, basis=basis, vertices=vertices, meta={"source_mesh": "test"})


def _write_basis_receipt(path: Path, *, promotion: int = 1) -> None:
    BasisReceipt(
        carrier_receipt_hash="carrier-sha256",
        basis_vertex_count=4,
        basis_dimension=4,
        construction_method="sinusoidal_qr_h1",
        reconstruction_error=0.0,
        promotion=promotion,
        blocked_consumers=[],
        basis_hash="basis-sha256",
        promotion_threshold=0.05,
    ).to_json(path)


def _write_samples(path: Path, *, synthetic: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "synthetic": synthetic,
                    "pose_source": "synthetic_smplx_sweep"
                    if synthetic
                    else "rom_corpus_aggregated",
                },
                "samples": [
                    {
                        "pose_id": "neutral",
                        "coeffs": {
                            "pressure": [0.0, 1.0, 0.2, 0.1],
                            "shear": [0.0, 0.5, 0.1, 0.0],
                        },
                    },
                    {
                        "pose_id": "squat",
                        "coeffs": {
                            "pressure": [0.0, 0.7, 0.4, 0.2],
                            "shear": [0.1, 0.2, 0.3, 0.0],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _run_aggregation(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, "examples/rom_aggregate_from_samples.py", *args],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_rom_aggregation_emits_diagnostic_synthetic_receipt(tmp_path: Path) -> None:
    basis_path = tmp_path / "basis.npz"
    basis_receipt_path = tmp_path / "basis_receipt.json"
    samples_path = tmp_path / "samples.json"
    output_dir = tmp_path / "rom"
    fields_path = output_dir / "rom_fields.npz"
    receipt_path = output_dir / "rom_field_receipt.json"

    _write_basis(basis_path)
    _write_basis_receipt(basis_receipt_path)
    _write_samples(samples_path, synthetic=True)

    result = _run_aggregation(
        "--samples",
        str(samples_path),
        "--basis",
        str(basis_path),
        "--basis-receipt",
        str(basis_receipt_path),
        "--output-dir",
        str(output_dir),
        "--out-rom-fields",
        str(fields_path),
        "--out-rom-field-receipt",
        str(receipt_path),
        "--save-costs",
        str(output_dir / "seam_costs.npz"),
        "--field",
        "pressure",
    )

    assert result.returncode == 0, result.stderr
    assert "Wrote ROM field receipt" in result.stdout

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["basis_receipt_hash"] == _sha256_file(basis_receipt_path)
    assert receipt["samples_hash"] == _sha256_file(samples_path)
    assert receipt["fields_hash"] == _sha256_file(fields_path)
    assert receipt["pose_count"] == 2
    assert receipt["total_samples"] == 2
    assert receipt["pose_source"] == "synthetic_smplx_sweep"
    assert receipt["fields_computed"] == ["pressure", "shear"]
    assert receipt["vertex_count"] == 4
    assert receipt["field_uniformity"] < 0.95
    assert receipt["synthetic"] is True
    assert receipt["promotion"] == 0
    assert "seam_cost_field" in receipt["blocked_consumers"]

    fields = np.load(fields_path)
    assert "pressure_mean" in fields
    assert "pressure_peak" in fields


def test_rom_aggregation_can_explicitly_promote_synthetic_receipt(
    tmp_path: Path,
) -> None:
    basis_path = tmp_path / "basis.npz"
    basis_receipt_path = tmp_path / "basis_receipt.json"
    samples_path = tmp_path / "samples.json"
    receipt_path = tmp_path / "rom_field_receipt.json"

    _write_basis(basis_path)
    _write_basis_receipt(basis_receipt_path)
    _write_samples(samples_path, synthetic=True)

    result = _run_aggregation(
        "--samples",
        str(samples_path),
        "--basis",
        str(basis_path),
        "--basis-receipt",
        str(basis_receipt_path),
        "--output-dir",
        str(tmp_path / "rom"),
        "--out-rom-field-receipt",
        str(receipt_path),
        "--allow-synthetic-promotion",
        "--field",
        "pressure",
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["promotion"] == 1
    assert receipt["blocked_consumers"] == []


def test_rom_aggregation_rejects_unpromoted_basis_before_outputs(
    tmp_path: Path,
) -> None:
    basis_path = tmp_path / "basis.npz"
    basis_receipt_path = tmp_path / "basis_receipt.json"
    samples_path = tmp_path / "samples.json"
    output_dir = tmp_path / "rom"

    _write_basis(basis_path)
    _write_basis_receipt(basis_receipt_path, promotion=0)
    _write_samples(samples_path, synthetic=False)

    result = _run_aggregation(
        "--samples",
        str(samples_path),
        "--basis",
        str(basis_path),
        "--basis-receipt",
        str(basis_receipt_path),
        "--output-dir",
        str(output_dir),
        "--out-rom-field-receipt",
        str(output_dir / "rom_field_receipt.json"),
        "--field",
        "pressure",
    )

    assert result.returncode != 0
    assert "BasisReceipt not promoted" in result.stderr
    assert not output_dir.exists()

