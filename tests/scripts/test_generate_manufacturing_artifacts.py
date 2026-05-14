from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from smii.seams import PanelUnwrapReceipt


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
