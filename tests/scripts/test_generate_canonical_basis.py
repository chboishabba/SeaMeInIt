from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from smii.meshing.body_carrier_receipt import BodyCarrierReceipt


def _body_payload(*, promotion: int = 1, vertex_count: int = 4) -> dict[str, object]:
    return {
        "source_hash": "source-abc",
        "raw_reprojection_hash": "raw-def",
        "refined_pre_repair_hash": "refined-ghi",
        "repaired_export_hash": "repaired-jkl",
        "vertex_count": vertex_count,
        "face_count": 2,
        "topology_label": f"A_v{vertex_count}",
        "landmark_residuals": {"nose": 0.8},
        "skull_rigidity_residual": 0.03,
        "body_fit_confidence": 0.91,
        "promotion": promotion,
        "blocked_consumers": [],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_vertices(path: Path) -> None:
    np.savez(
        path,
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
    )


def _run_basis_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, "scripts/generate_canonical_basis.py", *args],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_generate_canonical_basis_emits_promoted_basis_receipt(tmp_path: Path) -> None:
    vertices_path = tmp_path / "body.npz"
    basis_path = tmp_path / "canonical_basis.npz"
    body_receipt_path = tmp_path / "body_carrier_receipt.json"
    receipt_path = tmp_path / "basis_receipt.json"

    _write_vertices(vertices_path)
    BodyCarrierReceipt.from_mapping(_body_payload()).to_json(body_receipt_path)

    result = _run_basis_script(
        "--vertices",
        str(vertices_path),
        "--body-receipt",
        str(body_receipt_path),
        "--components",
        "4",
        "--harmonics",
        "1",
        "--output",
        str(basis_path),
        "--receipt-output",
        str(receipt_path),
    )

    assert result.returncode == 0, result.stderr
    assert "Wrote basis receipt" in result.stdout

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["carrier_receipt_hash"] == _sha256_file(body_receipt_path)
    assert receipt["basis_hash"] == _sha256_file(basis_path)
    assert receipt["basis_vertex_count"] == 4
    assert receipt["basis_dimension"] == 4
    assert receipt["construction_method"] == "sinusoidal_qr_h1"
    assert receipt["promotion_threshold"] == 0.05
    assert receipt["promotion"] == 1
    assert receipt["blocked_consumers"] == []


def test_generate_canonical_basis_rejects_unpromoted_body_receipt(
    tmp_path: Path,
) -> None:
    vertices_path = tmp_path / "body.npz"
    basis_path = tmp_path / "canonical_basis.npz"
    body_receipt_path = tmp_path / "body_carrier_receipt.json"
    receipt_path = tmp_path / "basis_receipt.json"

    _write_vertices(vertices_path)
    BodyCarrierReceipt.from_mapping(_body_payload(promotion=0)).to_json(
        body_receipt_path
    )

    result = _run_basis_script(
        "--vertices",
        str(vertices_path),
        "--body-receipt",
        str(body_receipt_path),
        "--components",
        "4",
        "--output",
        str(basis_path),
        "--receipt-output",
        str(receipt_path),
    )

    assert result.returncode != 0
    assert "BodyCarrierReceipt not promoted" in result.stderr
    assert not basis_path.exists()
    assert not receipt_path.exists()

