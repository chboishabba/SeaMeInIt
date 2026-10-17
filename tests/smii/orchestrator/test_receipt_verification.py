from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from smii.orchestrator.receipt_verification import (
    TrustStatus,
    VerificationSeverity,
    verify_receipt_dag,
)


from tests.smii.orchestrator.test_receipt_dag import (
    _basis_payload,
    _body_payload,
    _correspondence_payload,
    _cut_topology_payload,
    _manufacturing_payload,
    _metric_correction_payload,
    _panel_unwrap_payload,
    _rom_field_payload,
    _seam_cost_payload,
    _solver_payload,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return _sha256(path)


def _create_clean_chain(tmp_path: Path) -> dict[str, str]:
    """Create a fully-promoted, validly linked Gates 0 through 7 receipt chain."""
    hashes: dict[str, str] = {}

    # Gate 0: Body Carrier Receipt
    body_path = tmp_path / "body_carrier_receipt.json"
    hashes["body"] = _write_json(body_path, _body_payload())

    # Gate 1: Correspondence Receipt (Optional for A_v3240, but present)
    corr_path = tmp_path / "correspondence_receipt.json"
    hashes["correspondence"] = _write_json(corr_path, _correspondence_payload())

    # Gate 2: Basis Receipt
    basis_data = _basis_payload()
    basis_data["carrier_receipt_hash"] = hashes["body"]
    basis_path = tmp_path / "basis_receipt.json"
    hashes["basis"] = _write_json(basis_path, basis_data)

    # Gate 3: ROM Field Receipt
    rom_data = _rom_field_payload()
    rom_data["basis_receipt_hash"] = hashes["basis"]
    rom_path = tmp_path / "rom_field_receipt.json"
    hashes["rom_field"] = _write_json(rom_path, rom_data)

    # Gate 4: Seam Cost Receipt
    sc_data = _seam_cost_payload()
    sc_data["rom_field_receipt_hash"] = hashes["rom_field"]
    sc_data["body_receipt_hash"] = hashes["body"]
    sc_data["correspondence_receipt_hash"] = None
    sc_data["solve_domain"] = "A_v3240"
    seam_cost_path = tmp_path / "seam_cost_receipt.json"
    hashes["seam_cost"] = _write_json(seam_cost_path, sc_data)

    # Gate 5: Solver Promotion Receipt
    solver_data = _solver_payload()
    solver_data["seam_cost_receipt_hash"] = hashes["seam_cost"]
    solver_path = tmp_path / "solver_promotion_receipt.json"
    hashes["solver"] = _write_json(solver_path, solver_data)

    # Gate 5.5: Cut Topology Receipt
    cut_data = _cut_topology_payload()
    cut_data["solver_receipt_hash"] = hashes["solver"]
    cut_topology_path = tmp_path / "cut_topology_receipt.json"
    hashes["cut_topology"] = _write_json(cut_topology_path, cut_data)

    # Gate 5.6: Metric Correction Receipt
    metric_data = _metric_correction_payload(
        solver_receipt_hash=hashes["solver"],
        cut_topology_receipt_hash=hashes["cut_topology"],
    )
    metric_path = tmp_path / "metric_correction_receipt.json"
    hashes["metric_correction"] = _write_json(metric_path, metric_data)

    # Gate 6: Panel Unwrap Receipt
    panel_data = _panel_unwrap_payload()
    panel_data["solver_receipt_hash"] = hashes["solver"]
    panel_data["cut_topology_receipt_hash"] = hashes["cut_topology"]
    panel_data["metric_correction_receipt_hash"] = hashes["metric_correction"]
    panel_path = tmp_path / "panel_unwrap_receipt.json"
    hashes["panel"] = _write_json(panel_path, panel_data)

    # Gate 7: Manufacturing Receipt
    mfg_data = _manufacturing_payload()
    mfg_data["panel_unwrap_receipt_hash"] = hashes["panel"]
    mfg_path = tmp_path / "manufacturing_receipt.json"
    hashes["manufacture"] = _write_json(mfg_path, mfg_data)

    return hashes


def test_clean_chain_is_accepted(tmp_path: Path) -> None:
    _create_clean_chain(tmp_path)
    snapshot = verify_receipt_dag(tmp_path, solve_domain="A_v3240")

    assert snapshot.is_accepted is True
    assert snapshot.is_blocked is False
    assert snapshot.status == TrustStatus.ACCEPTED
    assert len(snapshot.errors) == 0
    assert len(snapshot.verified_links) >= 7


def test_tampered_hash_fails_closed(tmp_path: Path) -> None:
    _create_clean_chain(tmp_path)

    # Tamper with body receipt by modifying payload on disk
    body_path = tmp_path / "body_carrier_receipt.json"
    data = json.loads(body_path.read_text(encoding="utf-8"))
    data["tampered"] = True
    body_path.write_text(json.dumps(data), encoding="utf-8")

    snapshot = verify_receipt_dag(tmp_path, solve_domain="A_v3240")

    assert snapshot.is_blocked is True
    assert snapshot.is_accepted is False
    assert snapshot.status == TrustStatus.BLOCKED
    error_codes = [issue.code for issue in snapshot.errors]
    assert "PROVENANCE_HASH_MISMATCH" in error_codes


def test_unpromoted_parent_fails_closed(tmp_path: Path) -> None:
    _create_clean_chain(tmp_path)

    # Demote basis receipt while child (rom_field) remains promoted
    basis_path = tmp_path / "basis_receipt.json"
    basis_data = json.loads(basis_path.read_text(encoding="utf-8"))
    basis_data["promotion"] = 0
    basis_path.write_text(json.dumps(basis_data), encoding="utf-8")

    # Update rom_field's reference so provenance passes, but semantic authority fails
    new_basis_hash = _sha256(basis_path)
    rom_path = tmp_path / "rom_field_receipt.json"
    rom_data = json.loads(rom_path.read_text(encoding="utf-8"))
    rom_data["basis_receipt_hash"] = new_basis_hash
    rom_path.write_text(json.dumps(rom_data), encoding="utf-8")

    # Update seam_cost reference to new rom_field hash
    new_rom_hash = _sha256(rom_path)
    seam_cost_path = tmp_path / "seam_cost_receipt.json"
    sc_data = json.loads(seam_cost_path.read_text(encoding="utf-8"))
    sc_data["rom_field_receipt_hash"] = new_rom_hash
    seam_cost_path.write_text(json.dumps(sc_data), encoding="utf-8")

    snapshot = verify_receipt_dag(tmp_path, solve_domain="A_v3240")

    assert snapshot.is_blocked is True
    error_codes = [issue.code for issue in snapshot.errors]
    assert "AUTHORITY_PARENT_NOT_PROMOTED" in error_codes


def test_blockers_present_on_promoted_fails_closed(tmp_path: Path) -> None:
    _create_clean_chain(tmp_path)

    # Introduce blocker into promoted panel unwrap receipt
    panel_path = tmp_path / "panel_unwrap_receipt.json"
    panel_data = json.loads(panel_path.read_text(encoding="utf-8"))
    panel_data["panel_unwrap_blockers"] = ["distortion_exceeds_threshold"]
    panel_path.write_text(json.dumps(panel_data), encoding="utf-8")

    # Update manufacturing receipt hash to preserve provenance
    mfg_path = tmp_path / "manufacturing_receipt.json"
    mfg_data = json.loads(mfg_path.read_text(encoding="utf-8"))
    mfg_data["panel_unwrap_receipt_hash"] = _sha256(panel_path)
    mfg_path.write_text(json.dumps(mfg_data), encoding="utf-8")

    snapshot = verify_receipt_dag(tmp_path, solve_domain="A_v3240")

    assert snapshot.is_blocked is True
    error_codes = [issue.code for issue in snapshot.errors]
    assert "AUTHORITY_BLOCKERS_PRESENT_ON_PROMOTED" in error_codes


def test_invalid_json_fails_closed(tmp_path: Path) -> None:
    _create_clean_chain(tmp_path)

    # Write corrupt JSON
    (tmp_path / "solver_promotion_receipt.json").write_text("{corrupt json ...", encoding="utf-8")

    snapshot = verify_receipt_dag(tmp_path, solve_domain="A_v3240")

    assert snapshot.is_blocked is True
    error_codes = [issue.code for issue in snapshot.errors]
    assert "SCHEMA_INVALID_JSON" in error_codes


def test_invalid_promotion_value_fails_closed(tmp_path: Path) -> None:
    _create_clean_chain(tmp_path)

    body_path = tmp_path / "body_carrier_receipt.json"
    data = json.loads(body_path.read_text(encoding="utf-8"))
    data["promotion"] = 999  # Invalid promotion value
    body_path.write_text(json.dumps(data), encoding="utf-8")

    snapshot = verify_receipt_dag(tmp_path, solve_domain="A_v3240")

    assert snapshot.is_blocked is True
    error_codes = [issue.code for issue in snapshot.errors]
    assert "SCHEMA_INVALID_PROMOTION" in error_codes


def test_non_a_v3240_domain_requires_correspondence(tmp_path: Path) -> None:
    _create_clean_chain(tmp_path)

    # Remove correspondence receipt and evaluate under non-A solve domain
    (tmp_path / "correspondence_receipt.json").unlink()

    snapshot = verify_receipt_dag(tmp_path, solve_domain="B_ogre")

    assert snapshot.is_blocked is True
    error_codes = [issue.code for issue in snapshot.errors]
    assert "AUTHORITY_CORRESPONDENCE_REQUIRED" in error_codes
