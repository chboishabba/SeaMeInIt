from __future__ import annotations

import hashlib
from pathlib import Path

from smii.orchestrator.receipt_dag import ReceiptDagState
from smii.orchestrator.suit_closure_dag import read_suit_closure_dag
from smii.seams.seam_derivation import FinishedSeamReceipt
from smii.seams.suit_closure_receipt import (
    AssemblyReceipt,
    PhysicalROMValidationReceipt,
    TriadicTransportWitness,
    WearFitReceipt,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digital_state(root: Path) -> ReceiptDagState:
    return ReceiptDagState(
        run_dir=root,
        solve_domain="A_v3240",
        body=1,
        correspondence=0,
        basis=1,
        rom_field=1,
        seam_cost=1,
        solver=1,
        cut_topology=1,
        metric_correction=1,
        panel=1,
        manufacture=1,
        first_blocker=None,
    )


def _write_chain(root: Path) -> None:
    finished_path = root / "finished_seam_receipt.json"
    FinishedSeamReceipt(
        body_receipt_hash="body-sha256",
        rom_receipt_hash="rom-sha256",
        fabric_receipt_hash="fabric-sha256",
        basis_receipt_hash="basis-sha256",
        stage_receipt_hashes={"manufacturing": "manufacturing-sha256"},
        selected_seam_count=3,
        panel_count=2,
        shaping_operator_counts={},
        allowance_policy="variable_boundary_field",
        atlas_boundary="adaptive_body_rom_fabric_seam_atlas",
        promotion=1,
        blocked_consumers=[],
        blocker_log=[],
    ).to_json(finished_path)

    assembly_path = root / "assembly_receipt.json"
    AssemblyReceipt(
        finished_seam_receipt_hash=_sha256(finished_path),
        manufactured_artifacts_hash="manufactured-sha256",
        assembled_geometry_hash="assembled-sha256",
        assembly_evidence_hash="assembly-evidence-sha256",
        piece_count=2,
        realized_seam_count=3,
        max_geometry_deviation=0.004,
        geometry_deviation_threshold=0.010,
        material_lot_hashes=["lot-sha256"],
        promotion=1,
        blocked_consumers=[],
        blocker_log=[],
    ).to_json(assembly_path)

    wear_path = root / "wear_fit_receipt.json"
    WearFitReceipt(
        assembly_receipt_hash=_sha256(assembly_path),
        rom_receipt_hash="rom-sha256",
        task_profile_hash="task-sha256",
        predicted_field_hash="predicted-sha256",
        measured_field_hash="measured-sha256",
        measurement_evidence_hash="measurement-sha256",
        sample_count=100,
        task_coverage=1.0,
        field_residuals={"pressure": 0.01},
        residual_thresholds={"pressure": 0.05},
        promotion=1,
        blocked_consumers=[],
        blocker_log=[],
    ).to_json(wear_path)

    PhysicalROMValidationReceipt(
        wear_fit_receipt_hash=_sha256(wear_path),
        source_rom_receipt_hash="rom-sha256",
        validation_evidence_hash="validation-sha256",
        field_residual_hash="residual-sha256",
        closure_iteration=1,
        calibrated_rom_receipt_hash="calibrated-rom-sha256",
        triadic_transport=TriadicTransportWitness(
            transport_model="bt369_c3",
            address_schema="smii.bt369.triadic_address.v1",
            phase_provenance_hash="phase-sha256",
            orientation_receipt_hash="orientation-sha256",
            c3_equivariance_error=0.01,
            c3_equivariance_threshold=0.02,
        ),
        promotion=1,
        blocked_consumers=[],
        blocker_log=[],
    ).to_json(root / "physical_rom_validation_receipt.json")


def test_full_closure_dag_promotes(monkeypatch, tmp_path: Path) -> None:
    _write_chain(tmp_path)
    monkeypatch.setattr(
        "smii.orchestrator.suit_closure_dag.read_receipt_dag",
        lambda root, solve_domain=None: _digital_state(Path(root)),
    )

    state = read_suit_closure_dag(tmp_path, solve_domain="A_v3240")

    assert state.first_blocker is None
    assert state.hash_chain_valid()
    assert state.can_assemble()
    assert state.can_wear_test()
    assert state.can_validate_physical_rom()
    assert state.can_deploy()


def test_physical_hash_mismatch_blocks_closure(monkeypatch, tmp_path: Path) -> None:
    _write_chain(tmp_path)
    assembly_path = tmp_path / "assembly_receipt.json"
    assembly_path.write_text(assembly_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "smii.orchestrator.suit_closure_dag.read_receipt_dag",
        lambda root, solve_domain=None: _digital_state(Path(root)),
    )

    state = read_suit_closure_dag(tmp_path, solve_domain="A_v3240")

    assert state.first_blocker == "physical_hash_chain"
    assert not state.hash_chain_valid()
    assert not state.can_deploy()
