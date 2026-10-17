from __future__ import annotations

import hashlib
from pathlib import Path

from smii.orchestrator.adaptive_suit_dag import read_adaptive_suit_dag
from smii.orchestrator.receipt_dag import ReceiptDagState
from smii.orchestrator.suit_closure_dag import SuitClosureDagState
from smii.seams.adaptive_suit_receipt import (
    AdaptiveModuleReceipt,
    ControllerSafetyReceipt,
    SuitDeploymentReceipt,
)
from smii.seams.suit_closure_receipt import (
    PhysicalROMValidationReceipt,
    TriadicTransportWitness,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _physical_receipt() -> PhysicalROMValidationReceipt:
    return PhysicalROMValidationReceipt(
        wear_fit_receipt_hash="wear-sha256",
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
            c3_equivariance_error=0.0,
            c3_equivariance_threshold=0.01,
        ),
        promotion=1,
        blocked_consumers=[],
        blocker_log=[],
    )


def _physical_state(root: Path) -> SuitClosureDagState:
    digital = ReceiptDagState(
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
    return SuitClosureDagState(
        digital=digital,
        finished_seam=1,
        assembly=1,
        wear_fit=1,
        physical_rom=1,
        first_blocker=None,
        physical_rom_receipt=_physical_receipt(),
        hash_chain_errors=(),
    )


def _write_chain(root: Path) -> None:
    physical_path = root / "physical_rom_validation_receipt.json"
    _physical_receipt().to_json(physical_path)

    adaptive_path = root / "adaptive_module_receipt.json"
    AdaptiveModuleReceipt(
        physical_rom_receipt_hash=_sha256(physical_path),
        module_manifest_hash="modules-sha256",
        sensor_map_hash="sensors-sha256",
        actuator_map_hash="actuators-sha256",
        power_budget_hash="power-sha256",
        fail_safe_evidence_hash="failsafe-sha256",
        module_count=2,
        sensor_count=8,
        actuator_count=4,
        command_envelope_violation_rate=0.0,
        command_envelope_violation_threshold=0.001,
        fail_safe_test_count=20,
        fail_safe_failure_count=0,
        promotion=1,
        blocked_consumers=[],
        blocker_log=[],
    ).to_json(adaptive_path)

    controller_path = root / "controller_safety_receipt.json"
    ControllerSafetyReceipt(
        adaptive_module_receipt_hash=_sha256(adaptive_path),
        controller_artifact_hash="controller-sha256",
        safety_envelope_hash="envelope-sha256",
        scenario_manifest_hash="scenarios-sha256",
        controller_test_evidence_hash="tests-sha256",
        scenario_count=100,
        hard_violation_count=0,
        max_command_latency_ms=10.0,
        command_latency_threshold_ms=20.0,
        emergency_stop_test_count=20,
        emergency_stop_failure_count=0,
        promotion=1,
        blocked_consumers=[],
        blocker_log=[],
    ).to_json(controller_path)

    SuitDeploymentReceipt(
        controller_safety_receipt_hash=_sha256(controller_path),
        deployment_profile_hash="profile-sha256",
        deployment_evidence_hash="deployment-sha256",
        authority_review_receipt_hash="review-sha256",
        authority_review_passed=True,
        promotion=1,
        blocked_consumers=[],
        blocker_log=[],
    ).to_json(root / "suit_deployment_receipt.json")


def test_adaptive_dag_reaches_field_deployment(monkeypatch, tmp_path: Path) -> None:
    _write_chain(tmp_path)
    monkeypatch.setattr(
        "smii.orchestrator.adaptive_suit_dag.read_suit_closure_dag",
        lambda root, solve_domain=None: _physical_state(Path(root)),
    )

    state = read_adaptive_suit_dag(tmp_path, solve_domain="A_v3240")

    assert state.first_blocker is None
    assert state.hash_chain_valid()
    assert state.can_validate_controller()
    assert state.can_request_deployment_authority()
    assert state.can_field_deploy()


def test_adaptive_hash_tamper_blocks_field_deployment(monkeypatch, tmp_path: Path) -> None:
    _write_chain(tmp_path)
    adaptive_path = tmp_path / "adaptive_module_receipt.json"
    adaptive_path.write_text(adaptive_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "smii.orchestrator.adaptive_suit_dag.read_suit_closure_dag",
        lambda root, solve_domain=None: _physical_state(Path(root)),
    )

    state = read_adaptive_suit_dag(tmp_path, solve_domain="A_v3240")

    assert state.first_blocker == "adaptive_hash_chain"
    assert not state.hash_chain_valid()
    assert not state.can_field_deploy()
