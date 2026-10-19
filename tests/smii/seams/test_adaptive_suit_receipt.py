from __future__ import annotations

from smii.seams.adaptive_suit_receipt import (
    can_consume_adaptive_module_receipt,
    can_consume_controller_safety_receipt,
    can_consume_suit_deployment_receipt,
    derive_adaptive_module_receipt,
    derive_controller_safety_receipt,
    derive_suit_deployment_receipt,
)
from smii.seams.suit_closure_receipt import (
    PhysicalROMValidationReceipt,
    TriadicTransportWitness,
)


def _physical() -> PhysicalROMValidationReceipt:
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
            c3_equivariance_error=0.01,
            c3_equivariance_threshold=0.02,
        ),
        promotion=1,
        blocked_consumers=[],
        blocker_log=[],
    )


def _adaptive():
    return derive_adaptive_module_receipt(
        physical_rom_receipt_hash="physical-sha256",
        physical_rom_receipt=_physical(),
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
    )


def _controller():
    return derive_controller_safety_receipt(
        adaptive_module_receipt_hash="adaptive-sha256",
        adaptive_module_receipt=_adaptive(),
        controller_artifact_hash="controller-sha256",
        safety_envelope_hash="envelope-sha256",
        scenario_manifest_hash="scenarios-sha256",
        controller_test_evidence_hash="controller-tests-sha256",
        scenario_count=100,
        hard_violation_count=0,
        max_command_latency_ms=12.0,
        command_latency_threshold_ms=20.0,
        emergency_stop_test_count=20,
        emergency_stop_failure_count=0,
    )


def test_adaptive_chain_promotes_only_with_separate_authority_review() -> None:
    adaptive = _adaptive()
    assert adaptive.promotion == 1
    assert can_consume_adaptive_module_receipt(adaptive, "controller_safety")

    controller = _controller()
    assert controller.promotion == 1
    assert can_consume_controller_safety_receipt(controller, "deployment")

    diagnostic = derive_suit_deployment_receipt(
        controller_safety_receipt_hash="controller-safety-sha256",
        controller_safety_receipt=controller,
        deployment_profile_hash="profile-sha256",
        deployment_evidence_hash="deployment-evidence-sha256",
        authority_review_receipt_hash=None,
        authority_review_passed=None,
    )
    assert diagnostic.promotion == 0
    assert "external_authority_review_missing" in diagnostic.blocker_log
    assert not can_consume_suit_deployment_receipt(diagnostic, "field_deployment")

    promoted = derive_suit_deployment_receipt(
        controller_safety_receipt_hash="controller-safety-sha256",
        controller_safety_receipt=controller,
        deployment_profile_hash="profile-sha256",
        deployment_evidence_hash="deployment-evidence-sha256",
        authority_review_receipt_hash="review-sha256",
        authority_review_passed=True,
    )
    assert promoted.promotion == 1
    assert can_consume_suit_deployment_receipt(promoted, "field_deployment")


def test_adaptive_hardware_violation_rejects() -> None:
    receipt = derive_adaptive_module_receipt(
        physical_rom_receipt_hash="physical-sha256",
        physical_rom_receipt=_physical(),
        module_manifest_hash="modules-sha256",
        sensor_map_hash="sensors-sha256",
        actuator_map_hash="actuators-sha256",
        power_budget_hash="power-sha256",
        fail_safe_evidence_hash="failsafe-sha256",
        module_count=2,
        sensor_count=8,
        actuator_count=4,
        command_envelope_violation_rate=0.02,
        command_envelope_violation_threshold=0.001,
        fail_safe_test_count=20,
        fail_safe_failure_count=1,
    )
    assert receipt.promotion == -1
    assert "command_envelope_violation_rate_exceeds_threshold" in receipt.blocker_log
    assert "fail_safe_test_failure" in receipt.blocker_log


def test_controller_safety_violation_rejects() -> None:
    receipt = derive_controller_safety_receipt(
        adaptive_module_receipt_hash="adaptive-sha256",
        adaptive_module_receipt=_adaptive(),
        controller_artifact_hash="controller-sha256",
        safety_envelope_hash="envelope-sha256",
        scenario_manifest_hash="scenarios-sha256",
        controller_test_evidence_hash="controller-tests-sha256",
        scenario_count=100,
        hard_violation_count=1,
        max_command_latency_ms=25.0,
        command_latency_threshold_ms=20.0,
        emergency_stop_test_count=20,
        emergency_stop_failure_count=1,
    )
    assert receipt.promotion == -1
    assert "controller_hard_safety_violation" in receipt.blocker_log
    assert "controller_latency_exceeds_threshold" in receipt.blocker_log
    assert "emergency_stop_failure" in receipt.blocker_log


def test_external_review_rejection_is_hard_reject() -> None:
    receipt = derive_suit_deployment_receipt(
        controller_safety_receipt_hash="controller-safety-sha256",
        controller_safety_receipt=_controller(),
        deployment_profile_hash="profile-sha256",
        deployment_evidence_hash="deployment-evidence-sha256",
        authority_review_receipt_hash="review-sha256",
        authority_review_passed=False,
    )
    assert receipt.promotion == -1
    assert receipt.blocker_log == ["external_authority_review_rejected"]
