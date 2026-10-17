from __future__ import annotations

from pathlib import Path

from smii.seams.seam_derivation import FinishedSeamReceipt
from smii.seams.suit_closure_receipt import (
    AssemblyReceipt,
    PhysicalROMValidationReceipt,
    TriadicTransportWitness,
    WearFitReceipt,
    can_consume_assembly_receipt,
    can_consume_physical_rom_validation_receipt,
    can_consume_wear_fit_receipt,
    derive_assembly_receipt,
    derive_physical_rom_validation_receipt,
    derive_wear_fit_receipt,
    load_assembly_receipt,
    load_physical_rom_validation_receipt,
    load_wear_fit_receipt,
)


def _finished_seam(promotion: int = 1) -> FinishedSeamReceipt:
    return FinishedSeamReceipt(
        body_receipt_hash="body-sha256",
        rom_receipt_hash="rom-sha256",
        fabric_receipt_hash="fabric-sha256",
        basis_receipt_hash="basis-sha256",
        stage_receipt_hashes={
            "solver": "solver-sha256",
            "manufacturing": "manufacturing-sha256",
        },
        selected_seam_count=3,
        panel_count=2,
        shaping_operator_counts={"dart": 1},
        allowance_policy="variable_boundary_field",
        atlas_boundary="adaptive_body_rom_fabric_seam_atlas",
        promotion=promotion,
        blocked_consumers=[],
        blocker_log=[],
        manufacturing_exports_hash="exports-sha256",
    )


def _assembly() -> AssemblyReceipt:
    return derive_assembly_receipt(
        finished_seam_receipt_hash="finished-sha256",
        finished_seam_receipt=_finished_seam(),
        manufactured_artifacts_hash="manufactured-sha256",
        assembled_geometry_hash="assembled-sha256",
        assembly_evidence_hash="assembly-evidence-sha256",
        piece_count=2,
        realized_seam_count=3,
        max_geometry_deviation=0.004,
        geometry_deviation_threshold=0.010,
        material_lot_hashes=["fabric-lot-sha256"],
    )


def _wear_fit() -> WearFitReceipt:
    return derive_wear_fit_receipt(
        assembly_receipt_hash="assembly-sha256",
        assembly_receipt=_assembly(),
        rom_receipt_hash="rom-sha256",
        task_profile_hash="task-sha256",
        predicted_field_hash="predicted-field-sha256",
        measured_field_hash="measured-field-sha256",
        measurement_evidence_hash="measurement-evidence-sha256",
        sample_count=120,
        task_coverage=0.98,
        field_residuals={"pressure": 0.02, "shear": 0.03, "support": 0.01},
        residual_thresholds={"pressure": 0.05, "shear": 0.05, "support": 0.05},
    )


def _triadic_transport(error: float = 0.01) -> TriadicTransportWitness:
    return TriadicTransportWitness(
        transport_model="bt369_c3",
        address_schema="smii.bt369.triadic_address.v1",
        phase_provenance_hash="phase-sha256",
        orientation_receipt_hash="orientation-sha256",
        c3_equivariance_error=error,
        c3_equivariance_threshold=0.02,
    )


def test_complete_physical_closure_promotes() -> None:
    assembly = _assembly()
    assert assembly.promotion == 1
    assert can_consume_assembly_receipt(assembly, "wear_fit")

    wear_fit = _wear_fit()
    assert wear_fit.promotion == 1
    assert can_consume_wear_fit_receipt(wear_fit, "physical_rom_validation")

    physical = derive_physical_rom_validation_receipt(
        wear_fit_receipt_hash="wear-fit-sha256",
        wear_fit_receipt=wear_fit,
        source_rom_receipt_hash="rom-sha256",
        validation_evidence_hash="validation-sha256",
        field_residual_hash="residual-sha256",
        closure_iteration=1,
        calibrated_rom_receipt_hash="rom-calibrated-sha256",
        triadic_transport=_triadic_transport(),
    )

    assert physical.promotion == 1
    assert physical.blocker_log == []
    assert can_consume_physical_rom_validation_receipt(physical, "analysis")
    assert physical.triadic_transport is not None
    assert physical.triadic_transport.transport_model == "bt369_c3"


def test_missing_physical_evidence_is_diagnostic_not_promoted() -> None:
    assembly = derive_assembly_receipt(
        finished_seam_receipt_hash="finished-sha256",
        finished_seam_receipt=_finished_seam(),
        manufactured_artifacts_hash="manufactured-sha256",
        assembled_geometry_hash="assembled-sha256",
        assembly_evidence_hash="assembly-evidence-sha256",
        piece_count=2,
        realized_seam_count=3,
        max_geometry_deviation=0.004,
        geometry_deviation_threshold=0.010,
        material_lot_hashes=[],
    )

    assert assembly.promotion == 0
    assert "material_lot_provenance_missing" in assembly.blocker_log
    assert not can_consume_assembly_receipt(assembly, "wear_fit")


def test_measured_threshold_violation_rejects() -> None:
    wear_fit = derive_wear_fit_receipt(
        assembly_receipt_hash="assembly-sha256",
        assembly_receipt=_assembly(),
        rom_receipt_hash="rom-sha256",
        task_profile_hash="task-sha256",
        predicted_field_hash="predicted-field-sha256",
        measured_field_hash="measured-field-sha256",
        measurement_evidence_hash="measurement-evidence-sha256",
        sample_count=120,
        task_coverage=0.98,
        field_residuals={"pressure": 0.08},
        residual_thresholds={"pressure": 0.05},
    )

    assert wear_fit.promotion == -1
    assert "field_residual_exceeds_threshold:pressure" in wear_fit.blocker_log


def test_c3_transport_is_a_real_gate_not_a_label() -> None:
    physical = derive_physical_rom_validation_receipt(
        wear_fit_receipt_hash="wear-fit-sha256",
        wear_fit_receipt=_wear_fit(),
        source_rom_receipt_hash="rom-sha256",
        validation_evidence_hash="validation-sha256",
        field_residual_hash="residual-sha256",
        closure_iteration=1,
        triadic_transport=_triadic_transport(error=0.03),
    )

    assert physical.promotion == -1
    assert "c3_equivariance_error_exceeds_threshold" in physical.blocker_log


def test_missing_required_c3_transport_remains_diagnostic() -> None:
    physical = derive_physical_rom_validation_receipt(
        wear_fit_receipt_hash="wear-fit-sha256",
        wear_fit_receipt=_wear_fit(),
        source_rom_receipt_hash="rom-sha256",
        validation_evidence_hash="validation-sha256",
        field_residual_hash="residual-sha256",
        closure_iteration=1,
        triadic_transport=None,
        require_triadic_transport=True,
    )

    assert physical.promotion == 0
    assert "triadic_transport_witness_missing" in physical.blocker_log


def test_receipts_json_round_trip(tmp_path: Path) -> None:
    assembly_path = _assembly().to_json(tmp_path / "assembly_receipt.json")
    wear_fit_path = _wear_fit().to_json(tmp_path / "wear_fit_receipt.json")
    physical = derive_physical_rom_validation_receipt(
        wear_fit_receipt_hash="wear-fit-sha256",
        wear_fit_receipt=_wear_fit(),
        source_rom_receipt_hash="rom-sha256",
        validation_evidence_hash="validation-sha256",
        field_residual_hash="residual-sha256",
        closure_iteration=1,
        triadic_transport=_triadic_transport(),
    )
    physical_path = physical.to_json(tmp_path / "physical_rom_validation_receipt.json")

    assert load_assembly_receipt(assembly_path).promotion == 1
    assert load_wear_fit_receipt(wear_fit_path).field_residuals["pressure"] == 0.02
    loaded_physical = load_physical_rom_validation_receipt(physical_path)
    assert loaded_physical.promotion == 1
    assert loaded_physical.triadic_transport is not None
    assert loaded_physical.triadic_transport.address_schema == "smii.bt369.triadic_address.v1"


def test_from_mapping_rejects_out_of_range_coverage() -> None:
    payload = _wear_fit().to_dict()
    payload["task_coverage"] = 1.1

    try:
        WearFitReceipt.from_mapping(payload)
    except ValueError as exc:
        assert "[0, 1]" in str(exc)
    else:
        raise AssertionError("expected invalid task coverage to be rejected")


def test_physical_roundtrip_preserves_optional_calibrated_rom() -> None:
    physical = PhysicalROMValidationReceipt.from_mapping(
        {
            "wear_fit_receipt_hash": "wear-fit-sha256",
            "source_rom_receipt_hash": "rom-sha256",
            "validation_evidence_hash": "validation-sha256",
            "field_residual_hash": "residual-sha256",
            "closure_iteration": 2,
            "calibrated_rom_receipt_hash": "rom-calibrated-sha256",
            "triadic_transport": _triadic_transport().to_dict(),
            "promotion": 1,
            "blocked_consumers": [],
            "blocker_log": [],
        }
    )
    assert physical.calibrated_rom_receipt_hash == "rom-calibrated-sha256"
