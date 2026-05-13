from __future__ import annotations

from pathlib import Path

from smii.meshing.body_carrier_receipt import BodyCarrierReceipt
from smii.meshing.correspondence_receipt import CorrespondenceReceipt
from smii.orchestrator.receipt_dag import read_receipt_dag
from smii.rom.basis_receipt import BasisReceipt


def _body_payload(promotion: int = 1) -> dict[str, object]:
    return {
        "source_hash": "source-abc",
        "raw_reprojection_hash": "raw-def",
        "refined_pre_repair_hash": "refined-ghi",
        "repaired_export_hash": "repaired-jkl",
        "vertex_count": 10475,
        "face_count": 20908,
        "topology_label": "smplx_body_v1",
        "landmark_residuals": {"nose": 0.8},
        "skull_rigidity_residual": 0.03,
        "body_fit_confidence": 0.91,
        "promotion": promotion,
        "blocked_consumers": [],
    }


def _correspondence_payload(promotion: int = 1) -> dict[str, object]:
    return {
        "source_mesh_hash": "source-mesh-abc",
        "target_mesh_hash": "target-mesh-def",
        "transform_type": "nearest_neighbor_transfer",
        "mean_distance": 0.42,
        "max_distance": 1.8,
        "collision_ratio": 0.01,
        "retention_ratio": 0.98,
        "unique_targets_used": 9400,
        "total_target_vertices": 10000,
        "edge_retention_ratio": 0.96,
        "promotion": promotion,
        "notes": [],
        "blocked_consumers": [],
    }


def _basis_payload(promotion: int = 1) -> dict[str, object]:
    return {
        "carrier_receipt_hash": "carrier-receipt-sha256",
        "basis_vertex_count": 10475,
        "basis_dimension": 24,
        "construction_method": "b0_qr_snapshots_v1",
        "reconstruction_error": 0.0025,
        "promotion": promotion,
        "blocked_consumers": [],
    }


def _write_known_receipts(run_dir: Path) -> None:
    BodyCarrierReceipt.from_mapping(_body_payload()).to_json(
        run_dir / "body_carrier_receipt.json"
    )
    CorrespondenceReceipt.from_mapping(_correspondence_payload()).to_json(
        run_dir / "correspondence_receipt.json"
    )
    BasisReceipt.from_mapping(_basis_payload()).to_json(run_dir / "basis_receipt.json")


def test_reads_known_receipts_and_defaults_future_gates_to_zero(tmp_path: Path) -> None:
    _write_known_receipts(tmp_path)

    state = read_receipt_dag(tmp_path)

    assert state.body == 1
    assert state.correspondence == 1
    assert state.basis == 1
    assert state.rom_field == 0
    assert state.seam_cost == 0
    assert state.solver == 0
    assert state.panel == 0
    assert state.manufacture == 0
    assert state.first_blocker == "rom_field"
    assert state.body_receipt is not None
    assert state.correspondence_receipt is not None
    assert state.basis_receipt is not None
    assert not state.is_solver_eligible()


def test_solver_eligibility_requires_body_basis_rom_and_correspondence(
    tmp_path: Path,
) -> None:
    _write_known_receipts(tmp_path)

    state = read_receipt_dag(tmp_path, rom_field=1)

    assert state.is_solver_eligible()
    assert state.first_blocker == "seam_cost"


def test_missing_receipts_are_unpromoted_and_record_first_blocker(
    tmp_path: Path,
) -> None:
    state = read_receipt_dag(tmp_path)

    assert state.body == 0
    assert state.correspondence == 0
    assert state.basis == 0
    assert state.first_blocker == "body"
    assert not state.is_solver_eligible()


def test_transform_receipt_is_used_as_correspondence_fallback(tmp_path: Path) -> None:
    BodyCarrierReceipt.from_mapping(_body_payload()).to_json(
        tmp_path / "body_carrier_receipt.json"
    )
    CorrespondenceReceipt.from_mapping(_correspondence_payload()).to_json(
        tmp_path / "transform_receipt.json"
    )
    BasisReceipt.from_mapping(_basis_payload()).to_json(tmp_path / "basis_receipt.json")

    state = read_receipt_dag(tmp_path, rom_field=1)

    assert state.correspondence == 1
    assert state.correspondence_receipt is not None
    assert state.is_solver_eligible()


def test_a_v3240_solver_domain_bypasses_correspondence_gate(tmp_path: Path) -> None:
    BodyCarrierReceipt.from_mapping(_body_payload()).to_json(
        tmp_path / "body_carrier_receipt.json"
    )
    BasisReceipt.from_mapping(_basis_payload()).to_json(tmp_path / "basis_receipt.json")

    state = read_receipt_dag(tmp_path, solve_domain="A_v3240", rom_field=1)

    assert state.correspondence == 0
    assert state.first_blocker == "seam_cost"
    assert state.is_solver_eligible()
