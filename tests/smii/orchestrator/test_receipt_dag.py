from __future__ import annotations

from pathlib import Path

from smii.meshing.body_carrier_receipt import BodyCarrierReceipt
from smii.meshing.correspondence_receipt import CorrespondenceReceipt
from smii.orchestrator.receipt_dag import read_receipt_dag
from smii.rom.basis_receipt import BasisReceipt
from smii.rom.rom_field_receipt import ROMFieldReceipt
from smii.seams.panel_unwrap_receipt import PanelUnwrapReceipt
from smii.seams.seam_cost_receipt import SeamCostReceipt
from smii.seams.solver_promotion_receipt import SolverPromotionReceipt


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


def _rom_field_payload(promotion: int = 1) -> dict[str, object]:
    return {
        "basis_receipt_hash": "basis-receipt-sha256",
        "samples_hash": "samples-sha256",
        "aggregation_summary_hash": "aggregation-sha256",
        "fields_hash": "fields-sha256",
        "pose_count": 6,
        "total_samples": 6,
        "pose_source": "rom_corpus_aggregated",
        "fields_computed": ["pressure", "shear"],
        "vertex_count": 10475,
        "peak_pressure_max": 1.0,
        "peak_pressure_percentile95": 0.8,
        "field_uniformity": 0.4,
        "synthetic": False,
        "promotion": promotion,
        "blocked_consumers": [],
    }


def _seam_cost_payload(promotion: int = 1) -> dict[str, object]:
    return {
        "rom_field_receipt_hash": "rom-field-receipt-sha256",
        "body_receipt_hash": "body-receipt-sha256",
        "correspondence_receipt_hash": None,
        "solve_domain": "A_v3240",
        "vertex_count": 10475,
        "edge_count": 20908,
        "finite_cost_coverage": 1.0,
        "cost_uniformity": 0.4,
        "peak_cost": 2.0,
        "mean_cost": 1.0,
        "weight_vector": {"w_P": 1.0, "w_S": 0.8},
        "costs_hash": "costs-sha256",
        "promotion": promotion,
        "blocked_consumers": [],
    }


def _solver_payload(promotion: int = 1) -> dict[str, object]:
    return {
        "seam_cost_receipt_hash": "seam-cost-receipt-sha256",
        "solver_mode": "shortest_path",
        "anchor_count": 8,
        "anchor_source": "field_minima",
        "connected_component_count": 1,
        "anchor_fallback_used": False,
        "seam_edge_count": 32,
        "seam_vertex_count": 40,
        "total_seam_cost": 12.0,
        "panel_count": 4,
        "panels_are_disks": True,
        "seam_hash": "seam-sha256",
        "promotion": promotion,
        "blocked_consumers": [],
    }


def _panel_unwrap_payload(promotion: int = 1) -> dict[str, object]:
    return {
        "solver_receipt_hash": "solver-receipt-sha256",
        "panel_count": 4,
        "panels_all_disks": True,
        "per_panel_distortion": [0.01, 0.02, 0.015, 0.012],
        "worst_panel_distortion": 0.02,
        "mean_panel_distortion": 0.01425,
        "distortion_threshold": 0.05,
        "subdivision_iterations": 0,
        "grain_directions": ["warp", "weft", "bias", "warp"],
        "uv_hash": "uv-sha256",
        "seam_topology_hash": "seam-sha256",
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


def test_solver_eligibility_requires_body_basis_rom_seam_cost_and_correspondence(
    tmp_path: Path,
) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(
        tmp_path / "rom_field_receipt.json"
    )
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(
        tmp_path / "seam_cost_receipt.json"
    )

    state = read_receipt_dag(tmp_path)

    assert state.is_solver_eligible()
    assert state.first_blocker == "solver"
    assert state.rom_field_receipt is not None
    assert state.seam_cost_receipt is not None
    assert not state.can_unwrap_panels()


def test_solver_receipt_enables_panel_unwrap_gate(tmp_path: Path) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(
        tmp_path / "rom_field_receipt.json"
    )
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(
        tmp_path / "seam_cost_receipt.json"
    )
    SolverPromotionReceipt.from_mapping(_solver_payload()).to_json(
        tmp_path / "solver_promotion_receipt.json"
    )

    state = read_receipt_dag(tmp_path)

    assert state.solver == 1
    assert state.solver_promotion_receipt is not None
    assert state.first_blocker == "panel"
    assert state.can_unwrap_panels()


def test_solver_receipt_overrides_manual_solver_gate(tmp_path: Path) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(
        tmp_path / "rom_field_receipt.json"
    )
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(
        tmp_path / "seam_cost_receipt.json"
    )
    SolverPromotionReceipt.from_mapping(_solver_payload(promotion=0)).to_json(
        tmp_path / "solver_promotion_receipt.json"
    )

    state = read_receipt_dag(tmp_path, solver=1)

    assert state.solver == 0
    assert state.first_blocker == "solver"
    assert not state.can_unwrap_panels()


def test_panel_unwrap_receipt_enables_manufacturing_gate(tmp_path: Path) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(
        tmp_path / "rom_field_receipt.json"
    )
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(
        tmp_path / "seam_cost_receipt.json"
    )
    SolverPromotionReceipt.from_mapping(_solver_payload()).to_json(
        tmp_path / "solver_promotion_receipt.json"
    )
    PanelUnwrapReceipt.from_mapping(_panel_unwrap_payload()).to_json(
        tmp_path / "panel_unwrap_receipt.json"
    )

    state = read_receipt_dag(tmp_path)

    assert state.panel == 1
    assert state.panel_unwrap_receipt is not None
    assert state.first_blocker == "manufacture"
    assert state.can_manufacture()


def test_panel_unwrap_receipt_overrides_manual_panel_gate(tmp_path: Path) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(
        tmp_path / "rom_field_receipt.json"
    )
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(
        tmp_path / "seam_cost_receipt.json"
    )
    SolverPromotionReceipt.from_mapping(_solver_payload()).to_json(
        tmp_path / "solver_promotion_receipt.json"
    )
    PanelUnwrapReceipt.from_mapping(_panel_unwrap_payload(promotion=0)).to_json(
        tmp_path / "panel_unwrap_receipt.json"
    )

    state = read_receipt_dag(tmp_path, panel=1)

    assert state.panel == 0
    assert state.first_blocker == "panel"
    assert not state.can_manufacture()


def test_solver_eligibility_blocks_without_seam_cost_receipt(tmp_path: Path) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(
        tmp_path / "rom_field_receipt.json"
    )

    state = read_receipt_dag(tmp_path)

    assert state.first_blocker == "seam_cost"
    assert not state.is_solver_eligible()


def test_seam_cost_receipt_overrides_manual_seam_cost_gate(tmp_path: Path) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(
        tmp_path / "rom_field_receipt.json"
    )
    SeamCostReceipt.from_mapping(_seam_cost_payload(promotion=0)).to_json(
        tmp_path / "seam_cost_receipt.json"
    )

    state = read_receipt_dag(tmp_path, seam_cost=1)

    assert state.seam_cost == 0
    assert state.first_blocker == "seam_cost"
    assert not state.is_solver_eligible()


def test_rom_field_receipt_overrides_manual_rom_field_gate(tmp_path: Path) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload(promotion=0)).to_json(
        tmp_path / "rom_field_receipt.json"
    )

    state = read_receipt_dag(tmp_path, rom_field=1)

    assert state.rom_field == 0
    assert state.first_blocker == "rom_field"
    assert not state.is_solver_eligible()


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
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(
        tmp_path / "rom_field_receipt.json"
    )
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(
        tmp_path / "seam_cost_receipt.json"
    )

    state = read_receipt_dag(tmp_path)

    assert state.correspondence == 1
    assert state.correspondence_receipt is not None
    assert state.is_solver_eligible()


def test_a_v3240_solver_domain_bypasses_correspondence_gate(tmp_path: Path) -> None:
    BodyCarrierReceipt.from_mapping(_body_payload()).to_json(
        tmp_path / "body_carrier_receipt.json"
    )
    BasisReceipt.from_mapping(_basis_payload()).to_json(tmp_path / "basis_receipt.json")
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(
        tmp_path / "rom_field_receipt.json"
    )
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(
        tmp_path / "seam_cost_receipt.json"
    )

    state = read_receipt_dag(tmp_path, solve_domain="A_v3240")

    assert state.correspondence == 0
    assert state.first_blocker == "solver"
    assert state.is_solver_eligible()
