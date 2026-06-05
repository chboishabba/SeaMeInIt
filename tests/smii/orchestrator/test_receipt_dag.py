from __future__ import annotations

import hashlib
from pathlib import Path

from smii.meshing.body_carrier_receipt import BodyCarrierReceipt
from smii.meshing.correspondence_receipt import CorrespondenceReceipt
from smii.orchestrator.receipt_dag import read_receipt_dag
from smii.rom.basis_receipt import BasisReceipt
from smii.rom.rom_field_receipt import ROMFieldReceipt
from smii.seams.cut_topology_receipt import CutTopologyReceipt
from smii.seams.manufacturing_receipt import ManufacturingReceipt
from smii.seams.metric_correction_receipt import MetricCorrectionReceipt
from smii.seams.panel_unwrap_receipt import PanelUnwrapReceipt
from smii.seams.seam_cost_receipt import SeamCostReceipt
from smii.seams.solver_promotion_receipt import SolverPromotionReceipt

_ROM_HASH_A = "a" * 64
_ROM_HASH_B = "b" * 64
_ROM_HASH_C = "c" * 64
_ROM_HASH_D = "d" * 64
_BODY_HASH_A = "e" * 64
_BODY_HASH_B = "f" * 64
_BODY_HASH_C = "1" * 64
_BODY_HASH_D = "2" * 64


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _body_payload(promotion: int = 1) -> dict[str, object]:
    return {
        "source_hash": _BODY_HASH_A,
        "raw_reprojection_hash": _BODY_HASH_B,
        "refined_pre_repair_hash": _BODY_HASH_C,
        "repaired_export_hash": _BODY_HASH_D,
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
        "basis_receipt_hash": _ROM_HASH_A,
        "samples_hash": _ROM_HASH_B,
        "aggregation_summary_hash": _ROM_HASH_C,
        "fields_hash": _ROM_HASH_D,
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


def _cut_topology_payload(promotion: int = 1) -> dict[str, object]:
    return {
        "solver_receipt_hash": "solver-receipt-sha256",
        "mesh_hash": "mesh-sha256",
        "seam_edges_hash": "seam-sha256",
        "seam_edge_segment_count": 32,
        "seam_vertex_count": 40,
        "seam_connected_component_count": 1,
        "seam_endpoint_count": 0,
        "seam_branch_vertex_count": 0,
        "panel_count": 4,
        "panel_face_counts": [100, 100, 100, 100],
        "panel_boundary_edge_counts": [40, 40, 40, 40],
        "panels_are_disks": True,
        "typed_dart_count": 0,
        "typed_gusset_count": 0,
        "promotion": promotion,
        "blocked_consumers": [],
        "cut_topology_blockers": [] if promotion == 1 else ["seam_graph_not_cut_graph"],
    }


def _metric_correction_payload(
    *,
    solver_receipt_hash: str = "solver-receipt-sha256",
    cut_topology_receipt_hash: str = "cut-topology-receipt-sha256",
    seam_edges_hash: str = "seam-sha256",
    promotion: int = 1,
) -> dict[str, object]:
    return {
        "solver_receipt_hash": solver_receipt_hash,
        "cut_topology_receipt_hash": cut_topology_receipt_hash,
        "seam_edges_hash": seam_edges_hash,
        "panels_requiring_correction": [0],
        "corrections": [
            {
                "panel_label": 0,
                "correction_type": "dart",
                "delta_metric_meaning": "local first-fundamental-form relaxation",
                "raw_residual": 0.04,
                "corrected_residual": 0.01,
                "energy_terms": {"shape": 0.01},
                "result_state": "correctionOk",
                "blockers": [],
            }
        ],
        "raw_residual_total": 0.04,
        "corrected_residual_total": 0.01,
        "residual_gate": 0.05,
        "promotion": promotion,
        "blocked_consumers": [],
        "metric_correction_blockers": [],
    }


def _manufacturing_payload(promotion: int = 1) -> dict[str, object]:
    return {
        "panel_unwrap_receipt_hash": "panel-receipt-sha256",
        "panel_count": 4,
        "manufacturing_method": "home_sewing",
        "accessibility_level": "consumer",
        "seam_allowance_hash": "allowance-sha256",
        "seam_allowance_mean": 0.016,
        "seam_allowance_min": 0.015,
        "seam_allowance_max": 0.020,
        "allowance_varies": True,
        "grain_directions": ["warp", "weft", "bias", "warp"],
        "panel_hashes": [
            "panel-0-sha256",
            "panel-1-sha256",
            "panel-2-sha256",
            "panel-3-sha256",
        ],
        "cutting_artifacts_hash": "cutting-sha256",
        "notches_present": True,
        "labels_present": True,
        "promotion": promotion,
        "blocked_consumers": [],
        "notes": "",
    }


def _write_known_receipts(run_dir: Path) -> None:
    BodyCarrierReceipt.from_mapping(_body_payload()).to_json(run_dir / "body_carrier_receipt.json")
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


def test_receipt_dag_reports_hash_chain_mismatches(tmp_path: Path) -> None:
    BodyCarrierReceipt.from_mapping(_body_payload()).to_json(tmp_path / "body_carrier_receipt.json")
    BasisReceipt.from_mapping(_basis_payload()).to_json(tmp_path / "basis_receipt.json")

    state = read_receipt_dag(tmp_path)

    assert not state.hash_chain_valid()
    assert any("basis.carrier_receipt_hash mismatch" in item for item in state.hash_chain_errors)

    basis_payload = _basis_payload()
    basis_payload["carrier_receipt_hash"] = _sha256_file(tmp_path / "body_carrier_receipt.json")
    BasisReceipt.from_mapping(basis_payload).to_json(tmp_path / "basis_receipt.json")

    state = read_receipt_dag(tmp_path)

    assert state.hash_chain_valid()
    assert state.hash_chain_errors == ()


def test_solver_eligibility_requires_body_basis_rom_seam_cost_and_correspondence(
    tmp_path: Path,
) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(tmp_path / "rom_field_receipt.json")
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(tmp_path / "seam_cost_receipt.json")

    state = read_receipt_dag(tmp_path)

    assert state.is_solver_eligible()
    assert state.first_blocker == "solver"
    assert state.rom_field_receipt is not None
    assert state.seam_cost_receipt is not None
    assert not state.can_unwrap_panels()


def test_solver_receipt_enables_panel_unwrap_gate(tmp_path: Path) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(tmp_path / "rom_field_receipt.json")
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(tmp_path / "seam_cost_receipt.json")
    SolverPromotionReceipt.from_mapping(_solver_payload()).to_json(
        tmp_path / "solver_promotion_receipt.json"
    )

    state = read_receipt_dag(tmp_path)

    assert state.solver == 1
    assert state.solver_promotion_receipt is not None
    assert state.first_blocker == "cut_topology"
    assert not state.can_unwrap_panels()


def test_cut_topology_receipt_enables_panel_unwrap_gate(tmp_path: Path) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(tmp_path / "rom_field_receipt.json")
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(tmp_path / "seam_cost_receipt.json")
    SolverPromotionReceipt.from_mapping(_solver_payload()).to_json(
        tmp_path / "solver_promotion_receipt.json"
    )
    CutTopologyReceipt.from_mapping(_cut_topology_payload()).to_json(
        tmp_path / "cut_topology_receipt.json"
    )

    state = read_receipt_dag(tmp_path)

    assert state.cut_topology == 1
    assert state.cut_topology_receipt is not None
    assert state.first_blocker == "panel"
    assert state.can_unwrap_panels()


def test_typed_cut_topology_requires_metric_correction_for_unwrap_gate(
    tmp_path: Path,
) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(tmp_path / "rom_field_receipt.json")
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(tmp_path / "seam_cost_receipt.json")
    solver_path = tmp_path / "solver_promotion_receipt.json"
    cut_topology_path = tmp_path / "cut_topology_receipt.json"
    metric_correction_path = tmp_path / "metric_correction_receipt.json"
    SolverPromotionReceipt.from_mapping(_solver_payload()).to_json(solver_path)
    typed_cut_payload = _cut_topology_payload()
    typed_cut_payload["typed_dart_count"] = 1
    typed_cut_payload["typed_operator_count"] = 1
    typed_cut_payload["seam_graph_classifications"] = ["typed_correction_operator"]
    CutTopologyReceipt.from_mapping(typed_cut_payload).to_json(cut_topology_path)

    state_without_metric = read_receipt_dag(tmp_path)

    assert state_without_metric.cut_topology == 1
    assert not state_without_metric.can_unwrap_panels()

    MetricCorrectionReceipt.from_mapping(
        _metric_correction_payload(
            solver_receipt_hash=_sha256_file(solver_path),
            cut_topology_receipt_hash=_sha256_file(cut_topology_path),
        )
    ).to_json(metric_correction_path)

    state_with_metric = read_receipt_dag(tmp_path)

    assert state_with_metric.metric_correction == 1
    assert state_with_metric.metric_correction_receipt is not None
    assert state_with_metric.can_unwrap_panels()


def test_solver_receipt_overrides_manual_solver_gate(tmp_path: Path) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(tmp_path / "rom_field_receipt.json")
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(tmp_path / "seam_cost_receipt.json")
    SolverPromotionReceipt.from_mapping(_solver_payload(promotion=0)).to_json(
        tmp_path / "solver_promotion_receipt.json"
    )

    state = read_receipt_dag(tmp_path, solver=1)

    assert state.solver == 0
    assert state.first_blocker == "solver"
    assert not state.can_unwrap_panels()


def test_panel_unwrap_receipt_enables_manufacturing_gate(tmp_path: Path) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(tmp_path / "rom_field_receipt.json")
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(tmp_path / "seam_cost_receipt.json")
    SolverPromotionReceipt.from_mapping(_solver_payload()).to_json(
        tmp_path / "solver_promotion_receipt.json"
    )
    CutTopologyReceipt.from_mapping(_cut_topology_payload()).to_json(
        tmp_path / "cut_topology_receipt.json"
    )
    PanelUnwrapReceipt.from_mapping(_panel_unwrap_payload()).to_json(
        tmp_path / "panel_unwrap_receipt.json"
    )

    state = read_receipt_dag(tmp_path)

    assert state.panel == 1
    assert state.panel_unwrap_receipt is not None
    assert state.first_blocker == "manufacture"
    assert not state.can_manufacture()


def test_manufacturing_receipt_completes_receipt_chain(tmp_path: Path) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(tmp_path / "rom_field_receipt.json")
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(tmp_path / "seam_cost_receipt.json")
    SolverPromotionReceipt.from_mapping(_solver_payload()).to_json(
        tmp_path / "solver_promotion_receipt.json"
    )
    CutTopologyReceipt.from_mapping(_cut_topology_payload()).to_json(
        tmp_path / "cut_topology_receipt.json"
    )
    PanelUnwrapReceipt.from_mapping(_panel_unwrap_payload()).to_json(
        tmp_path / "panel_unwrap_receipt.json"
    )
    ManufacturingReceipt.from_mapping(_manufacturing_payload()).to_json(
        tmp_path / "manufacturing_receipt.json"
    )

    state = read_receipt_dag(tmp_path)

    assert state.manufacture == 1
    assert state.manufacturing_receipt is not None
    assert state.first_blocker is None
    assert state.can_manufacture()


def test_manufacturing_receipt_overrides_manual_manufacture_gate(tmp_path: Path) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(tmp_path / "rom_field_receipt.json")
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(tmp_path / "seam_cost_receipt.json")
    SolverPromotionReceipt.from_mapping(_solver_payload()).to_json(
        tmp_path / "solver_promotion_receipt.json"
    )
    CutTopologyReceipt.from_mapping(_cut_topology_payload()).to_json(
        tmp_path / "cut_topology_receipt.json"
    )
    PanelUnwrapReceipt.from_mapping(_panel_unwrap_payload()).to_json(
        tmp_path / "panel_unwrap_receipt.json"
    )
    ManufacturingReceipt.from_mapping(_manufacturing_payload(promotion=0)).to_json(
        tmp_path / "manufacturing_receipt.json"
    )

    state = read_receipt_dag(tmp_path, manufacture=1)

    assert state.manufacture == 0
    assert state.first_blocker == "manufacture"
    assert not state.can_manufacture()


def test_panel_unwrap_receipt_overrides_manual_panel_gate(tmp_path: Path) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(tmp_path / "rom_field_receipt.json")
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(tmp_path / "seam_cost_receipt.json")
    SolverPromotionReceipt.from_mapping(_solver_payload()).to_json(
        tmp_path / "solver_promotion_receipt.json"
    )
    CutTopologyReceipt.from_mapping(_cut_topology_payload()).to_json(
        tmp_path / "cut_topology_receipt.json"
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
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(tmp_path / "rom_field_receipt.json")

    state = read_receipt_dag(tmp_path)

    assert state.first_blocker == "seam_cost"
    assert not state.is_solver_eligible()


def test_seam_cost_receipt_overrides_manual_seam_cost_gate(tmp_path: Path) -> None:
    _write_known_receipts(tmp_path)
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(tmp_path / "rom_field_receipt.json")
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
    BodyCarrierReceipt.from_mapping(_body_payload()).to_json(tmp_path / "body_carrier_receipt.json")
    CorrespondenceReceipt.from_mapping(_correspondence_payload()).to_json(
        tmp_path / "transform_receipt.json"
    )
    BasisReceipt.from_mapping(_basis_payload()).to_json(tmp_path / "basis_receipt.json")
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(tmp_path / "rom_field_receipt.json")
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(tmp_path / "seam_cost_receipt.json")

    state = read_receipt_dag(tmp_path)

    assert state.correspondence == 1
    assert state.correspondence_receipt is not None
    assert state.is_solver_eligible()


def test_a_v3240_solver_domain_bypasses_correspondence_gate(tmp_path: Path) -> None:
    BodyCarrierReceipt.from_mapping(_body_payload()).to_json(tmp_path / "body_carrier_receipt.json")
    BasisReceipt.from_mapping(_basis_payload()).to_json(tmp_path / "basis_receipt.json")
    ROMFieldReceipt.from_mapping(_rom_field_payload()).to_json(tmp_path / "rom_field_receipt.json")
    SeamCostReceipt.from_mapping(_seam_cost_payload()).to_json(tmp_path / "seam_cost_receipt.json")

    state = read_receipt_dag(tmp_path, solve_domain="A_v3240")

    assert state.correspondence == 0
    assert state.first_blocker == "solver"
    assert state.is_solver_eligible()
