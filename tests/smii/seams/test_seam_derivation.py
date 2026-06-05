from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from smii.seams.cut_topology_receipt import CutTopologyReceipt
from smii.seams.manufacturing_receipt import ManufacturingReceipt
from smii.seams.metric_correction_receipt import MetricCorrectionEntry, MetricCorrectionReceipt
from smii.seams.panel_unwrap_receipt import PanelUnwrapReceipt
from smii.seams.seam_cost_receipt import SeamCostReceipt
from smii.seams.seam_derivation import (
    FinishedSeamReceipt,
    can_consume_finished_seam_receipt,
    derive_finished_seams,
    load_finished_seam_receipt,
)
from smii.seams.solver_promotion_receipt import SolverPromotionReceipt


def _seam_cost_receipt(promotion: int = 1) -> SeamCostReceipt:
    return SeamCostReceipt(
        rom_field_receipt_hash="rom-field-sha256",
        body_receipt_hash="body-sha256",
        correspondence_receipt_hash=None,
        solve_domain="A_v3240",
        vertex_count=4,
        edge_count=5,
        finite_cost_coverage=1.0,
        cost_uniformity=0.4,
        peak_cost=2.0,
        mean_cost=1.0,
        weight_vector={"w_P": 1.0, "w_S": 0.8},
        costs_hash="costs-sha256",
        promotion=promotion,
        blocked_consumers=[],
    )


def _solver_receipt(promotion: int = 1) -> SolverPromotionReceipt:
    return SolverPromotionReceipt(
        seam_cost_receipt_hash="seam-cost-receipt-sha256",
        solver_mode="shortest_path",
        anchor_count=4,
        anchor_source="field_minima",
        connected_component_count=1,
        anchor_fallback_used=False,
        seam_edge_count=3,
        seam_vertex_count=4,
        total_seam_cost=2.5,
        panel_count=2,
        panels_are_disks=True,
        seam_hash="seam-sha256",
        promotion=promotion,
        blocked_consumers=[],
    )


def _cut_topology_receipt(promotion: int = 1) -> CutTopologyReceipt:
    return CutTopologyReceipt(
        solver_receipt_hash="solver-sha256",
        mesh_hash="mesh-sha256",
        seam_edges_hash="seam-edges-sha256",
        seam_edge_segment_count=8,
        seam_vertex_count=8,
        seam_connected_component_count=1,
        seam_endpoint_count=0,
        seam_branch_vertex_count=0,
        panel_count=2,
        panel_face_counts=[12, 12],
        panel_boundary_edge_counts=[4, 4],
        panels_are_disks=True,
        typed_dart_count=0,
        typed_gusset_count=0,
        typed_relief_cut_count=0,
        typed_ease_count=0,
        typed_stretch_zone_count=0,
        promotion=promotion,
        blocked_consumers=[],
        cut_topology_blockers=[] if promotion == 1 else ["seam_graph_not_cut_graph"],
    )


def _metric_correction_receipt(promotion: int = 1) -> MetricCorrectionReceipt:
    return MetricCorrectionReceipt(
        solver_receipt_hash="solver-sha256",
        cut_topology_receipt_hash="cut-topology-sha256",
        seam_edges_hash="seam-edges-sha256",
        panels_requiring_correction=[0],
        corrections=[
            MetricCorrectionEntry(
                panel_label=0,
                correction_type="dart",
                delta_metric_meaning="local first-fundamental-form relaxation",
                raw_residual=0.08,
                corrected_residual=0.02,
                energy_terms={"shape": 0.02, "seam": 0.01},
                result_state="correctionOk",
                blockers=[],
            )
        ],
        raw_residual_total=0.08,
        corrected_residual_total=0.02,
        residual_gate=0.05,
        promotion=promotion,
        blocked_consumers=[],
        metric_correction_blockers=[] if promotion == 1 else ["missingDeltaMetricMeaning"],
        correction_payload_hash="metric-correction-sha256",
    )


def _panel_unwrap_receipt(promotion: int = 1) -> PanelUnwrapReceipt:
    return PanelUnwrapReceipt(
        solver_receipt_hash="solver-receipt-sha256",
        panel_count=2,
        panels_all_disks=True,
        per_panel_distortion=[0.01, 0.02],
        worst_panel_distortion=0.02,
        mean_panel_distortion=0.015,
        distortion_threshold=0.05,
        subdivision_iterations=0,
        grain_directions=["warp", "weft"],
        uv_hash="uv-sha256",
        seam_topology_hash="seam-sha256",
        promotion=promotion,
        blocked_consumers=[],
        cut_topology_receipt_hash="cut-topology-sha256",
        unwrap_backend="lscm",
        backend_is_bootstrap=False,
        distortion_margin=0.03,
        panel_unwrap_blockers=[] if promotion == 1 else ["distortion_exceeds_threshold"],
    )


def _manufacturing_receipt(promotion: int = 1) -> ManufacturingReceipt:
    return ManufacturingReceipt(
        panel_unwrap_receipt_hash="panel-receipt-sha256",
        panel_count=2,
        manufacturing_method="home_sewing",
        accessibility_level="consumer",
        seam_allowance_hash="allowance-sha256",
        seam_allowance_mean=0.016,
        seam_allowance_min=0.015,
        seam_allowance_max=0.020,
        allowance_varies=True,
        grain_directions=["warp", "weft"],
        panel_hashes=["panel-0-sha256", "panel-1-sha256"],
        cutting_artifacts_hash="cutting-sha256",
        notches_present=True,
        labels_present=True,
        promotion=promotion,
        blocked_consumers=[],
        notes="",
    )


def test_derive_finished_seams_promotes_complete_receipt_chain() -> None:
    receipt = derive_finished_seams(
        body_receipt_hash="body-sha256",
        rom_receipt_hash="rom-sha256",
        fabric_receipt_hash="fabric-sha256",
        basis_receipt_hash="basis-sha256",
        seam_cost_receipt=_seam_cost_receipt(),
        solver_receipt=_solver_receipt(),
        cut_topology_receipt=_cut_topology_receipt(),
        panel_unwrap_receipt=_panel_unwrap_receipt(),
        metric_correction_receipt=_metric_correction_receipt(),
        manufacturing_receipt=_manufacturing_receipt(),
        manufacturing_exports_hash="exports-sha256",
    )

    assert receipt.promotion == 1
    assert can_consume_finished_seam_receipt(receipt, "manufacturing")
    assert receipt.blocker_log == []
    assert receipt.selected_seam_count == 3
    assert receipt.panel_count == 2
    assert receipt.shaping_operator_counts["dart"] == 1
    assert receipt.stage_receipt_hashes["metric_correction"] == "metric-correction-sha256"
    assert receipt.claim_boundary.endswith("not_geometry_truth")


def test_derive_finished_seams_blocks_missing_manufacturing_receipt() -> None:
    receipt = derive_finished_seams(
        body_receipt_hash="body-sha256",
        rom_receipt_hash="rom-sha256",
        fabric_receipt_hash="fabric-sha256",
        basis_receipt_hash="basis-sha256",
        seam_cost_receipt=_seam_cost_receipt(),
        solver_receipt=_solver_receipt(),
        cut_topology_receipt=_cut_topology_receipt(),
        panel_unwrap_receipt=_panel_unwrap_receipt(),
        metric_correction_receipt=None,
        manufacturing_receipt=None,
    )

    assert receipt.promotion == 0
    assert "manufacturing_receipt_missing" in receipt.blocker_log
    assert "manufacturing" in receipt.blocked_consumers
    assert not can_consume_finished_seam_receipt(receipt, "manufacturing")


def test_finished_seam_receipt_json_round_trip(tmp_path: Path) -> None:
    path = derive_finished_seams(
        body_receipt_hash="body-sha256",
        rom_receipt_hash="rom-sha256",
        fabric_receipt_hash="fabric-sha256",
        basis_receipt_hash="basis-sha256",
        seam_cost_receipt=_seam_cost_receipt(),
        solver_receipt=_solver_receipt(),
        cut_topology_receipt=_cut_topology_receipt(),
        panel_unwrap_receipt=_panel_unwrap_receipt(),
        metric_correction_receipt=_metric_correction_receipt(),
        manufacturing_receipt=_manufacturing_receipt(),
    ).to_json(tmp_path / "finished_seam_receipt.json")

    loaded = load_finished_seam_receipt(path)

    assert loaded.body_receipt_hash == "body-sha256"
    assert loaded.stage_receipt_hashes["panel_unwrap"] == "uv-sha256"
    assert loaded.allowance_policy == "variable_boundary_field"
    assert loaded.atlas_boundary == "adaptive_body_rom_fabric_seam_atlas"


def test_finished_seam_receipt_matches_schema() -> None:
    schema_path = Path("schemas/finished_seam_receipt.schema.json")
    schema = schema_path.read_text(encoding="utf-8")
    validator = Draft202012Validator(json.loads(schema))
    payload = derive_finished_seams(
        body_receipt_hash="body-sha256",
        rom_receipt_hash="rom-sha256",
        fabric_receipt_hash="fabric-sha256",
        basis_receipt_hash="basis-sha256",
        seam_cost_receipt=_seam_cost_receipt(),
        solver_receipt=_solver_receipt(),
        cut_topology_receipt=_cut_topology_receipt(),
        panel_unwrap_receipt=_panel_unwrap_receipt(),
        metric_correction_receipt=_metric_correction_receipt(),
        manufacturing_receipt=_manufacturing_receipt(),
        manufacturing_exports_hash="exports-sha256",
    ).to_dict()

    validator.validate(payload)
    assert payload["claim_boundary"]["export_is_geometry_truth"] is False
    assert payload["claim_boundary"]["claims_global_optimum"] is False
    assert payload["claim_boundary"]["claims_isometry"] is False
    assert payload["claim_boundary"]["claims_true_inverse"] is False
    assert payload["claim_boundary"]["claims_manufacturing_authority_without_gate"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("panel_count", -1),
        ("selected_seam_count", -1),
        ("promotion", 2),
        ("stage_receipt_hashes", []),
        ("shaping_operator_counts", []),
        ("blocked_consumers", "manufacturing"),
    ],
)
def test_finished_seam_receipt_rejects_invalid_values(field: str, value: object) -> None:
    payload = derive_finished_seams(
        body_receipt_hash="body-sha256",
        rom_receipt_hash="rom-sha256",
        fabric_receipt_hash="fabric-sha256",
        basis_receipt_hash="basis-sha256",
        seam_cost_receipt=_seam_cost_receipt(),
        solver_receipt=_solver_receipt(),
        cut_topology_receipt=_cut_topology_receipt(),
        panel_unwrap_receipt=_panel_unwrap_receipt(),
        metric_correction_receipt=_metric_correction_receipt(),
        manufacturing_receipt=_manufacturing_receipt(),
    ).to_dict()
    payload[field] = value

    with pytest.raises((TypeError, ValueError)):
        FinishedSeamReceipt.from_mapping(payload)
