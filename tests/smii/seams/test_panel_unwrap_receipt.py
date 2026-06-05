from __future__ import annotations

import json
from pathlib import Path

import pytest

from smii.seams.panel_unwrap_receipt import (
    PanelUnwrapReceipt,
    can_consume_panel_unwrap_receipt,
    load_panel_unwrap_receipt,
)


def _receipt(promotion: int = 1) -> PanelUnwrapReceipt:
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
        unwrap_backend="bootstrap_projection",
        backend_is_bootstrap=True,
        distortion_margin=0.03,
        panel_unwrap_blockers=[] if promotion == 1 else ["distortion_exceeds_threshold"],
        fabric_metric_receipt={
            "schema_version": "smii.fabric_aware_panel_metric.v1",
            "promotion": 1,
            "blockers": [],
        },
        correction_tree_receipt={
            "schema_version": "smii.correction_tree_receipt.v1",
            "promotion": 1,
            "blockers": [],
        },
        correction_operator_scoring_receipt={
            "schema_version": "smii.correction_operator_scoring.v1",
            "promotion": 1,
            "blockers": [],
        },
        realized_correction_operator_receipt={
            "schema_version": "smii.realized_correction_operator.v1",
            "promotion": 1,
            "blockers": [],
        },
        correction_tree_materialization_receipt={
            "schema_version": "smii.correction_tree_materialization.v1",
            "promotion": 1,
            "blockers": [],
        },
    )


def test_panel_unwrap_receipt_json_round_trip(tmp_path: Path) -> None:
    path = _receipt().to_json(tmp_path / "panel_unwrap_receipt.json")

    loaded = load_panel_unwrap_receipt(path)

    assert loaded.solver_receipt_hash == "solver-receipt-sha256"
    assert loaded.panels_all_disks
    assert loaded.per_panel_distortion == [0.01, 0.02]
    assert loaded.grain_directions == ["warp", "weft"]
    assert loaded.blocked_consumers == []
    assert loaded.unwrap_backend == "bootstrap_projection"
    assert loaded.cut_topology_receipt_hash == "cut-topology-sha256"
    assert loaded.backend_is_bootstrap is True
    assert loaded.distortion_margin == 0.03
    assert loaded.panel_unwrap_blockers == []
    assert loaded.fabric_metric_receipt == {
        "schema_version": "smii.fabric_aware_panel_metric.v1",
        "promotion": 1,
        "blockers": [],
    }
    assert loaded.correction_tree_receipt == {
        "schema_version": "smii.correction_tree_receipt.v1",
        "promotion": 1,
        "blockers": [],
    }
    assert loaded.correction_operator_scoring_receipt == {
        "schema_version": "smii.correction_operator_scoring.v1",
        "promotion": 1,
        "blockers": [],
    }
    assert loaded.realized_correction_operator_receipt == {
        "schema_version": "smii.realized_correction_operator.v1",
        "promotion": 1,
        "blockers": [],
    }
    assert loaded.correction_tree_materialization_receipt == {
        "schema_version": "smii.correction_tree_materialization.v1",
        "promotion": 1,
        "blockers": [],
    }


def test_panel_unwrap_receipt_loads_legacy_payload_without_backend_fields(
    tmp_path: Path,
) -> None:
    payload = _receipt().to_dict()
    for key in (
        "unwrap_backend",
        "cut_topology_receipt_hash",
        "backend_is_bootstrap",
        "distortion_margin",
        "panel_unwrap_blockers",
        "fabric_metric_receipt",
        "correction_tree_receipt",
        "correction_operator_scoring_receipt",
        "realized_correction_operator_receipt",
        "correction_tree_materialization_receipt",
    ):
        payload.pop(key)
    path = tmp_path / "legacy_panel_unwrap_receipt.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_panel_unwrap_receipt(path)

    assert loaded.unwrap_backend is None
    assert loaded.cut_topology_receipt_hash is None
    assert loaded.backend_is_bootstrap is None
    assert loaded.distortion_margin is None
    assert loaded.panel_unwrap_blockers is None
    assert loaded.fabric_metric_receipt is None
    assert loaded.correction_tree_receipt is None
    assert loaded.correction_operator_scoring_receipt is None
    assert loaded.realized_correction_operator_receipt is None
    assert loaded.correction_tree_materialization_receipt is None


def test_unpromoted_panel_unwrap_receipt_blocks_manufacturing() -> None:
    receipt = _receipt(promotion=0)

    assert "manufacturing" in receipt.blocked_consumers
    assert not can_consume_panel_unwrap_receipt(receipt, "manufacturing")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("panel_count", -1),
        ("panels_all_disks", "yes"),
        ("per_panel_distortion", [0.01]),
        ("grain_directions", ["selvedge", "warp"]),
        ("promotion", 2),
        ("unwrap_backend", "arap"),
        ("backend_is_bootstrap", "yes"),
        ("panel_unwrap_blockers", "distortion_exceeds_threshold"),
        ("fabric_metric_receipt", []),
        ("correction_tree_receipt", []),
        ("correction_operator_scoring_receipt", []),
        ("realized_correction_operator_receipt", []),
        ("correction_tree_materialization_receipt", []),
    ],
)
def test_panel_unwrap_receipt_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    payload = _receipt().to_dict()
    payload[field] = value

    with pytest.raises((TypeError, ValueError)):
        PanelUnwrapReceipt.from_mapping(payload)
