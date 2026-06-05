from __future__ import annotations

import json
from pathlib import Path

import pytest

from smii.seams.solver_promotion_receipt import (
    SolverPromotionReceipt,
    can_consume_solver_promotion_receipt,
    load_solver_promotion_receipt,
)


def _receipt(promotion: int = 1) -> SolverPromotionReceipt:
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
        panel_count=1,
        panels_are_disks=True,
        seam_hash="seam-sha256",
        promotion=promotion,
        blocked_consumers=[],
    )


def test_solver_promotion_receipt_json_round_trip(tmp_path: Path) -> None:
    receipt = _receipt()
    receipt = SolverPromotionReceipt(
        **receipt.to_dict(),
        requested_anchor_count=4,
        candidate_anchor_count=4,
        low_cost_anchor_component_count=2,
        min_seam_edge_count=1,
        min_seam_vertex_count=2,
        solver_blockers=[],
    )
    path = receipt.to_json(tmp_path / "solver_promotion_receipt.json")

    loaded = load_solver_promotion_receipt(path)

    assert loaded.seam_cost_receipt_hash == "seam-cost-receipt-sha256"
    assert loaded.anchor_source == "field_minima"
    assert not loaded.anchor_fallback_used
    assert loaded.panels_are_disks
    assert loaded.blocked_consumers == []
    assert loaded.requested_anchor_count == 4
    assert loaded.candidate_anchor_count == 4
    assert loaded.low_cost_anchor_component_count == 2
    assert loaded.min_seam_edge_count == 1
    assert loaded.min_seam_vertex_count == 2
    assert loaded.solver_blockers == []


def test_solver_promotion_receipt_loads_legacy_payload_without_diagnostics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy_solver_promotion_receipt.json"
    path.write_text(
        json.dumps(_receipt().to_dict()),
        encoding="utf-8",
    )

    loaded = load_solver_promotion_receipt(path)

    assert loaded.requested_anchor_count is None
    assert loaded.solver_blockers is None


def test_unpromoted_solver_receipt_blocks_panel_unwrap() -> None:
    receipt = _receipt(promotion=0)

    assert "panel_unwrap" in receipt.blocked_consumers
    assert not can_consume_solver_promotion_receipt(receipt, "panel_unwrap")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("solver_mode", "unknown"),
        ("anchor_source", "unknown"),
        ("anchor_count", -1),
        ("anchor_fallback_used", "no"),
        ("panels_are_disks", "yes"),
        ("solver_blockers", "insufficient_seam_edges"),
    ],
)
def test_solver_promotion_receipt_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    payload = _receipt().to_dict()
    payload[field] = value

    with pytest.raises((TypeError, ValueError)):
        SolverPromotionReceipt.from_mapping(payload)
