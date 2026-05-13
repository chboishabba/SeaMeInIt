from __future__ import annotations

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
    path = _receipt().to_json(tmp_path / "solver_promotion_receipt.json")

    loaded = load_solver_promotion_receipt(path)

    assert loaded.seam_cost_receipt_hash == "seam-cost-receipt-sha256"
    assert loaded.anchor_source == "field_minima"
    assert not loaded.anchor_fallback_used
    assert loaded.panels_are_disks
    assert loaded.blocked_consumers == []


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
