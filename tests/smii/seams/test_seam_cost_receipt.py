from __future__ import annotations

from pathlib import Path

import pytest

from smii.seams.seam_cost_receipt import (
    SeamCostReceipt,
    can_consume_seam_cost_receipt,
    load_seam_cost_receipt,
)


def _receipt(promotion: int = 1) -> SeamCostReceipt:
    return SeamCostReceipt(
        rom_field_receipt_hash="rom-receipt-sha256",
        body_receipt_hash="body-receipt-sha256",
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


def test_seam_cost_receipt_json_round_trip(tmp_path: Path) -> None:
    path = _receipt().to_json(tmp_path / "seam_cost_receipt.json")

    loaded = load_seam_cost_receipt(path)

    assert loaded.rom_field_receipt_hash == "rom-receipt-sha256"
    assert loaded.solve_domain == "A_v3240"
    assert loaded.cost_uniformity == 0.4
    assert loaded.blocked_consumers == []


def test_unpromoted_seam_cost_receipt_blocks_downstream_consumers() -> None:
    receipt = _receipt(promotion=0)

    assert "solver_promotion" in receipt.blocked_consumers
    assert not can_consume_seam_cost_receipt(receipt, "solver_promotion")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("solve_domain", "unknown"),
        ("finite_cost_coverage", 1.1),
        ("cost_uniformity", -0.1),
        ("edge_count", -1),
    ],
)
def test_seam_cost_receipt_rejects_invalid_values(field: str, value: object) -> None:
    payload = _receipt().to_dict()
    payload[field] = value

    with pytest.raises((TypeError, ValueError)):
        SeamCostReceipt.from_mapping(payload)
