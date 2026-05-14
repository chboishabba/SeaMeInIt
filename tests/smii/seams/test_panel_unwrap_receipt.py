from __future__ import annotations

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
    )


def test_panel_unwrap_receipt_json_round_trip(tmp_path: Path) -> None:
    path = _receipt().to_json(tmp_path / "panel_unwrap_receipt.json")

    loaded = load_panel_unwrap_receipt(path)

    assert loaded.solver_receipt_hash == "solver-receipt-sha256"
    assert loaded.panels_all_disks
    assert loaded.per_panel_distortion == [0.01, 0.02]
    assert loaded.grain_directions == ["warp", "weft"]
    assert loaded.blocked_consumers == []


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
