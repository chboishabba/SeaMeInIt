from __future__ import annotations

from pathlib import Path

import pytest

from smii.seams.manufacturing_receipt import (
    ManufacturingReceipt,
    can_consume_manufacturing_receipt,
    load_manufacturing_receipt,
)


def _receipt(promotion: int = 1) -> ManufacturingReceipt:
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


def test_manufacturing_receipt_json_round_trip(tmp_path: Path) -> None:
    path = _receipt().to_json(tmp_path / "manufacturing_receipt.json")

    loaded = load_manufacturing_receipt(path)

    assert loaded.panel_unwrap_receipt_hash == "panel-receipt-sha256"
    assert loaded.manufacturing_method == "home_sewing"
    assert loaded.allowance_varies
    assert loaded.grain_directions == ["warp", "weft"]
    assert loaded.blocked_consumers == []
    assert can_consume_manufacturing_receipt(loaded)


def test_unpromoted_manufacturing_receipt_is_diagnostic_only() -> None:
    receipt = _receipt(promotion=0)

    assert not can_consume_manufacturing_receipt(receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("panel_count", -1),
        ("manufacturing_method", "glue_magic"),
        ("accessibility_level", "secret_lab"),
        ("seam_allowance_min", 0.030),
        ("allowance_varies", "yes"),
        ("grain_directions", ["warp"]),
        ("panel_hashes", ["panel-0-sha256"]),
        ("promotion", 2),
    ],
)
def test_manufacturing_receipt_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    payload = _receipt().to_dict()
    payload[field] = value

    with pytest.raises((TypeError, ValueError)):
        ManufacturingReceipt.from_mapping(payload)
