from __future__ import annotations

import pytest

from smii.rom.rom_field_receipt import (
    DEFAULT_ROM_FIELD_BLOCKED_CONSUMERS,
    ROMFieldReceipt,
    can_consume_rom_field_receipt,
    load_rom_field_receipt,
    normalize_promotion,
    with_promotion,
)


def _receipt_payload() -> dict[str, object]:
    return {
        "basis_receipt_hash": "basis-receipt-sha256",
        "samples_hash": "samples-sha256",
        "aggregation_summary_hash": "aggregation-sha256",
        "fields_hash": "fields-sha256",
        "pose_count": 6,
        "total_samples": 8,
        "pose_source": "rom_corpus_aggregated",
        "fields_computed": ["pressure", "shear"],
        "vertex_count": 10475,
        "peak_pressure_max": 1.5,
        "peak_pressure_percentile95": 1.2,
        "field_uniformity": 0.42,
        "synthetic": False,
        "promotion": 1,
        "blocked_consumers": [],
    }


def test_rom_field_receipt_from_mapping_coerces_json_scalars() -> None:
    payload = _receipt_payload()
    payload["pose_count"] = "6"
    payload["total_samples"] = "8"
    payload["vertex_count"] = "10475"
    payload["peak_pressure_max"] = "1.5"
    payload["peak_pressure_percentile95"] = "1.2"
    payload["field_uniformity"] = "0.42"
    payload["promotion"] = "1"

    receipt = ROMFieldReceipt.from_mapping(payload)

    assert receipt.basis_receipt_hash == "basis-receipt-sha256"
    assert receipt.pose_count == 6
    assert receipt.total_samples == 8
    assert receipt.fields_computed == ["pressure", "shear"]
    assert receipt.field_uniformity == 0.42
    assert receipt.promotion == 1
    assert receipt.blocked_consumers == []


def test_rom_field_receipt_json_round_trip(tmp_path) -> None:
    receipt = ROMFieldReceipt.from_mapping(_receipt_payload())
    path = receipt.to_json(tmp_path / "rom_field_receipt.json")

    loaded = load_rom_field_receipt(path)

    assert loaded == receipt
    assert path.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pose_count", 0),
        ("total_samples", -1),
        ("vertex_count", 1.5),
        ("field_uniformity", 1.1),
        ("fields_computed", []),
    ],
)
def test_rom_field_receipt_rejects_invalid_values(field: str, value: object) -> None:
    payload = _receipt_payload()
    payload[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        ROMFieldReceipt.from_mapping(payload)


def test_rom_field_receipt_rejects_invalid_promotion() -> None:
    payload = _receipt_payload()
    payload["promotion"] = 2

    with pytest.raises(ValueError, match="promotion"):
        ROMFieldReceipt.from_mapping(payload)


def test_unpromoted_rom_field_receipt_blocks_downstream_consumers() -> None:
    payload = _receipt_payload()
    payload["promotion"] = 0

    receipt = ROMFieldReceipt.from_mapping(payload)

    assert receipt.blocked_consumers == list(DEFAULT_ROM_FIELD_BLOCKED_CONSUMERS)
    assert not can_consume_rom_field_receipt(receipt, "seam_cost_field")


def test_promotion_helper_copies_without_enabling_held_receipt() -> None:
    receipt = ROMFieldReceipt.from_mapping(_receipt_payload())
    held = with_promotion(receipt, 0)

    assert can_consume_rom_field_receipt(receipt)
    assert held.blocked_consumers == list(DEFAULT_ROM_FIELD_BLOCKED_CONSUMERS)
    assert not can_consume_rom_field_receipt(held)


def test_normalize_promotion_rejects_unknown_state() -> None:
    with pytest.raises(ValueError):
        normalize_promotion(True)

