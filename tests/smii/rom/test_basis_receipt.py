from __future__ import annotations

import pytest

from smii.rom.basis_receipt import (
    BasisReceipt,
    DEFAULT_BASIS_BLOCKED_CONSUMERS,
    can_consume_basis_receipt,
    load_basis_receipt,
    normalize_promotion,
    with_promotion,
)


def _receipt_payload() -> dict[str, object]:
    return {
        "carrier_receipt_hash": "carrier-receipt-sha256",
        "basis_vertex_count": 10475,
        "basis_dimension": 24,
        "construction_method": "b0_qr_snapshots_v1",
        "reconstruction_error": 0.0025,
        "promotion": 1,
        "blocked_consumers": [],
    }


def test_basis_receipt_from_mapping_coerces_json_scalars() -> None:
    payload = _receipt_payload()
    payload["basis_vertex_count"] = "10475"
    payload["basis_dimension"] = "24"
    payload["reconstruction_error"] = "0.0025"
    payload["promotion"] = "1"

    receipt = BasisReceipt.from_mapping(payload)

    assert receipt.carrier_receipt_hash == "carrier-receipt-sha256"
    assert receipt.basis_vertex_count == 10475
    assert receipt.basis_dimension == 24
    assert receipt.reconstruction_error == 0.0025
    assert receipt.promotion == 1
    assert receipt.blocked_consumers == []


def test_basis_receipt_json_round_trip(tmp_path) -> None:
    receipt = BasisReceipt.from_mapping(_receipt_payload())
    path = receipt.to_json(tmp_path / "basis_receipt.json")

    loaded = load_basis_receipt(path)

    assert loaded == receipt
    assert path.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("basis_vertex_count", 0),
        ("basis_vertex_count", -1),
        ("basis_dimension", 0),
        ("basis_dimension", 1.5),
    ],
)
def test_basis_receipt_rejects_invalid_dimensions(field: str, value: object) -> None:
    payload = _receipt_payload()
    payload[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        BasisReceipt.from_mapping(payload)


def test_basis_receipt_rejects_invalid_promotion() -> None:
    payload = _receipt_payload()
    payload["promotion"] = 2

    with pytest.raises(ValueError, match="promotion"):
        BasisReceipt.from_mapping(payload)


def test_unpromoted_basis_receipt_is_not_consumable() -> None:
    receipt = BasisReceipt.from_mapping(_receipt_payload())
    held = with_promotion(receipt, 0)
    rejected = with_promotion(receipt, -1)

    assert can_consume_basis_receipt(receipt)
    assert held.blocked_consumers == list(DEFAULT_BASIS_BLOCKED_CONSUMERS)
    assert not can_consume_basis_receipt(held)
    assert not can_consume_basis_receipt(rejected)


def test_unpromoted_basis_receipt_blocks_field_consumers() -> None:
    payload = _receipt_payload()
    payload["promotion"] = 0

    receipt = BasisReceipt.from_mapping(payload)

    assert receipt.blocked_consumers == list(DEFAULT_BASIS_BLOCKED_CONSUMERS)
    assert not can_consume_basis_receipt(receipt, "rom_field_aggregation")


def test_normalize_promotion_rejects_unknown_state() -> None:
    with pytest.raises(ValueError):
        normalize_promotion(True)
