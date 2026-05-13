from __future__ import annotations

import pytest

from smii.meshing.body_carrier_receipt import (
    BodyCarrierReceipt,
    DEFAULT_BODY_BLOCKED_CONSUMERS,
    can_consume_receipt,
    load_body_carrier_receipt,
    normalize_promotion,
    with_blocked_consumers,
    with_promotion,
)


def _receipt_payload() -> dict[str, object]:
    return {
        "source_hash": "source-abc",
        "raw_reprojection_hash": "raw-def",
        "refined_pre_repair_hash": "refined-ghi",
        "repaired_export_hash": "repaired-jkl",
        "vertex_count": 10475,
        "face_count": 20908,
        "topology_label": "smplx_body_v1",
        "landmark_residuals": {"nose": 0.8, "left_shoulder": 1.2},
        "skull_rigidity_residual": 0.03,
        "body_fit_confidence": 0.91,
        "promotion": 1,
        "blocked_consumers": [],
    }


def test_body_carrier_receipt_from_mapping_coerces_json_scalars() -> None:
    payload = _receipt_payload()
    payload["vertex_count"] = "10475"
    payload["landmark_residuals"] = {"nose": "0.8"}
    payload["promotion"] = "1"

    receipt = BodyCarrierReceipt.from_mapping(payload)

    assert receipt.vertex_count == 10475
    assert receipt.landmark_residuals == {"nose": 0.8}
    assert receipt.promotion == 1
    assert receipt.to_dict()["blocked_consumers"] == []


def test_body_carrier_receipt_rejects_invalid_promotion() -> None:
    payload = _receipt_payload()
    payload["promotion"] = 2

    with pytest.raises(ValueError, match="promotion"):
        BodyCarrierReceipt.from_mapping(payload)


def test_body_carrier_receipt_json_round_trip(tmp_path) -> None:
    receipt = BodyCarrierReceipt.from_mapping(_receipt_payload())
    path = receipt.to_json(tmp_path / "receipt.json")

    loaded = load_body_carrier_receipt(path)

    assert loaded == receipt
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_promotion_helpers_copy_and_gate_consumers() -> None:
    receipt = BodyCarrierReceipt.from_mapping(_receipt_payload())
    held = with_promotion(receipt, 0)

    assert held.promotion == 0
    assert held.blocked_consumers == list(DEFAULT_BODY_BLOCKED_CONSUMERS)
    assert receipt.promotion == 1
    assert not can_consume_receipt(held, "generate_undersuit")

    blocked = with_blocked_consumers(receipt, ["undersuit"], promotion=-1)

    assert blocked.promotion == -1
    assert blocked.blocked_consumers == ["undersuit"]
    assert not can_consume_receipt(blocked, "undersuit")
    assert can_consume_receipt(receipt, "undersuit")


def test_non_promoted_receipt_defaults_to_explicit_downstream_blocks() -> None:
    payload = _receipt_payload()
    payload["promotion"] = 0

    receipt = BodyCarrierReceipt.from_mapping(payload)

    assert receipt.blocked_consumers == list(DEFAULT_BODY_BLOCKED_CONSUMERS)
    assert not can_consume_receipt(receipt, "generate_undersuit")


def test_can_consume_receipt_without_consumer_requires_no_blocks() -> None:
    receipt = BodyCarrierReceipt.from_mapping(_receipt_payload())
    blocked = with_blocked_consumers(receipt, ["hard_shell"])

    assert can_consume_receipt(receipt)
    assert not can_consume_receipt(blocked)
    assert can_consume_receipt(blocked, "undersuit")


def test_normalize_promotion_rejects_unknown_state() -> None:
    with pytest.raises(ValueError):
        normalize_promotion(42)
