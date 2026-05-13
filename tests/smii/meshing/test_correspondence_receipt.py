from __future__ import annotations

import pytest

from smii.meshing.correspondence_receipt import (
    CorrespondenceReceipt,
    DEFAULT_CORRESPONDENCE_BLOCKED_CONSUMERS,
    can_consume_correspondence_receipt,
    is_diagnostic_nn_collapse,
    load_correspondence_receipt,
    normalize_promotion,
    with_promotion,
)


def _receipt_payload() -> dict[str, object]:
    return {
        "source_mesh_hash": "source-mesh-abc",
        "target_mesh_hash": "target-mesh-def",
        "transform_type": "nearest_neighbor_transfer",
        "mean_distance": 0.42,
        "max_distance": 1.8,
        "collision_ratio": 0.01,
        "retention_ratio": 0.98,
        "unique_targets_used": 9400,
        "total_target_vertices": 10000,
        "edge_retention_ratio": 0.96,
        "promotion": 1,
        "notes": ["transfer metrics only; no geometry recomputation"],
        "blocked_consumers": [],
    }


def test_correspondence_receipt_from_mapping_coerces_json_scalars() -> None:
    payload = _receipt_payload()
    payload["mean_distance"] = "0.42"
    payload["unique_targets_used"] = "9400"
    payload["promotion"] = "1"
    payload["notes"] = "single note"

    receipt = CorrespondenceReceipt.from_mapping(payload)

    assert receipt.mean_distance == 0.42
    assert receipt.unique_targets_used == 9400
    assert receipt.promotion == 1
    assert receipt.notes == ["single note"]
    assert can_consume_correspondence_receipt(receipt)


def test_correspondence_receipt_json_round_trip(tmp_path) -> None:
    receipt = CorrespondenceReceipt.from_mapping(_receipt_payload())
    path = receipt.to_json(tmp_path / "correspondence_receipt.json")

    loaded = load_correspondence_receipt(path)

    assert loaded == receipt
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_correspondence_receipt_rejects_invalid_promotion() -> None:
    payload = _receipt_payload()
    payload["promotion"] = 2

    with pytest.raises(ValueError, match="promotion"):
        CorrespondenceReceipt.from_mapping(payload)


def test_normalize_promotion_rejects_bool() -> None:
    with pytest.raises(ValueError, match="promotion"):
        normalize_promotion(True)


def test_collapsed_nn_receipt_is_not_consumable() -> None:
    payload = _receipt_payload()
    payload["unique_targets_used"] = 1
    payload["total_target_vertices"] = 10000
    payload["promotion"] = 1
    payload["notes"] = ["diagnostic nearest-neighbor collapse"]
    receipt = CorrespondenceReceipt.from_mapping(payload)

    assert is_diagnostic_nn_collapse(receipt)
    assert not can_consume_correspondence_receipt(receipt)


def test_unpromoted_receipt_defaults_to_explicit_downstream_blocks() -> None:
    payload = _receipt_payload()
    payload["promotion"] = 0

    receipt = CorrespondenceReceipt.from_mapping(payload)

    assert receipt.blocked_consumers == list(DEFAULT_CORRESPONDENCE_BLOCKED_CONSUMERS)
    assert not can_consume_correspondence_receipt(receipt, "solver_promotion")


def test_promotion_helper_copies_without_enabling_held_receipt() -> None:
    receipt = CorrespondenceReceipt.from_mapping(_receipt_payload())
    held = with_promotion(receipt, 0)

    assert held.promotion == 0
    assert held.blocked_consumers == list(DEFAULT_CORRESPONDENCE_BLOCKED_CONSUMERS)
    assert receipt.promotion == 1
    assert not can_consume_correspondence_receipt(held)
