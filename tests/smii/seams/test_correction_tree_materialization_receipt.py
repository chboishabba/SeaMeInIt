from __future__ import annotations

import json
from pathlib import Path

import pytest

from smii.seams.correction_tree_materialization_receipt import (
    CORRECTION_TREE_MATERIALIZATION_SCHEMA,
    CorrectionTreeMaterializationEntry,
    CorrectionTreeMaterializationReceipt,
    can_consume_correction_tree_materialization_receipt,
    load_correction_tree_materialization_receipt,
)


def _entry(promotion: int = 1) -> CorrectionTreeMaterializationEntry:
    return CorrectionTreeMaterializationEntry(
        node_id="node-dart-0",
        operator_family="dart_apex",
        metric_realized=True,
        chart_materialized=True,
        materialization_kind="backend_constraint_patch",
        affected_panels=[0, 1],
        backend_constraints_emitted=True,
        promotion=promotion,
        blockers=[] if promotion == 1 else ["operator_not_materialized"],
    )


def _receipt(promotion: int = 1) -> CorrectionTreeMaterializationReceipt:
    return CorrectionTreeMaterializationReceipt(
        correction_tree_hash="correction-tree-sha256",
        materializations=[_entry(promotion=promotion)],
        materialized_operator_count=1,
        promotion=promotion,
        blocked_consumers=[],
        blockers=[] if promotion == 1 else ["materialization_incomplete"],
        correction_tree_receipt_hash="correction-tree-receipt-sha256",
        correction_operator_scoring_receipt_hash="operator-scoring-sha256",
    )


def test_correction_tree_materialization_receipt_json_round_trip(
    tmp_path: Path,
) -> None:
    path = _receipt().to_json(tmp_path / "correction_tree_materialization_receipt.json")

    loaded = load_correction_tree_materialization_receipt(path)

    assert loaded.to_dict()["schema_version"] == CORRECTION_TREE_MATERIALIZATION_SCHEMA
    assert loaded.correction_tree_hash == "correction-tree-sha256"
    assert loaded.materialized_operator_count == 1
    assert loaded.materializations[0].node_id == "node-dart-0"
    assert loaded.materializations[0].operator_family == "dart_apex"
    assert loaded.materializations[0].metric_realized
    assert loaded.materializations[0].chart_materialized
    assert loaded.materializations[0].backend_constraints_emitted
    assert loaded.materializations[0].affected_panels == [0, 1]
    assert loaded.blocked_consumers == []
    assert can_consume_correction_tree_materialization_receipt(loaded, "panel_unwrap")


def test_unpromoted_correction_tree_materialization_blocks_downstream() -> None:
    receipt = _receipt(promotion=0)

    assert "panel_unwrap" in receipt.blocked_consumers
    assert "manufacturing" in receipt.blocked_consumers
    assert not can_consume_correction_tree_materialization_receipt(
        receipt,
        "panel_unwrap",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("node_id", ""),
        ("metric_realized", "yes"),
        ("chart_materialized", 1),
        ("affected_panels", [-1]),
        ("backend_constraints_emitted", "yes"),
        ("promotion", 2),
        ("blockers", "operator_not_materialized"),
    ],
)
def test_correction_tree_materialization_entry_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    payload = _entry().to_dict()
    payload[field] = value

    with pytest.raises((TypeError, ValueError)):
        CorrectionTreeMaterializationEntry.from_mapping(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("correction_tree_hash", ""),
        ("materializations", ["node-dart-0"]),
        ("materialized_operator_count", 2),
        ("promotion", 2),
        ("blocked_consumers", "panel_unwrap"),
        ("blockers", "materialization_incomplete"),
    ],
)
def test_correction_tree_materialization_receipt_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    payload = _receipt().to_dict()
    payload[field] = value

    with pytest.raises((TypeError, ValueError)):
        CorrectionTreeMaterializationReceipt.from_mapping(payload)


def test_correction_tree_materialization_receipt_rejects_non_object_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(TypeError):
        load_correction_tree_materialization_receipt(path)
