from __future__ import annotations

import json
from pathlib import Path

import pytest

from smii.seams.metric_correction_receipt import (
    MetricCorrectionEntry,
    MetricCorrectionReceipt,
    can_consume_metric_correction_receipt,
    load_metric_correction_receipt,
)


def _receipt(promotion: int = 1) -> MetricCorrectionReceipt:
    return MetricCorrectionReceipt(
        solver_receipt_hash="solver-sha256",
        cut_topology_receipt_hash="cut-topology-sha256",
        seam_edges_hash="seam-edges-sha256",
        panels_requiring_correction=[0],
        corrections=[
            MetricCorrectionEntry(
                panel_label=0,
                correction_type="dart",
                delta_metric_meaning="local first-fundamental-form relaxation",
                raw_residual=0.08,
                corrected_residual=0.02,
                energy_terms={"shape": 0.02, "seam": 0.01},
                result_state="correctionOk",
                blockers=[],
            )
        ],
        raw_residual_total=0.08,
        corrected_residual_total=0.02,
        residual_gate=0.05,
        promotion=promotion,
        blocked_consumers=[],
        metric_correction_blockers=[] if promotion == 1 else ["missingDeltaMetricMeaning"],
    )


def test_metric_correction_receipt_json_round_trip(tmp_path: Path) -> None:
    path = _receipt().to_json(tmp_path / "metric_correction_receipt.json")

    loaded = load_metric_correction_receipt(path)

    assert loaded.panels_requiring_correction == [0]
    assert loaded.corrections[0].correction_type == "dart"
    assert loaded.corrections[0].result_state == "correctionOk"
    assert loaded.blocked_consumers == []
    assert can_consume_metric_correction_receipt(loaded, "panel_unwrap")


def test_unpromoted_metric_correction_blocks_downstream() -> None:
    receipt = _receipt(promotion=0)

    assert "panel_unwrap" in receipt.blocked_consumers
    assert not can_consume_metric_correction_receipt(receipt, "panel_unwrap")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("correction_type", "pleat"),
        ("result_state", "maybe"),
        ("raw_residual", -1.0),
        ("blockers", "missingDeltaMetricMeaning"),
    ],
)
def test_metric_correction_entry_rejects_invalid_values(field: str, value: object) -> None:
    payload = _receipt().corrections[0].to_dict()
    payload[field] = value

    with pytest.raises((TypeError, ValueError)):
        MetricCorrectionEntry.from_mapping(payload)


def test_metric_correction_receipt_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(TypeError):
        load_metric_correction_receipt(path)
