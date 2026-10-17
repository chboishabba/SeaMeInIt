from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.freeze_bt369_p2_winner import freeze_winner


def _make_receipt(candidate_family: str = "flow_trace", geometric_promotion: int = 1) -> dict:
    return {
        "panel_unwrap_receipt_hash": "dummy_panel_hash",
        "mesh_hash": "dummy_mesh_hash",
        "seam_edges_hash": "dummy_seam_hash",
        "panels": [
            {
                "panel_id": "P2",
                "candidates": [
                    {
                        "candidate_id": "P2_BT369_000_flow_trace_shortened",
                        "candidate_family": candidate_family,
                        "geometric_promotion": geometric_promotion,
                        "parent_acceptance": True,
                        "foldover_guard": True,
                        "exportable_guard": True,
                        "regularized_partition": [[0, 1], [2, 3]],
                        "policy": {"policy_key": "test_policy"},
                        "manufacturing": {"seam_length_over_sqrt_area": 10.0},
                        "sensitivity": {},
                        "metrics": {"score": 255.4, "worst_distortion": 0.6, "foldovers": 25},
                        "backend_competitions": [{"backend": "xatlas", "score": 255.4}],
                    }
                ],
            }
        ],
    }


def test_freeze_winner_success(tmp_path: Path) -> None:
    receipt_data = _make_receipt(candidate_family="flow_trace_shortened", geometric_promotion=1)
    receipt_file = tmp_path / "receipt.json"
    receipt_file.write_text(json.dumps(receipt_data), encoding="utf-8")
    out_file = tmp_path / "frozen.json"

    frozen = freeze_winner(
        source=receipt_file,
        candidate_id="P2_BT369_000_flow_trace_shortened",
        out=out_file,
    )

    assert frozen["schema_version"] == "smii.bt369_frozen_p2_winner.v1"
    assert frozen["candidate_id"] == "P2_BT369_000_flow_trace_shortened"
    assert frozen["gate6"]["geometric_promotion"] == 1
    assert out_file.exists()

    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["candidate_id"] == "P2_BT369_000_flow_trace_shortened"
    assert loaded["topology"] == [[0, 1], [2, 3]]


def test_freeze_winner_rejects_unpromoted(tmp_path: Path) -> None:
    receipt_data = _make_receipt(candidate_family="flow_trace_shortened", geometric_promotion=0)
    receipt_file = tmp_path / "receipt.json"
    receipt_file.write_text(json.dumps(receipt_data), encoding="utf-8")
    out_file = tmp_path / "frozen.json"

    with pytest.raises(ValueError, match="does not have measured Gate-6 promotion"):
        freeze_winner(
            source=receipt_file,
            candidate_id="P2_BT369_000_flow_trace_shortened",
            out=out_file,
        )


def test_freeze_winner_rejects_invalid_family(tmp_path: Path) -> None:
    receipt_data = _make_receipt(candidate_family="merge_back", geometric_promotion=1)
    receipt_file = tmp_path / "receipt.json"
    receipt_file.write_text(json.dumps(receipt_data), encoding="utf-8")
    out_file = tmp_path / "frozen.json"

    with pytest.raises(ValueError, match="must be a freshly evaluated flow_trace candidate"):
        freeze_winner(
            source=receipt_file,
            candidate_id="P2_BT369_000_flow_trace_shortened",
            out=out_file,
        )
