from __future__ import annotations

import json
from pathlib import Path

import pytest

from smii.seams.cut_topology_receipt import (
    CutTopologyReceipt,
    can_consume_cut_topology_receipt,
    load_cut_topology_receipt,
)


def _receipt(promotion: int = 1) -> CutTopologyReceipt:
    return CutTopologyReceipt(
        solver_receipt_hash="solver-sha256",
        mesh_hash="mesh-sha256",
        seam_edges_hash="seam-edges-sha256",
        seam_edge_segment_count=8,
        seam_vertex_count=8,
        seam_connected_component_count=1,
        seam_endpoint_count=0,
        seam_branch_vertex_count=0,
        panel_count=2,
        panel_face_counts=[12, 12],
        panel_boundary_edge_counts=[4, 4],
        panels_are_disks=True,
        typed_dart_count=0,
        typed_gusset_count=0,
        promotion=promotion,
        blocked_consumers=[],
        cut_topology_blockers=[] if promotion == 1 else ["seam_graph_not_cut_graph"],
    )


def test_cut_topology_receipt_json_round_trip(tmp_path: Path) -> None:
    path = _receipt().to_json(tmp_path / "cut_topology_receipt.json")

    loaded = load_cut_topology_receipt(path)

    assert loaded.solver_receipt_hash == "solver-sha256"
    assert loaded.panel_face_counts == [12, 12]
    assert loaded.panel_boundary_edge_counts == [4, 4]
    assert loaded.panels_are_disks
    assert loaded.blocked_consumers == []
    assert can_consume_cut_topology_receipt(loaded, "panel_unwrap")


def test_unpromoted_cut_topology_blocks_panel_unwrap() -> None:
    receipt = _receipt(promotion=0)

    assert "panel_unwrap" in receipt.blocked_consumers
    assert "manufacturing" in receipt.blocked_consumers
    assert not can_consume_cut_topology_receipt(receipt, "panel_unwrap")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("panel_count", -1),
        ("panel_face_counts", [12]),
        ("panel_boundary_edge_counts", [4]),
        ("panels_are_disks", "yes"),
        ("promotion", 2),
        ("cut_topology_blockers", "seam_graph_not_cut_graph"),
    ],
)
def test_cut_topology_receipt_rejects_invalid_values(field: str, value: object) -> None:
    payload = _receipt().to_dict()
    payload[field] = value

    with pytest.raises((TypeError, ValueError)):
        CutTopologyReceipt.from_mapping(payload)


def test_cut_topology_receipt_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(TypeError):
        load_cut_topology_receipt(path)
