from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from smii.seams import SolverPromotionReceipt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_solver_receipt(path: Path, seam_edges_path: Path, *, promotion: int = 1) -> None:
    payload = np.load(seam_edges_path, allow_pickle=False)
    seam_edges = np.asarray(payload["seam_edges"], dtype=int)
    seam_vertices = sorted({int(vertex) for edge in seam_edges for vertex in edge})
    SolverPromotionReceipt(
        seam_cost_receipt_hash="seam-cost-receipt-sha256",
        solver_mode="shortest_path",
        anchor_count=2,
        anchor_source="field_minima",
        connected_component_count=1,
        anchor_fallback_used=False,
        seam_edge_count=int(seam_edges.shape[0]),
        seam_vertex_count=len(seam_vertices),
        total_seam_cost=float(seam_edges.shape[0]),
        panel_count=2,
        panels_are_disks=True,
        seam_hash=_sha256_file(seam_edges_path),
        promotion=promotion,
        blocked_consumers=[],
    ).to_json(path)


def _write_tetra_mesh(path: Path) -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=int)
    np.savez(path, vertices=vertices, faces=faces)


def _write_square_mesh(path: Path) -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    np.savez(path, vertices=vertices, faces=faces)


def _run_validate(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, "scripts/validate_cut_topology.py", *args],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_validate_cut_topology_blocks_open_branched_closed_mesh_path(tmp_path: Path) -> None:
    mesh_path = tmp_path / "tetra.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    receipt_path = tmp_path / "cut_topology_receipt.json"
    _write_tetra_mesh(mesh_path)
    np.savez_compressed(seam_edges_path, seam_edges=np.array([[0, 1], [1, 3]], dtype=int))
    _write_solver_receipt(solver_receipt_path, seam_edges_path)

    result = _run_validate(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-cut-topology-receipt",
        str(receipt_path),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["promotion"] == 0
    assert receipt["seam_edge_segment_count"] == 2
    assert receipt["seam_endpoint_count"] == 2
    assert receipt["panel_boundary_edge_counts"] == [0]
    assert receipt["cut_topology_blockers"] == [
        "unresolved_open_boundary",
        "panel_fragmentation_invalid",
        "seam_graph_not_cut_graph",
        "no_cut_mesh_boundary",
    ]


def test_validate_cut_topology_promotes_simple_two_panel_cut(tmp_path: Path) -> None:
    mesh_path = tmp_path / "square.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    receipt_path = tmp_path / "cut_topology_receipt.json"
    _write_square_mesh(mesh_path)
    np.savez_compressed(seam_edges_path, seam_edges=np.array([[0, 2]], dtype=int))
    _write_solver_receipt(solver_receipt_path, seam_edges_path)

    result = _run_validate(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-cut-topology-receipt",
        str(receipt_path),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["promotion"] == 1
    assert receipt["panel_count"] == 2
    assert receipt["panel_face_counts"] == [1, 1]
    assert receipt["panel_boundary_edge_counts"] == [3, 3]
    assert receipt["cut_topology_blockers"] == []
    repair_receipt = json.loads(
        (tmp_path / "cut_topology_repair_receipt.json").read_text(encoding="utf-8")
    )
    assert repair_receipt["schema_version"] == "smii.cut_topology_repair.v1"
    assert repair_receipt["promotion"] == 1
    assert repair_receipt["panel_count"] == 2
    assert all(check["backend_serializable"] for check in repair_receipt["panel_checks"])


def test_validate_cut_topology_classifies_authorized_typed_branch(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "square.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    receipt_path = tmp_path / "cut_topology_receipt.json"
    _write_square_mesh(mesh_path)
    np.savez_compressed(
        seam_edges_path,
        seam_edges=np.array([[0, 2], [0, 1], [0, 3]], dtype=int),
    )
    _write_solver_receipt(solver_receipt_path, seam_edges_path)

    result = _run_validate(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-cut-topology-receipt",
        str(receipt_path),
        "--typed-dart-count",
        "1",
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["promotion"] == 1
    assert receipt["typed_operator_count"] == 1
    assert receipt["typed_dart_count"] == 1
    assert receipt["invalid_fragmentation_count"] == 0
    assert receipt["seam_graph_classifications"] == ["typed_correction_operator"]
    assert receipt["cut_topology_blockers"] == []


def test_validate_cut_topology_infers_typed_branch_from_corrections(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "square.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    receipt_path = tmp_path / "cut_topology_receipt.json"
    corrections_path = tmp_path / "corrections.json"
    _write_square_mesh(mesh_path)
    np.savez_compressed(
        seam_edges_path,
        seam_edges=np.array([[0, 2], [0, 1], [0, 3]], dtype=int),
    )
    _write_solver_receipt(solver_receipt_path, seam_edges_path)
    corrections_path.write_text(
        json.dumps(
            {
                "selected_corrections": [
                    {"family": "dart", "selected": True},
                    {"family": "ease", "selected": False},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = _run_validate(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-cut-topology-receipt",
        str(receipt_path),
        "--corrections",
        str(corrections_path),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["promotion"] == 1
    assert receipt["typed_operator_count"] == 1
    assert receipt["typed_dart_count"] == 1
    assert receipt["typed_ease_count"] == 0
    assert receipt["invalid_fragmentation_count"] == 0


def test_validate_cut_topology_rejects_unpromoted_solver(tmp_path: Path) -> None:
    mesh_path = tmp_path / "square.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    solver_receipt_path = tmp_path / "solver_promotion_receipt.json"
    _write_square_mesh(mesh_path)
    np.savez_compressed(seam_edges_path, seam_edges=np.array([[0, 2]], dtype=int))
    _write_solver_receipt(solver_receipt_path, seam_edges_path, promotion=0)

    result = _run_validate(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(seam_edges_path),
        "--mesh",
        str(mesh_path),
        "--out-cut-topology-receipt",
        str(tmp_path / "cut_topology_receipt.json"),
    )

    assert result.returncode != 0
    assert "SolverPromotionReceipt not promoted" in result.stderr
