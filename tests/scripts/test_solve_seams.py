from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from smii.rom import SeamCostField, save_seam_cost_field
from smii.seams import SeamCostReceipt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_mesh(path: Path) -> tuple[tuple[int, int], ...]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=int,
    )
    np.savez(path, vertices=vertices, faces=faces)
    return ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _edges_from_faces(faces: np.ndarray) -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = set()
    for a, b, c in np.asarray(faces, dtype=int):
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add(tuple(sorted((int(u), int(v)))))
    return tuple(sorted(edges))


def _write_square_mesh(path: Path) -> tuple[tuple[int, int], ...]:
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
    return _edges_from_faces(faces)


def _write_grid_mesh(path: Path, *, columns: int = 4, rows: int = 4) -> tuple[tuple[int, int], ...]:
    vertices = np.array(
        [[float(x), float(y), 0.0] for y in range(rows + 1) for x in range(columns + 1)],
        dtype=float,
    )
    faces: list[list[int]] = []
    for y in range(rows):
        for x in range(columns):
            a = y * (columns + 1) + x
            b = a + 1
            d = (y + 1) * (columns + 1) + x
            c = d + 1
            faces.append([a, b, c])
            faces.append([a, c, d])
    face_array = np.asarray(faces, dtype=int)
    np.savez(path, vertices=vertices, faces=face_array)
    return _edges_from_faces(face_array)


def _write_two_tetra_bridge_mesh(path: Path) -> tuple[tuple[int, int], ...]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
            [4, 5, 7],
            [4, 6, 7],
            [5, 6, 7],
            [0, 1, 4],
        ],
        dtype=int,
    )
    np.savez(path, vertices=vertices, faces=faces)
    edges: set[tuple[int, int]] = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add(tuple(sorted((int(u), int(v)))))
    return tuple(sorted(edges))


def _write_costs(path: Path, edges: tuple[tuple[int, int], ...]) -> None:
    save_seam_cost_field(
        SeamCostField(
            field="combined",
            vertex_costs=np.array([0.1, 0.2, 1.0, 0.3], dtype=float),
            edge_costs=np.array([0.1, 0.3, 0.5, 0.8, 0.4, 0.7], dtype=float),
            edges=edges,
            samples_used=2,
            metadata={"cost_uniformity": 0.4},
        ),
        path,
    )


def _write_custom_costs(
    path: Path,
    edges: tuple[tuple[int, int], ...],
    edge_costs: list[float],
) -> None:
    vertex_count = max(max(edge) for edge in edges) + 1
    save_seam_cost_field(
        SeamCostField(
            field="combined",
            vertex_costs=np.linspace(0.1, 1.0, vertex_count, dtype=float),
            edge_costs=np.asarray(edge_costs, dtype=float),
            edges=edges,
            samples_used=2,
            metadata={"cost_uniformity": 0.4},
        ),
        path,
    )


def _write_seam_cost_receipt(
    path: Path,
    costs_path: Path,
    *,
    promotion: int = 1,
    vertex_count: int = 4,
    edge_count: int = 6,
) -> None:
    SeamCostReceipt(
        rom_field_receipt_hash="rom-field-receipt-sha256",
        body_receipt_hash="body-receipt-sha256",
        correspondence_receipt_hash=None,
        solve_domain="A_v3240",
        vertex_count=vertex_count,
        edge_count=edge_count,
        finite_cost_coverage=1.0,
        cost_uniformity=0.4,
        peak_cost=0.8,
        mean_cost=0.45,
        weight_vector={"w_P": 1.0},
        costs_hash=_sha256_file(costs_path),
        promotion=promotion,
        blocked_consumers=[],
    ).to_json(path)


def _run_solve(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, "scripts/solve_seams.py", *args],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_validate_cut_topology(*args: str) -> subprocess.CompletedProcess[str]:
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


def test_solve_seams_emits_promoted_receipt_from_promoted_costs(tmp_path: Path) -> None:
    mesh_path = tmp_path / "body.npz"
    costs_path = tmp_path / "seam_costs.npz"
    receipt_path = tmp_path / "seam_cost_receipt.json"
    out_dir = tmp_path / "out"
    solver_receipt_path = out_dir / "solver_promotion_receipt.json"

    edges = _write_mesh(mesh_path)
    _write_costs(costs_path, edges)
    _write_seam_cost_receipt(receipt_path, costs_path)

    result = _run_solve(
        "--seam-cost-receipt",
        str(receipt_path),
        "--costs",
        str(costs_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-solver-receipt",
        str(solver_receipt_path),
        "--anchor-count",
        "3",
        "--min-geodesic-separation",
        "0",
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(solver_receipt_path.read_text(encoding="utf-8"))
    assert receipt["seam_cost_receipt_hash"] == _sha256_file(receipt_path)
    assert receipt["solver_mode"] == "shortest_path"
    assert receipt["anchor_source"] == "field_minima"
    assert receipt["seam_edge_count"] > 0
    assert receipt["seam_vertex_count"] > 0
    assert receipt["panels_are_disks"]
    assert receipt["promotion"] == 1
    assert receipt["solver_blockers"] == []

    seam_edges_path = out_dir / "seam_edges.npz"
    assert receipt["seam_hash"] == _sha256_file(seam_edges_path)
    assert "seam_edges" in np.load(seam_edges_path)


def test_solve_seams_blocks_unpromoted_seam_cost_receipt(tmp_path: Path) -> None:
    mesh_path = tmp_path / "body.npz"
    costs_path = tmp_path / "seam_costs.npz"
    receipt_path = tmp_path / "seam_cost_receipt.json"
    out_dir = tmp_path / "out"

    edges = _write_mesh(mesh_path)
    _write_costs(costs_path, edges)
    _write_seam_cost_receipt(receipt_path, costs_path, promotion=0)

    result = _run_solve(
        "--seam-cost-receipt",
        str(receipt_path),
        "--costs",
        str(costs_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
    )

    assert result.returncode != 0
    assert "SeamCostReceipt not promoted" in result.stderr
    assert not out_dir.exists()


def test_solve_seams_blocks_zero_edge_solution(tmp_path: Path) -> None:
    mesh_path = tmp_path / "body.npz"
    costs_path = tmp_path / "seam_costs.npz"
    receipt_path = tmp_path / "seam_cost_receipt.json"
    out_dir = tmp_path / "out"
    solver_receipt_path = out_dir / "solver_promotion_receipt.json"

    edges = _write_mesh(mesh_path)
    _write_costs(costs_path, edges)
    _write_seam_cost_receipt(receipt_path, costs_path)

    result = _run_solve(
        "--seam-cost-receipt",
        str(receipt_path),
        "--costs",
        str(costs_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-solver-receipt",
        str(solver_receipt_path),
        "--anchor-count",
        "1",
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(solver_receipt_path.read_text(encoding="utf-8"))
    assert receipt["seam_edge_count"] == 0
    assert receipt["promotion"] == 0
    assert "insufficient_solver_anchors" in receipt["solver_blockers"]
    assert "insufficient_seam_edges" in receipt["solver_blockers"]
    assert "insufficient_seam_vertices" in receipt["solver_blockers"]
    assert "panel_unwrap" in receipt["blocked_consumers"]
    seam_edges = np.load(out_dir / "seam_edges.npz")["seam_edges"]
    assert seam_edges.shape == (0, 2)


def test_solve_seams_does_not_shrink_to_low_cost_anchor_component(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "body.npz"
    costs_path = tmp_path / "seam_costs.npz"
    receipt_path = tmp_path / "seam_cost_receipt.json"
    out_dir = tmp_path / "out"
    solver_receipt_path = out_dir / "solver_promotion_receipt.json"

    edges = _write_two_tetra_bridge_mesh(mesh_path)
    bridge_edges = {(0, 4), (1, 4)}
    _write_custom_costs(
        costs_path,
        edges,
        [100.0 if edge in bridge_edges else 0.1 for edge in edges],
    )
    _write_seam_cost_receipt(receipt_path, costs_path, vertex_count=8, edge_count=len(edges))

    result = _run_solve(
        "--seam-cost-receipt",
        str(receipt_path),
        "--costs",
        str(costs_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-solver-receipt",
        str(solver_receipt_path),
        "--anchor-source",
        "manual",
        "--manual-anchors",
        "0,1,4,5",
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(solver_receipt_path.read_text(encoding="utf-8"))
    assert receipt["anchor_count"] == 4
    assert receipt["candidate_anchor_count"] == 4
    assert receipt["low_cost_anchor_component_count"] > 1
    assert not receipt["anchor_fallback_used"]
    assert receipt["seam_edge_count"] > 0
    assert receipt["promotion"] == 1


def test_solve_seams_metric_panelization_emits_correction_payload(tmp_path: Path) -> None:
    mesh_path = tmp_path / "body.npz"
    costs_path = tmp_path / "seam_costs.npz"
    receipt_path = tmp_path / "seam_cost_receipt.json"
    out_dir = tmp_path / "out"
    solver_receipt_path = out_dir / "solver_promotion_receipt.json"

    edges = _write_grid_mesh(mesh_path, columns=4, rows=4)
    _write_custom_costs(costs_path, edges, [0.1 for _edge in edges])
    _write_seam_cost_receipt(
        receipt_path,
        costs_path,
        vertex_count=25,
        edge_count=len(edges),
    )

    result = _run_solve(
        "--seam-cost-receipt",
        str(receipt_path),
        "--costs",
        str(costs_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-solver-receipt",
        str(solver_receipt_path),
        "--solver-mode",
        "metric_panelization",
        "--target-panel-count",
        "2",
        "--min-panel-faces",
        "1",
        "--correction-families",
        "dart,relief_cut,ease,gusset,stretch_zone,variable_knit,pleat,bias_orientation",
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(solver_receipt_path.read_text(encoding="utf-8"))
    correction_path = out_dir / "corrections.json"
    correction_payload = json.loads(correction_path.read_text(encoding="utf-8"))
    assert receipt["solver_mode"] == "metric_panelization"
    assert receipt["correction_payload_hash"] == _sha256_file(correction_path)
    assert receipt["raw_residual_total"] >= receipt["corrected_residual_total"]
    assert receipt["selected_correction_count"] == correction_payload["selected_count"]
    assert correction_payload["variational_object"].startswith("(M,g_M")
    assert (out_dir / "seam_edges.npz").exists()


def test_solve_seams_rejects_cost_hash_mismatch(tmp_path: Path) -> None:
    mesh_path = tmp_path / "body.npz"
    costs_path = tmp_path / "seam_costs.npz"
    receipt_path = tmp_path / "seam_cost_receipt.json"

    edges = _write_mesh(mesh_path)
    _write_costs(costs_path, edges)
    _write_seam_cost_receipt(receipt_path, costs_path)
    _write_costs(costs_path, tuple(reversed(edges)))

    result = _run_solve(
        "--seam-cost-receipt",
        str(receipt_path),
        "--costs",
        str(costs_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(tmp_path / "out"),
    )

    assert result.returncode != 0
    assert "Seam costs hash does not match" in result.stderr


def test_solve_seams_cut_graph_promotes_two_triangle_square(tmp_path: Path) -> None:
    mesh_path = tmp_path / "square.npz"
    costs_path = tmp_path / "seam_costs.npz"
    receipt_path = tmp_path / "seam_cost_receipt.json"
    out_dir = tmp_path / "out"
    solver_receipt_path = out_dir / "solver_promotion_receipt.json"

    edges = _write_square_mesh(mesh_path)
    _write_custom_costs(costs_path, edges, [1.0 for _edge in edges])
    _write_seam_cost_receipt(receipt_path, costs_path, edge_count=len(edges))

    result = _run_solve(
        "--seam-cost-receipt",
        str(receipt_path),
        "--costs",
        str(costs_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-solver-receipt",
        str(solver_receipt_path),
        "--solver-mode",
        "cut_graph",
        "--target-panel-count",
        "2",
        "--min-panel-faces",
        "1",
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(solver_receipt_path.read_text(encoding="utf-8"))
    assert receipt["solver_mode"] == "cut_graph"
    assert receipt["promotion"] == 1
    assert receipt["panel_count"] == 2
    assert receipt["seam_edge_count"] == 1
    seam_edges = np.load(out_dir / "seam_edges.npz")["seam_edges"]
    assert {tuple(edge) for edge in seam_edges.tolist()} == {(0, 2)}

    cut_topology_receipt_path = out_dir / "cut_topology_receipt.json"
    validate_result = _run_validate_cut_topology(
        "--solver-receipt",
        str(solver_receipt_path),
        "--seam-edges",
        str(out_dir / "seam_edges.npz"),
        "--mesh",
        str(mesh_path),
        "--out-cut-topology-receipt",
        str(cut_topology_receipt_path),
    )

    assert validate_result.returncode == 0, validate_result.stderr
    cut_receipt = json.loads(cut_topology_receipt_path.read_text(encoding="utf-8"))
    assert cut_receipt["promotion"] == 1


def test_solve_seams_cut_graph_blocks_undersized_closed_mesh_panels(tmp_path: Path) -> None:
    mesh_path = tmp_path / "tetra.npz"
    costs_path = tmp_path / "seam_costs.npz"
    receipt_path = tmp_path / "seam_cost_receipt.json"
    out_dir = tmp_path / "out"
    solver_receipt_path = out_dir / "solver_promotion_receipt.json"

    edges = _write_mesh(mesh_path)
    _write_costs(costs_path, edges)
    _write_seam_cost_receipt(receipt_path, costs_path)

    result = _run_solve(
        "--seam-cost-receipt",
        str(receipt_path),
        "--costs",
        str(costs_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-solver-receipt",
        str(solver_receipt_path),
        "--solver-mode",
        "cut_graph",
        "--target-panel-count",
        "2",
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(solver_receipt_path.read_text(encoding="utf-8"))
    assert receipt["promotion"] == 0
    assert receipt["panel_count"] >= 2
    assert "undersized_cut_panel" in receipt["solver_blockers"]


def test_solve_seams_cut_graph_grid_emits_multiple_panel_boundaries(tmp_path: Path) -> None:
    mesh_path = tmp_path / "grid.npz"
    costs_path = tmp_path / "seam_costs.npz"
    receipt_path = tmp_path / "seam_cost_receipt.json"
    out_dir = tmp_path / "out"
    solver_receipt_path = out_dir / "solver_promotion_receipt.json"

    edges = _write_grid_mesh(mesh_path)
    _write_custom_costs(costs_path, edges, [1.0 for _edge in edges])
    _write_seam_cost_receipt(
        receipt_path,
        costs_path,
        vertex_count=25,
        edge_count=len(edges),
    )

    result = _run_solve(
        "--seam-cost-receipt",
        str(receipt_path),
        "--costs",
        str(costs_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-solver-receipt",
        str(solver_receipt_path),
        "--solver-mode",
        "cut_graph",
        "--target-panel-count",
        "4",
        "--min-panel-faces",
        "1",
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(solver_receipt_path.read_text(encoding="utf-8"))
    assert receipt["promotion"] == 1
    assert 2 <= receipt["panel_count"] <= 4
    assert receipt["seam_edge_count"] > 0


def test_solve_seams_cut_graph_ignores_invalid_dart_candidates(tmp_path: Path) -> None:
    mesh_path = tmp_path / "square.npz"
    costs_path = tmp_path / "seam_costs.npz"
    receipt_path = tmp_path / "seam_cost_receipt.json"
    candidates_path = tmp_path / "bad_candidates.json"
    out_dir = tmp_path / "out"
    solver_receipt_path = out_dir / "solver_promotion_receipt.json"

    edges = _write_square_mesh(mesh_path)
    _write_custom_costs(costs_path, edges, [1.0 for _edge in edges])
    _write_seam_cost_receipt(receipt_path, costs_path, edge_count=len(edges))
    candidates_path.write_text("{not-json", encoding="utf-8")

    result = _run_solve(
        "--seam-cost-receipt",
        str(receipt_path),
        "--costs",
        str(costs_path),
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--out-solver-receipt",
        str(solver_receipt_path),
        "--solver-mode",
        "cut_graph",
        "--target-panel-count",
        "2",
        "--min-panel-faces",
        "1",
        "--dart-relief-candidates",
        str(candidates_path),
    )

    assert result.returncode == 0, result.stderr
    assert "invalid_dart_relief_candidates" in result.stderr
    receipt = json.loads(solver_receipt_path.read_text(encoding="utf-8"))
    assert receipt["promotion"] == 1
