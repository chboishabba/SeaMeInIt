from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from smii.seams import PanelUnwrapReceipt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_panel_uvs(path: Path) -> None:
    np.savez_compressed(
        path,
        panel_0=np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.6, 0.7], [0.1, 0.8]],
            dtype=float,
        ),
    )


def _write_panel_receipt(path: Path, panel_uvs_path: Path) -> None:
    PanelUnwrapReceipt(
        solver_receipt_hash="solver-receipt-sha256",
        panel_count=1,
        panels_all_disks=True,
        per_panel_distortion=[0.18],
        worst_panel_distortion=0.18,
        mean_panel_distortion=0.18,
        distortion_threshold=0.05,
        subdivision_iterations=0,
        grain_directions=["weft"],
        uv_hash=_sha256_file(panel_uvs_path),
        seam_topology_hash="seam-sha256",
        promotion=0,
        blocked_consumers=[],
        unwrap_backend="bootstrap_projection",
        backend_is_bootstrap=True,
        distortion_margin=-0.13,
        panel_unwrap_blockers=["distortion_exceeds_threshold"],
    ).to_json(path)


def _write_mesh(path: Path) -> None:
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


def _write_seam_edges(path: Path) -> None:
    np.savez_compressed(path, seam_edges=np.array([[0, 1], [1, 3]], dtype=int))


def _write_corrections(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "families": ["dart", "relief_cut", "ease"],
                "panels": [
                    {
                        "panel_label": 0,
                        "raw_metric_residual": 0.22,
                        "corrected_metric_residual": 0.08,
                    }
                ],
                "selected_corrections": [
                    {
                        "panel_label": 0,
                        "family": "dart",
                        "selected": True,
                    },
                    {
                        "panel_label": 0,
                        "family": "ease",
                        "selected": True,
                    },
                ],
                "selected_count": 2,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_render(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, "scripts/render_panel_patterns.py", *args],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_render_panel_patterns_emits_diagnostic_visuals_for_blocked_receipt(
    tmp_path: Path,
) -> None:
    panel_uvs_path = tmp_path / "panel_uvs.npz"
    panel_receipt_path = tmp_path / "panel_unwrap_receipt.json"
    out_dir = tmp_path / "diagnostics"
    _write_panel_uvs(panel_uvs_path)
    _write_panel_receipt(panel_receipt_path, panel_uvs_path)

    result = _run_render(
        "--panel-receipt",
        str(panel_receipt_path),
        "--panel-uvs",
        str(panel_uvs_path),
        "--out-dir",
        str(out_dir),
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "panel_uv_diagnostic.svg").exists()
    assert (out_dir / "panel_uv_diagnostic.png").exists()
    assert (out_dir / "diagnostic_2d_patterns.svg").exists()
    summary = json.loads((out_dir / "diagnostic_pattern_summary.json").read_text("utf-8"))
    assert summary["diagnostic_only"] is True
    assert summary["manufacturing_authorized"] is False
    assert summary["panel_unwrap_promotion"] == 0
    assert summary["panel_unwrap_blockers"] == ["distortion_exceeds_threshold"]
    assert summary["output_roles"]["uv_svg"] == "raw_uv_point_cloud_not_panel_topology"
    assert summary["output_roles"]["uv_png"] == "raw_uv_point_cloud_not_panel_topology"
    assert summary["output_roles"]["patterns_svg"] == "legacy_coarse_hull_preview_deprecated"
    assert summary["visual_review_guidance"] == {
        "primary_panel_review_artifact": None,
        "raw_uv_diagnostic_is_panel_topology": False,
        "raw_uv_diagnostic_note": (
            "panel_uv_diagnostic shows sampled UV coordinates and convex hulls only; "
            "it intentionally does not prove or display face-backed panel topology."
        ),
    }
    assert {
        "name": "panel_uv_diagnostic.svg",
        "role": "raw_uv_point_cloud_not_panel_topology",
        "primary": False,
        "topology_backed": False,
        "path": str(out_dir / "panel_uv_diagnostic.svg"),
    } in summary["artifact_hierarchy"]
    assert {
        "name": "diagnostic_2d_patterns.svg",
        "role": "legacy_coarse_hull_preview",
        "primary": False,
        "deprecated": True,
        "path": str(out_dir / "diagnostic_2d_patterns.svg"),
    } in summary["artifact_hierarchy"]
    pattern_svg = (out_dir / "diagnostic_2d_patterns.svg").read_text("utf-8")
    assert "Convex-hull previews only" in pattern_svg
    assert 'data-diagnostic="true"' in pattern_svg
    uv_svg = (out_dir / "panel_uv_diagnostic.svg").read_text("utf-8")
    assert "Raw panel UV point-cloud diagnostic" in uv_svg
    assert "not the face-backed panel topology" in uv_svg
    assert "diagnostic_flat_cut_sheet.svg" in uv_svg


def test_render_panel_patterns_rejects_uv_hash_mismatch(tmp_path: Path) -> None:
    panel_uvs_path = tmp_path / "panel_uvs.npz"
    panel_receipt_path = tmp_path / "panel_unwrap_receipt.json"
    _write_panel_uvs(panel_uvs_path)
    _write_panel_receipt(panel_receipt_path, panel_uvs_path)
    np.savez_compressed(panel_uvs_path, panel_0=np.zeros((3, 2), dtype=float))

    result = _run_render(
        "--panel-receipt",
        str(panel_receipt_path),
        "--panel-uvs",
        str(panel_uvs_path),
        "--out-dir",
        str(tmp_path / "diagnostics"),
    )

    assert result.returncode != 0
    assert "Panel UV hash does not match" in result.stderr


def test_render_panel_patterns_emits_3d_mesh_overlay(tmp_path: Path) -> None:
    panel_uvs_path = tmp_path / "panel_uvs.npz"
    panel_receipt_path = tmp_path / "panel_unwrap_receipt.json"
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    out_dir = tmp_path / "diagnostics"
    _write_panel_uvs(panel_uvs_path)
    _write_panel_receipt(panel_receipt_path, panel_uvs_path)
    _write_mesh(mesh_path)
    _write_seam_edges(seam_edges_path)

    result = _run_render(
        "--panel-receipt",
        str(panel_receipt_path),
        "--panel-uvs",
        str(panel_uvs_path),
        "--out-dir",
        str(out_dir),
        "--mesh",
        str(mesh_path),
        "--seam-edges",
        str(seam_edges_path),
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "mesh_seam_overlay.png").exists()
    assert (out_dir / "diagnostic_flat_cut_sheet.svg").exists()
    summary = json.loads((out_dir / "diagnostic_pattern_summary.json").read_text("utf-8"))
    assert summary["outputs"]["mesh_overlay_png"] == str(out_dir / "mesh_seam_overlay.png")
    assert summary["outputs"]["cut_sheet_svg"] == str(out_dir / "diagnostic_flat_cut_sheet.svg")
    assert summary["visual_review_guidance"]["primary_panel_review_artifact"] == str(
        out_dir / "diagnostic_flat_cut_sheet.svg"
    )
    assert summary["visual_review_guidance"]["raw_uv_diagnostic_is_panel_topology"] is False
    assert summary["mesh_overlay"]["vertex_count"] == 4
    assert summary["mesh_overlay"]["face_count"] == 4
    assert summary["mesh_overlay"]["seam_edge_count"] == 2
    assert summary["cut_sheet"]["panel_face_counts"] == [4]
    assert summary["cut_sheet"]["panel_cut_edge_counts"] == [2]
    assert summary["cut_sheet"]["panel_seam_segment_counts"] == [2]
    assert summary["cut_sheet"]["seam_graph_summary"] == {
        "branch_vertex_count": 0,
        "connected_component_count": 1,
        "edge_segment_count": 2,
        "endpoint_count": 2,
        "largest_component_edge_count": 2,
        "vertex_count": 3,
    }
    assert summary["cut_sheet"]["cut_sheet_warnings"] == [
        "panel_unwrap_not_promoted",
        "single_panel_cut_sheet",
        "seam_graph_not_cut_graph",
        "no_patch_boundary_edges",
        "no_cut_mesh_boundary",
        "open_or_branched_seam_graph",
    ]
    assert summary["panels"] == [
        {
            "boundary_edge_count": 0,
            "corrected_metric_residual": None,
            "face_count": 4,
            "panel_label": 0,
            "raw_metric_residual": None,
            "raw_uv_distortion": 0.18,
            "seam_segment_count": 2,
            "selected_backend": "bootstrap_projection",
            "selected_correction_families": [],
            "serialization_candidates": [],
        }
    ]
    assert summary["artifact_hierarchy"][3]["name"] == "diagnostic_flat_cut_sheet.svg"
    assert summary["artifact_hierarchy"][3]["primary"] is True
    assert summary["artifact_hierarchy"][3]["topology_backed"] is True
    assert summary["artifact_hierarchy"][4]["deprecated"] is True
    cut_sheet = (out_dir / "diagnostic_flat_cut_sheet.svg").read_text("utf-8")
    assert "Diagnostic flat cut sheet" in cut_sheet
    assert 'data-face-count="4"' in cut_sheet
    assert "seam segments 2" in cut_sheet
    assert 'class="seam-segment"' in cut_sheet


def test_render_panel_patterns_summarizes_metric_panelization_provenance(
    tmp_path: Path,
) -> None:
    panel_uvs_path = tmp_path / "panel_uvs.npz"
    panel_receipt_path = tmp_path / "panel_unwrap_receipt.json"
    mesh_path = tmp_path / "body.npz"
    seam_edges_path = tmp_path / "seam_edges.npz"
    corrections_path = tmp_path / "corrections.json"
    out_dir = tmp_path / "diagnostics"
    _write_panel_uvs(panel_uvs_path)
    _write_corrections(corrections_path)
    PanelUnwrapReceipt(
        solver_receipt_hash="solver-receipt-sha256",
        panel_count=1,
        panels_all_disks=True,
        per_panel_distortion=[0.18],
        worst_panel_distortion=0.18,
        mean_panel_distortion=0.18,
        distortion_threshold=0.05,
        subdivision_iterations=0,
        grain_directions=["weft"],
        uv_hash=_sha256_file(panel_uvs_path),
        seam_topology_hash="seam-sha256",
        promotion=0,
        blocked_consumers=[],
        unwrap_backend="bootstrap_projection",
        backend_is_bootstrap=True,
        distortion_margin=-0.13,
        panel_unwrap_blockers=[
            "distortion_exceeds_threshold",
            "open_or_branched_seam_graph",
            "seam_graph_not_cut_graph",
            "no_cut_mesh_boundary",
        ],
        per_panel_corrected_residual=[0.08],
        worst_corrected_residual=0.08,
        mean_corrected_residual=0.08,
        correction_payload_hash=_sha256_file(corrections_path),
        correction_tree_receipt={
            "schema_version": "smii.correction_tree_receipt.v1",
            "promotion": 1,
            "branch_count": 1,
            "typed_branch_count": 1,
            "diagnostic_branch_count": 0,
        },
        realized_correction_operator_receipt={
            "schema_version": "smii.realized_correction_operator.v1",
            "promotion": 1,
            "realized_operator_count": 1,
            "operator_families": ["stretch_zone"],
            "nodes": [
                {
                    "branch_id": "branch_000",
                    "branch_vertex": 0,
                    "operator": "stretch_zone",
                    "realized": True,
                    "pattern_annotation": "stretch_zone",
                }
            ],
        },
        correction_tree_materialization_receipt={
            "schema_version": "smii.correction_tree_materialization.v1",
            "promotion": 1,
            "materialized_operator_count": 1,
            "blockers": [],
            "materializations": [
                {
                    "node_id": "branch_000",
                    "operator_family": "stretch_zone",
                    "metric_realized": True,
                    "chart_materialized": True,
                    "materialization_kind": "backend_hint",
                    "affected_panels": [0],
                    "backend_constraints_emitted": True,
                    "promotion": 1,
                    "blockers": [],
                }
            ],
        },
    ).to_json(panel_receipt_path)
    _write_mesh(mesh_path)
    _write_seam_edges(seam_edges_path)

    result = _run_render(
        "--panel-receipt",
        str(panel_receipt_path),
        "--panel-uvs",
        str(panel_uvs_path),
        "--out-dir",
        str(out_dir),
        "--mesh",
        str(mesh_path),
        "--seam-edges",
        str(seam_edges_path),
        "--corrections",
        str(corrections_path),
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((out_dir / "diagnostic_pattern_summary.json").read_text("utf-8"))
    assert summary["manufacturing_authorized"] is False
    assert summary["provenance"] == {
        "correction_source": {
            "families": ["dart", "relief_cut", "ease"],
            "hash": _sha256_file(corrections_path),
            "path": str(corrections_path),
            "selected_count": 2,
        },
        "manufacturing_authorized": False,
        "manufacturing_blocked_because": [
            "distortion_exceeds_threshold",
            "no_cut_mesh_boundary",
            "no_patch_boundary_edges",
            "open_or_branched_seam_graph",
            "panel_unwrap_not_promoted",
            "seam_graph_not_cut_graph",
            "single_panel_cut_sheet",
        ],
        "selected_backend_per_panel": [],
        "solver_mode": "metric_panelization",
        "topology_source": "mesh_components_after_solver_seams",
        "unwrap_backend": "bootstrap_projection",
    }
    assert summary["panels"][0] == {
        "boundary_edge_count": 0,
        "corrected_metric_residual": 0.08,
        "face_count": 4,
        "panel_label": 0,
        "raw_metric_residual": 0.22,
        "raw_uv_distortion": 0.18,
        "seam_segment_count": 2,
        "selected_backend": "bootstrap_projection",
        "selected_correction_families": ["dart", "ease"],
        "serialization_candidates": [],
    }
    assert summary["realized_correction_operator_receipt"]["operator_families"] == ["stretch_zone"]
    assert summary["correction_tree_materialization"]["status"] == "materialized"
    assert summary["correction_tree_materialization"]["chart_materialized_operator_count"] == 1
    assert (
        summary["cut_sheet"]["realized_correction_operator_receipt"]["realized_operator_count"] == 1
    )
    cut_sheet = (out_dir / "diagnostic_flat_cut_sheet.svg").read_text("utf-8")
    assert 'data-correction-tree-status="materialized"' in cut_sheet
    assert 'data-role="realized_correction_operator"' in cut_sheet
    assert "stretch_zone" in cut_sheet
