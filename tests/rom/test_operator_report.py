import importlib.util
import json
from pathlib import Path

import numpy as np

from smii.rom.basis import KernelBasis
from smii.rom.seam_costs import SeamCostField, save_seam_cost_field


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_rom_operator_report_generates_html_and_manifest(tmp_path):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "render_rom_operator_report.py"
    module = _load_script_module(script_path, "render_rom_operator_report")

    basis = KernelBasis.from_arrays(
        np.eye(2),
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        source_mesh="demo",
        normalization="qr-orthonormalized",
    )
    basis_path = tmp_path / "basis.npz"
    np.savez_compressed(
        basis_path,
        basis=basis.matrix,
        vertices=basis.vertices,
        meta={"source_mesh": "demo", "normalization": "qr-orthonormalized"},
    )

    meta_path = tmp_path / "rom_meta.json"
    meta_path.write_text(
        json.dumps({"meta": {"vertex_count": 2, "costs_path": "costs.npz"}, "pose_ids": ["a", "b"]}),
        encoding="utf-8",
    )
    coeff_path = tmp_path / "coeffs.json"
    coeff_path.write_text(
        json.dumps(
            {
                "meta": {"field_name": "seam_sensitivity"},
                "samples": [
                    {"pose_id": "a", "weight": 1.0, "coeffs": {"seam_sensitivity": [0.5, 1.5]}, "metadata": {}},
                    {"pose_id": "b", "weight": 0.5, "coeffs": {"seam_sensitivity": [1.0, 0.0]}, "metadata": {}},
                ],
            }
        ),
        encoding="utf-8",
    )
    cert_path = tmp_path / "certificate.json"
    cert_path.write_text(
        json.dumps({"status": "PASS", "rank_correlation": 0.99}),
        encoding="utf-8",
    )
    costs = SeamCostField(
        field="rom_fd_sensitivity",
        vertex_costs=np.array([0.2, 0.8], dtype=float),
        edge_costs=np.zeros(0, dtype=float),
        edges=tuple(),
        samples_used=2,
        metadata={},
    )
    costs_path = tmp_path / "costs.npz"
    save_seam_cost_field(costs, costs_path)
    sample_dir = tmp_path / "rom_samples"
    sample_dir.mkdir()
    sample_mesh = sample_dir / "sample_01__pose_a.npz"
    np.savez_compressed(
        sample_mesh,
        vertices=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        faces=np.array([[0, 1, 1]], dtype=int),
    )
    sample_manifest = sample_dir / "rom_sample_manifest.json"
    sample_manifest.write_text(
        json.dumps(
            {
                "meta": {"source_vertex_count": 2},
                "samples": [
                    {
                        "pose_id": "pose_a",
                        "selection_reasons": ["max_field_l2_norm"],
                        "field_l2_norm": 3.0,
                        "displacement_mean_norm": 1.2,
                        "mesh_name": sample_mesh.name,
                        "mesh_path": str(sample_mesh),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "report"
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "overlay.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' width='8' height='8'></svg>",
        encoding="utf-8",
    )
    (media_dir / "orbit.webm").write_bytes(b"webm")
    module.main(
        [
            "--basis",
            str(basis_path),
            "--rom-meta",
            str(meta_path),
            "--coeff-samples",
            str(coeff_path),
            "--sample-manifest",
            str(sample_manifest),
            "--certificate",
            str(cert_path),
            "--costs",
            str(costs_path),
            "--media-path",
            str(media_dir),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert (out_dir / "index.html").exists()
    assert (out_dir / "report_manifest.json").exists()
    assert (out_dir / "coeff_summary.json").exists()
    assert not (out_dir / "coeff_top_variance.png").exists()
    assert not (out_dir / "coeff_sample_norms.png").exists()

    manifest = json.loads((out_dir / "report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"]["coefficient_samples_available"] is True
    assert manifest["summary"]["consistency_status"] == "PASS"
    assert any(entry["artifact_level"] == "operator" for entry in manifest["artifacts"])
    assert any(entry["artifact_level"] == "topology" for entry in manifest["artifacts"])
    assert len(manifest["embedded_media"]) == 2
    assert manifest["sample_morphology"]["samples"][0]["pose_id"] == "pose_a"
    html_text = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "<svg" in html_text
    assert "<img " in html_text
    assert "<video" in html_text
    assert "Representative ROM Sample Morphologies" in html_text
    assert "sample_01__pose_a.npz" in html_text


def test_render_rom_operator_report_flags_topology_mismatch(tmp_path):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "render_rom_operator_report.py"
    module = _load_script_module(script_path, "render_rom_operator_report_mismatch")

    basis = KernelBasis.from_arrays(
        np.eye(2),
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        source_mesh="demo",
        normalization="qr-orthonormalized",
    )
    basis_path = tmp_path / "basis.npz"
    np.savez_compressed(
        basis_path,
        basis=basis.matrix,
        vertices=basis.vertices,
        meta={"source_mesh": "demo", "normalization": "qr-orthonormalized"},
    )
    meta_path = tmp_path / "rom_meta.json"
    meta_path.write_text(json.dumps({"meta": {"vertex_count": 2}}), encoding="utf-8")
    costs = SeamCostField(
        field="rom_fd_sensitivity",
        vertex_costs=np.array([0.1, 0.2, 0.3], dtype=float),
        edge_costs=np.zeros(0, dtype=float),
        edges=tuple(),
        samples_used=1,
        metadata={},
    )
    costs_path = tmp_path / "costs.npz"
    save_seam_cost_field(costs, costs_path)
    body_path = tmp_path / "body.npz"
    np.savez_compressed(body_path, vertices=np.zeros((4, 3), dtype=float))

    out_dir = tmp_path / "report"
    module.main(
        [
            "--basis",
            str(basis_path),
            "--rom-meta",
            str(meta_path),
            "--costs",
            str(costs_path),
            "--body",
            str(body_path),
            "--out-dir",
            str(out_dir),
        ]
    )

    manifest = json.loads((out_dir / "report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"]["consistency_status"] == "WARN"
    assert any("Cost-field vertex count" in flag for flag in manifest["summary"]["consistency_flags"])
    assert any("Body vertex count" in flag for flag in manifest["summary"]["consistency_flags"])
    topology_entries = [entry for entry in manifest["artifacts"] if entry["artifact_level"] == "topology"]
    assert {entry["topology"] for entry in topology_entries} == {"v3", "v4"}
