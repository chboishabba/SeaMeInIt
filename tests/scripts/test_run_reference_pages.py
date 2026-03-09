import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_run_reference_embeds_completed_assets_and_ignores_frames(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "render_run_reference.py"
    module = _load_script_module(script_path, "render_run_reference")

    run_root = tmp_path / "run"
    run_root.mkdir()
    body_dir = run_root / "body_raw"
    body_dir.mkdir()
    np.savez(body_dir / "afflec_body.npz", vertices=np.zeros((3, 3), dtype=float), faces=np.array([[0, 1, 2]], dtype=int))
    (body_dir / "afflec_fit_diagnostics.json").write_text("{}", encoding="utf-8")
    (body_dir / "measurement_report.png").write_bytes(b"png")

    rom_dir = run_root / "rom_raw"
    rom_dir.mkdir()
    (rom_dir / "rom_run.json").write_text("{}", encoding="utf-8")
    operator_dir = rom_dir / "operator_report"
    operator_dir.mkdir()
    (operator_dir / "index.html").write_text("<html>operator</html>", encoding="utf-8")
    (operator_dir / "coeff_top_variance.png").write_bytes(b"legacy")

    seam_dir = run_root / "seams_raw"
    seam_dir.mkdir()
    (seam_dir / "seam_report.json").write_text("{}", encoding="utf-8")
    (seam_dir / "overlay.png").write_bytes(b"overlay")
    (seam_dir / "orbit.webm").write_bytes(b"webm")
    frames_dir = seam_dir / "demo_orbit_frames"
    frames_dir.mkdir()
    (frames_dir / "frame_000.png").write_bytes(b"frame")

    out_dir = run_root / "run_reference"
    module.main(["--run-root", str(run_root), "--out-dir", str(out_dir)])

    manifest = json.loads((out_dir / "run_report_manifest.json").read_text(encoding="utf-8"))
    media_names = {entry["name"] for entry in manifest["sections"]["media"]}
    assert "overlay.png" in media_names
    assert "orbit.webm" in media_names
    assert "measurement_report.png" in media_names
    assert "coeff_top_variance.png" not in media_names
    assert "frame_000.png" not in media_names

    subpage_names = {entry["parent"] for entry in manifest["sections"]["subpages"]}
    assert "operator_report" in subpage_names

    html_text = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "Run Reference:" in html_text
    assert "Media Gallery" in html_text
    assert "operator_report/index.html" in html_text


def test_render_run_index_lists_runs_with_reference_pages(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "render_run_index.py"
    module = _load_script_module(script_path, "render_run_index")

    comparisons_root = tmp_path / "comparisons"
    bundles_root = tmp_path / "assets_bundles"
    comparisons_root.mkdir()
    bundles_root.mkdir()
    run_a = comparisons_root / "20260309_120000__compare_a"
    run_b = bundles_root / "20260309_130000__bundle_b"
    (run_a / "run_reference").mkdir(parents=True)
    (run_b / "run_reference").mkdir(parents=True)
    (run_a / "run_reference" / "index.html").write_text("<html>a</html>", encoding="utf-8")
    (run_a / "run_reference" / "run_report_manifest.json").write_text(
        json.dumps({"summary": {"media_count": 3, "body_count": 1, "rom_artifact_count": 2, "seam_report_count": 1}}),
        encoding="utf-8",
    )
    (run_b / "placeholder.txt").write_text("x", encoding="utf-8")

    out_path = tmp_path / "outputs_index.html"
    module.main(["--runs-root", str(comparisons_root), "--runs-root", str(bundles_root), "--out", str(out_path)])

    assert out_path.exists()
    payload = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["run_count"] == 2
    has_report = {entry["name"]: entry["has_report"] for entry in payload["runs"]}
    assert has_report == {"20260309_130000__bundle_b": False, "20260309_120000__compare_a": True}
    run_types = {entry["name"]: entry["run_type"] for entry in payload["runs"]}
    assert run_types == {
        "20260309_130000__bundle_b": "bundle",
        "20260309_120000__compare_a": "comparison",
    }

    html_text = out_path.read_text(encoding="utf-8")
    assert "2026-03-09 13:00:00" in html_text
    assert "bundle" in html_text
    assert "comparison" in html_text
    assert "20260309_120000__compare_a" in html_text
    assert "run_reference/index.html" in html_text
