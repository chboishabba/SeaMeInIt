from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path("scripts/run_p3_afflec_gate0_diagnostics.py")
    spec = importlib.util.spec_from_file_location("run_p3_afflec_gate0_diagnostics", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_reference_images(root: Path) -> None:
    root.mkdir(parents=True)
    for name in (
        "a.jpg",
        "b.webp",
        "Screenshot_20260604_135454.png",
        "images.jpg",
    ):
        (root / name).write_bytes(b"stub")


def test_build_lanes_creates_all_and_curated_raw_refined_specs(tmp_path: Path) -> None:
    module = _load_module()
    ref_root = tmp_path / "refs"
    _write_reference_images(ref_root)

    lanes = module.build_lanes(
        reference_root=ref_root,
        output_root=tmp_path / "out",
        python="python",
    )

    assert [lane.name for lane in lanes] == [
        "all_refs_raw",
        "all_refs_refined",
        "curated_refs_raw",
        "curated_refs_refined",
    ]
    assert len(lanes[0].image_paths) == 4
    assert len(lanes[2].image_paths) == 2
    assert "--skip-measurement-refinement" in lanes[0].command
    assert "--skip-measurement-refinement" not in lanes[1].command
    assert "--require-high-trust-detector" in lanes[0].command
    assert lanes[0].command[lanes[0].command.index("--detector") + 1] == "mediapipe"


def test_write_reports_records_threshold_policy(tmp_path: Path) -> None:
    module = _load_module()
    summary = {
        "name": "all_refs_raw",
        "image_count": 4,
        "diagnostics": {
            "status": "WARN",
            "flags": ["WARN:low_view_diversity"],
        },
        "body_receipt": {
            "promotion": 0,
            "skull_rigidity_residual": 0.36,
        },
        "meshes": {
            "final_export": {"crown_eccentricity_residual": 0.36},
            "raw_reprojection": {"crown_eccentricity_residual": 0.25},
            "refined_pre_repair": {"crown_eccentricity_residual": 0.35},
        },
    }

    json_path, markdown_path = module.write_reports(
        output_root=tmp_path / "out",
        reference_root=tmp_path / "refs",
        lane_summaries=[summary],
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert payload["policy"]["threshold_changes"] == "diagnostic_only_first"
    assert "all_refs_raw" in markdown
    assert "Do not loosen the skull residual threshold" in markdown
