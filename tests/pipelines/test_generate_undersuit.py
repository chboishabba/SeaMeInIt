import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import importlib

suit_pkg = importlib.import_module("suit")
if not hasattr(suit_pkg, "UnderSuitGenerator"):
    undersuit_mod = importlib.import_module("suit.undersuit_generator")
    suit_pkg.UnderSuitGenerator = undersuit_mod.UnderSuitGenerator
    suit_pkg.UnderSuitOptions = undersuit_mod.UnderSuitOptions

import smii.pipelines.generate_undersuit as cli


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cli_exports_pattern_files(tmp_path: Path) -> None:
    body_path = tmp_path / "body.json"
    base_vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    base_faces = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ]
    _write_json(body_path, {"vertices": base_vertices, "faces": base_faces})

    panels_path = tmp_path / "panels.json"
    _write_json(
        panels_path,
        {
            "panels": [
                {
                    "name": "torso_front",
                    "indices": [0, 1, 2],
                }
            ]
        },
    )

    seams_path = tmp_path / "seams.json"
    _write_json(seams_path, {"panels": {"torso_front": {"seam_allowance": 0.015}}})

    output_dir = tmp_path / "undersuit"

    exit_code = cli.main(
        [
            str(body_path),
            "--output",
            str(output_dir),
            "--panels-json",
            str(panels_path),
            "--seams-json",
            str(seams_path),
        ]
    )

    assert exit_code == 0

    pattern_dir = output_dir / "patterns"
    for fmt in ("svg", "pdf", "dxf"):
        path = pattern_dir / f"undersuit_pattern.{fmt}"
        assert path.exists(), f"Expected pattern file {path} to be generated"

    metadata_path = output_dir / "metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "patterns" in metadata

    pattern_meta = metadata["patterns"]
    assert pattern_meta["panel_source"] == str(panels_path)
    assert pattern_meta["seam_source"] == str(seams_path)
    assert pattern_meta["panels"] == ["torso_front"]
    assert set(pattern_meta["files"].keys()) == {"svg", "pdf", "dxf"}
    for path in pattern_meta["files"].values():
        assert Path(path).exists()
