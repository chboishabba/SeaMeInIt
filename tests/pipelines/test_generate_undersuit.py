import json
import math
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


def _build_cylinder_body() -> tuple[list[list[float]], list[list[int]]]:
    segments = 12
    levels = [0.0, 0.28, 0.56, 0.82]
    radii = [0.22, 0.28, 0.25, 0.2]
    vertices: list[list[float]] = []
    for z, radius in zip(levels, radii):
        for segment in range(segments):
            theta = (2.0 * math.pi * segment) / segments
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            vertices.append([x, y, z])
    faces: list[list[int]] = []
    for level in range(len(levels) - 1):
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            lower = level * segments
            upper = (level + 1) * segments
            a = lower + segment
            b = lower + next_segment
            c = upper + segment
            d = upper + next_segment
            faces.append([a, b, c])
            faces.append([b, d, c])
    return vertices, faces


def test_cli_exports_pattern_files(tmp_path: Path) -> None:
    body_path = tmp_path / "body.json"
    base_vertices, base_faces = _build_cylinder_body()
    _write_json(body_path, {"vertices": base_vertices, "faces": base_faces})

    joint_map_path = tmp_path / "joints.json"
    _write_json(
        joint_map_path,
        {
            "pelvis": [0.0, 0.0, 0.0],
            "neck": [0.0, 0.0, 0.9],
            "left_shoulder": [-0.2, 0.25, 0.65],
            "right_shoulder": [0.2, 0.25, 0.65],
        },
    )

    output_dir = tmp_path / "undersuit"

    exit_code = cli.main(
        [
            str(body_path),
            "--output",
            str(output_dir),
            "--joint-map",
            str(joint_map_path),
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
    assert pattern_meta["panel_count"] > 0
    assert sorted(pattern_meta["files"].keys()) == ["dxf", "pdf", "svg"]
    for path in pattern_meta["files"].values():
        assert Path(path).exists()
    assert "measurement_loops" in pattern_meta
    assert "seams" in pattern_meta
