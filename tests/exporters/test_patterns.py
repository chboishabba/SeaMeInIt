"""Tests for the undersuit pattern exporter."""

from __future__ import annotations

import builtins
import math
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from exporters.lscm_backend import LSCMConformalBackend
from exporters.patterns import (
    FoldAnnotation,
    GrainlineAnnotation,
    LabelAnnotation,
    NotchAnnotation,
    Panel2D,
    Panel3D,
    PatternExporter,
    build_panel_annotations,
    LSCMUnwrapBackend,
)
from suit import NEOPRENE_DEFAULT_BUDGETS


class DummyBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []

    def flatten_panels(  # noqa: D401 - simple stub
        self,
        panels,
        seams,
        *,
        scale,
        seam_allowance,
        budgets=None,
    ):
        self.calls.append((scale, seam_allowance))
        outline = [
            (0.0, 0.0),
            (0.5 * scale, 0.0),
            (0.5 * scale, 1.0 * scale),
            (0.0, 1.0 * scale),
        ]
        grain = GrainlineAnnotation(
            origin=(0.25 * scale, 0.5 * scale),
            direction=(0.0, 1.0),
            length=1.0 * scale,
        )
        notch = NotchAnnotation(
            position=(0.5 * scale, 0.0),
            tangent=(1.0, 0.0),
            normal=(0.0, -1.0),
            depth=0.03 * scale,
            width=0.05 * scale,
            label="CF",
        )
        fold = FoldAnnotation(
            start=(0.0, 0.5 * scale),
            end=(0.5 * scale, 0.5 * scale),
            kind="valley",
        )
        label = LabelAnnotation(text="Test Panel", position=(0.25 * scale, 0.75 * scale))
        return [
            Panel2D(
                name="test_panel",
                outline=outline,
                seam_outline=outline,
                seam_allowance=0.012,
                metadata={"source": "dummy"},
                grainlines=[grain],
                notches=[notch],
                folds=[fold],
                label=label,
            )
        ]


def test_panel2d_generates_cut_outline() -> None:
    seam_outline = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    panel = Panel2D(name="square", seam_outline=seam_outline, seam_allowance=0.1)

    assert panel.cut_outline
    assert len(panel.cut_outline) == len(seam_outline)
    expected = [(-0.1, -0.1), (1.1, -0.1), (1.1, 1.1), (-0.1, 1.1)]
    for actual, target in zip(panel.cut_outline, expected):
        assert actual == pytest.approx(target, rel=1e-6)


def test_panel2d_offsets_guard_for_degenerate_geometry() -> None:
    panel = Panel2D(
        name="line",
        seam_outline=[(0.0, 0.0), (1.0, 0.0)],
        seam_allowance=0.2,
    )

    assert panel.cut_outline == []
    warnings = panel.metadata.get("warnings", [])
    assert any("sufficient geometry" in warning for warning in warnings)


@pytest.fixture()
def mesh_payload() -> dict:
    return {
        "panels": [
            {
                "name": "test_panel",
                "vertices": [
                    (0.0, 0.0, 0.0),
                    (0.5, 0.0, 0.0),
                    (0.5, 1.0, 0.0),
                ],
                "faces": [(0, 1, 2)],
            }
        ]
    }


@pytest.fixture()
def seam_payload() -> dict:
    return {"test_panel": {"seam_allowance": 0.012}}


def test_export_creates_requested_formats(
    tmp_path: Path, mesh_payload: dict, seam_payload: dict
) -> None:
    backend = DummyBackend()
    exporter = PatternExporter(backend=backend, scale=0.95, seam_allowance=0.01)

    created = exporter.export(
        mesh_payload,
        seam_payload,
        output_dir=tmp_path,
        formats=["pdf", "svg", "dxf"],
        metadata={"variant": "test"},
    )

    assert backend.calls == [(0.95, 0.01)]
    assert set(created) == {"pdf", "svg", "dxf"}
    for fmt, path in created.items():
        assert path.exists(), f"Expected {fmt} output to exist"


def test_export_writes_metadata_and_annotations(
    tmp_path: Path, mesh_payload: dict, seam_payload: dict
) -> None:
    backend = DummyBackend()
    exporter = PatternExporter(backend=backend, scale=1.1, seam_allowance=0.02)

    created = exporter.export(
        mesh_payload,
        seam_payload,
        output_dir=tmp_path,
        formats=["svg", "dxf", "pdf"],
    )

    svg_text = created["svg"].read_text(encoding="utf-8")
    assert "Scale: 1.1" in svg_text
    assert 'class="panel-outline"' in svg_text
    assert "panel-grain" in svg_text
    assert "panel-notch" in svg_text
    assert ">Test Panel<" in svg_text

    dxf_text = created["dxf"].read_text(encoding="utf-8")
    assert "$SMII_SCALE" in dxf_text
    assert "_GRAIN" in dxf_text
    assert "_NOTCH" in dxf_text
    assert "Test Panel" in dxf_text

    pdf_bytes = created["pdf"].read_bytes()
    assert b"Scale: 1.1" in pdf_bytes
    assert b"Test Panel" in pdf_bytes
    assert b"CF" in pdf_bytes
    assert 'data-seam-allowance="0.012"' in svg_text
    assert 'class="seam-outline"' in svg_text
    assert 'class="cut-outline"' in svg_text

    dxf_text = created["dxf"].read_text(encoding="utf-8")
    assert "$SMII_SCALE" in dxf_text
    assert "1.1" in dxf_text
    assert "0.0120" in dxf_text
    assert "test_panel_CUT" in dxf_text

    pdf_bytes = created["pdf"].read_bytes()
    assert b"Scale: 1.1" in pdf_bytes
    assert b"test_panel seam:" in pdf_bytes
    assert b"test_panel cut:" in pdf_bytes


def test_pattern_exporter_tiles_pdf(tmp_path: Path) -> None:
    exporter = PatternExporter()
    mesh_payload = {
        "panels": [
            {
                "name": "tile_panel",
                "vertices": [
                    (0.0, 0.0, 0.0),
                    (0.3, 0.0, 0.0),
                    (0.3, 0.1, 0.0),
                    (0.0, 0.1, 0.0),
                ],
                "faces": [(0, 1, 2), (0, 2, 3)],
            }
        ]
    }

    created = exporter.export(mesh_payload, None, output_dir=tmp_path, formats=["pdf"])

    assert exporter.last_metadata
    assert exporter.last_metadata["pdf_page_size"] == "a4"
    assert exporter.last_metadata["pdf_page_count"] == 2
    pdf_bytes = created["pdf"].read_bytes()
    page_count = pdf_bytes.count(b"/Type /Page") - 1
    assert page_count == 2


def _square_panel(name: str = "square") -> dict:
    return {
        "name": name,
        "vertices": [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        "faces": [(0, 1, 2), (0, 2, 3)],
    }


def _rect_panel(name: str = "rect") -> dict:
    return {
        "name": name,
        "vertices": [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        "faces": [(0, 1, 2), (0, 2, 3)],
    }


def _square_loop_targets(perimeter: float = 4.0, length: float = 1.0) -> dict:
    return {
        "circumference": {
            "indices": [0, 1, 2, 3],
            "target": perimeter,
            "tolerance": 0.02,
            "closed": True,
        },
        "length": {
            "indices": [0, 3],
            "target": length,
            "tolerance": 0.02,
            "closed": False,
        },
    }


def test_lscm_backend_matches_loop_targets() -> None:
    backend = LSCMConformalBackend(max_iterations=5)
    panel = Panel3D.from_mapping(_square_panel("panel_a"))
    seams = {
        "panel_a": {
            "seam_allowance": 0.01,
            "loop_targets": _square_loop_targets(),
        }
    }

    flattened = backend.flatten_panels([panel], seams, scale=1.0, seam_allowance=0.0)
    assert len(flattened) == 1
    flattened_panel = flattened[0]

    loop_metrics = flattened_panel.metadata["loop_metrics"]
    circumference = loop_metrics["circumference"]
    length = loop_metrics["length"]

    assert circumference["within_tolerance"] is True
    assert length["within_tolerance"] is True
    assert circumference["uv_length"] == pytest.approx(4.0, rel=1e-3)
    assert length["uv_length"] == pytest.approx(1.0, rel=1e-3)
    assert flattened_panel.metadata["stretch_ratio"] == pytest.approx(1.0, rel=1e-3)
    assert flattened_panel.metadata["flattening"]["loop_iterations"] <= 1


def test_lscm_backend_flags_out_of_tolerance() -> None:
    backend = LSCMConformalBackend(max_iterations=0)
    panel = Panel3D.from_mapping(_square_panel("panel_warning"))
    seams = {
        "panel_warning": {
            "seam_allowance": 0.01,
            "loop_targets": _square_loop_targets(perimeter=2.0, length=0.5),
        }
    }

    flattened = backend.flatten_panels([panel], seams, scale=1.0, seam_allowance=0.0)
    flattened_panel = flattened[0]

    assert flattened_panel.metadata["requires_subdivision"] is True
    assert flattened_panel.metadata["warnings"], "Expected warnings when loop targets are missed"
    circumference = flattened_panel.metadata["loop_metrics"]["circumference"]
    assert circumference["within_tolerance"] is False


def test_pattern_exporter_collects_panel_warnings(tmp_path: Path) -> None:
    backend = LSCMConformalBackend(max_iterations=0)
    exporter = PatternExporter(backend=backend)

    mesh_payload = {"panels": [_square_panel("panel_warning")]}
    seams = {
        "panel_warning": {
            "loop_targets": _square_loop_targets(perimeter=2.0, length=0.5),
        }
    }

    created = exporter.export(mesh_payload, seams, output_dir=tmp_path, formats=["svg"])

    svg_text = created["svg"].read_text(encoding="utf-8")
    assert "panel_warning" in svg_text
    # The exporter attaches warnings into the combined metadata comment header.
    assert "panel_warnings" in svg_text


def test_pattern_exporter_annotates_panel_issues(tmp_path: Path) -> None:
    exporter = PatternExporter(budgets=NEOPRENE_DEFAULT_BUDGETS)
    mesh_payload = {"panels": [_square_panel("panel_issues")]}

    created = exporter.export(mesh_payload, None, output_dir=tmp_path, formats=["svg"])

    svg_text = created["svg"].read_text(encoding="utf-8")
    assert "panel_issues" in svg_text
    assert "issue-marker" in svg_text


def test_pattern_exporter_flags_seam_mismatch(tmp_path: Path) -> None:
    exporter = PatternExporter()
    mesh_payload = {"panels": [_square_panel("panel_a"), _rect_panel("panel_b")]}
    seams = {
        "panel_a": {"seam_partner": "panel_b", "seam_length_tolerance": 0.01},
        "panel_b": {"seam_partner": "panel_a", "seam_length_tolerance": 0.01},
    }

    exporter.export(mesh_payload, seams, output_dir=tmp_path, formats=["svg"])

    assert exporter.last_metadata
    panel_issues = exporter.last_metadata.get("panel_issues", {})
    assert any(issue["code"] == "SEAM_MISMATCH" for issue in panel_issues["panel_a"])
    assert any(issue["code"] == "SEAM_MISMATCH" for issue in panel_issues["panel_b"])


def test_pattern_exporter_propagates_seam_split_metadata(tmp_path: Path) -> None:
    exporter = PatternExporter()
    mesh_payload = {"panels": [_square_panel("panel_a")]}
    seams = {
        "panel_a": {
            "seam_avoid_ranges": [(2, 4)],
            "seam_midpoint_index": 3,
        }
    }

    exporter.export(mesh_payload, seams, output_dir=tmp_path, formats=["svg"])

    assert exporter.last_metadata
    panel_issues = exporter.last_metadata.get("panel_issues", {})
    assert panel_issues == {}
    panel = exporter.backend.flatten_panels(
        [Panel3D.from_mapping(_square_panel("panel_a"))],
        seams,
        scale=1.0,
        seam_allowance=0.01,
    )[0]
    assert panel.metadata["seam_avoid_ranges"] == [(2, 4)]
    assert panel.metadata["seam_midpoint_index"] == 3


def test_annotations_do_not_change_outline(tmp_path: Path) -> None:
    exporter = PatternExporter(budgets=NEOPRENE_DEFAULT_BUDGETS)
    mesh_payload = {"panels": [_square_panel("panel_annotations")]}

    created_off = exporter.export(
        mesh_payload,
        None,
        output_dir=tmp_path / "plain",
        formats=["svg"],
        annotate_level="off",
    )
    created_summary = exporter.export(
        mesh_payload,
        None,
        output_dir=tmp_path / "annotated",
        formats=["svg"],
        annotate_level="summary",
    )

    off_text = created_off["svg"].read_text(encoding="utf-8")
    summary_text = created_summary["svg"].read_text(encoding="utf-8")

    assert "<circle" not in off_text
    assert "issue-marker" in summary_text

    pattern = r'id="panel_annotations-outline" class="panel-outline" points="([^"]+)"'
    off_match = re.search(pattern, off_text)
    summary_match = re.search(pattern, summary_text)
    assert off_match
    assert summary_match
    assert off_match.group(1) == summary_match.group(1)


def test_simple_backend_orders_outline() -> None:
    exporter = PatternExporter()
    mesh_payload = {
        "panels": [
            {
                "name": "chaotic",
                "vertices": [
                    (0.0, 1.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, -1.0, 0.0),
                    (-1.0, 0.0, 0.0),
                    (0.7, 0.7, 0.0),
                ],
                "faces": [(0, 1, 2)],
            }
        ]
    }

    panel = Panel3D.from_mapping(mesh_payload["panels"][0])
    flattened = exporter.backend.flatten_panels([panel], None, scale=1.0, seam_allowance=0.01)

    outline = flattened[0].seam_outline
    angles = [math.atan2(y, x) for x, y in outline]
    assert angles == sorted(angles)


def test_lscm_backend_requires_igl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "igl", raising=False)

    real_import = builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "igl":
            raise ModuleNotFoundError("No module named 'igl'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    with pytest.raises(ModuleNotFoundError):
        LSCMUnwrapBackend()


def _install_igl_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("igl")

    def boundary_loop(faces: np.ndarray) -> np.ndarray:
        return np.asarray([0, 1, 2], dtype=int)

    def lscm(vertices, triangles, anchors, targets):
        coords = np.asarray(vertices, dtype=float)[:, :2]
        return 0, coords

    module.boundary_loop = boundary_loop
    module.lscm = lscm
    monkeypatch.setitem(sys.modules, "igl", module)


def test_lscm_backend_flattens_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_igl_stub(monkeypatch)
    backend = LSCMUnwrapBackend()
    panel = Panel3D(
        name="tri",
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
    )

    flattened = backend.flatten_panels([panel], seams=None, scale=1.0, seam_allowance=0.01)

    assert flattened[0].name == "tri"
    assert flattened[0].metadata.get("backend") == "lscm"
    assert len(flattened[0].seam_outline) >= 3


def test_pattern_exporter_supports_lscm_backend(
    tmp_path: Path,
    mesh_payload: dict,
    seam_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_igl_stub(monkeypatch)
    exporter = PatternExporter(backend="lscm")

    created = exporter.export(
        mesh_payload,
        seam_payload,
        output_dir=tmp_path,
        formats=["svg"],
    )

    assert "svg" in created and created["svg"].exists()


def test_build_panel_annotations_from_metadata() -> None:
    outline = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    seam_metadata = {
        "notches": [
            {"fraction": 0.25, "label": "A"},
        ],
        "folds": [
            {"start_fraction": 0.0, "end_fraction": 0.5, "kind": "mountain"},
        ],
    }
    panel_metadata = {
        "grainline": {"direction": [0.0, 1.0], "origin": [0.5, 0.5], "length": 1.0},
        "label": {"text": "Front", "position": [0.5, 0.75]},
    }

    annotations = build_panel_annotations(
        outline,
        seam_metadata=seam_metadata,
        panel_metadata=panel_metadata,
        panel_name="front",
    )

    assert annotations.grainlines and annotations.grainlines[0].origin == (0.5, 0.5)
    assert annotations.notches and annotations.notches[0].label == "A"
    assert annotations.folds and annotations.folds[0].kind == "mountain"
    assert annotations.label and annotations.label.text == "Front"
