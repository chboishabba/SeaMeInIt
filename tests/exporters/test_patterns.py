"""Tests for the undersuit pattern exporter."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from exporters.lscm_backend import LSCMConformalBackend
from exporters.patterns import Panel2D, Panel3D, PatternExporter
from exporters.patterns import LSCMUnwrapBackend, Panel2D, Panel3D, PatternExporter


class DummyBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []

    def flatten_panels(self, panels, seams, *, scale, seam_allowance):  # noqa: D401 - simple stub
        self.calls.append((scale, seam_allowance))
        outline = [(0.0, 0.0), (0.5 * scale, 0.0), (0.5 * scale, 1.0 * scale)]
        return [
            Panel2D(name="test_panel", outline=outline, seam_allowance=0.012, metadata={"source": "dummy"})
        ]


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


def test_export_creates_requested_formats(tmp_path: Path, mesh_payload: dict, seam_payload: dict) -> None:
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


def test_export_writes_metadata(tmp_path: Path, mesh_payload: dict, seam_payload: dict) -> None:
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
    assert "data-seam-allowance=\"0.012\"" in svg_text

    dxf_text = created["dxf"].read_text(encoding="utf-8")
    assert "$SMII_SCALE" in dxf_text
    assert "1.1" in dxf_text
    assert "0.0120" in dxf_text

    pdf_bytes = created["pdf"].read_bytes()
    assert b"Scale: 1.1" in pdf_bytes
    assert b"allowance=0.012" in pdf_bytes


<<<<<<< HEAD
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

=======
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

    outline = flattened[0].outline
    angles = [math.atan2(y, x) for x, y in outline]
    assert angles == sorted(angles)


def test_lscm_backend_requires_igl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "igl", raising=False)
    with pytest.raises(ModuleNotFoundError):
        LSCMUnwrapBackend()


def _install_igl_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("igl")

    def boundary_loop(faces: np.ndarray) -> np.ndarray:
        return np.asarray([0, 1, 2], dtype=int)

    def lscm(vertices, triangles, anchors, targets):
        coords = np.column_stack(
            [
                np.linspace(0.0, 1.0, len(vertices), dtype=float),
                np.linspace(0.0, 0.5, len(vertices), dtype=float),
            ]
        )
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
    assert len(flattened[0].outline) >= 3


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
