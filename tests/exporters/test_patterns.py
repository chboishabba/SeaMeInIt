"""Tests for the undersuit pattern exporter."""

from __future__ import annotations

from pathlib import Path

import pytest

from exporters.patterns import Panel2D, PatternExporter


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

