"""Tests for the hard layer exporter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from exporters.hard_layer_export import (
    HardLayerCADBackend,
    HardLayerExportError,
    HardLayerExporter,
    ShellPanel,
)


def _make_backend(tmp_path: Path) -> tuple[mock.Mock, dict[str, str]]:
    model: dict[str, str] = {"id": "model"}
    backend = mock.create_autospec(HardLayerCADBackend, instance=True)
    backend.load_panel.return_value = model

    def export_side_effect(model_obj: dict[str, str], destination: Path, fmt: str) -> Path:
        destination.write_text(f"{model_obj['id']}::{fmt}", encoding="utf-8")
        return destination

    backend.export.side_effect = export_side_effect
    return backend, model


def test_exporter_creates_files_and_metadata(tmp_path: Path) -> None:
    backend, model = _make_backend(tmp_path)
    exporter = HardLayerExporter(backend=backend)
    panel = ShellPanel(
        name="Chest Panel",
        label="C-01",
        size_marker="M",
        fit_test_geometry={"type": "peg"},
    )

    result = exporter.export([panel], tmp_path, metadata={"project": "Atlas"})

    stl_path = tmp_path / "chest-panel.stl"
    step_path = tmp_path / "chest-panel.step"
    assert stl_path.exists()
    assert step_path.exists()

    metadata_path = tmp_path / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert payload["project"] == "Atlas"
    assert payload["panels"][0]["label"] == "C-01"
    assert payload["panels"][0]["size_marker"] == "M"
    assert payload["panels"][0]["fit_test_geometry"] == {"type": "peg"}
    assert result["panels"][0]["stl"] == "chest-panel.stl"
    assert result["panels"][0]["step"] == "chest-panel.step"

    backend.load_panel.assert_called_once_with(panel)
    backend.embed_label.assert_called_once_with(model, "C-01")
    backend.embed_sizing_marker.assert_called_once_with(model, "M")
    backend.embed_fit_test_geometry.assert_called_once_with(model, {"type": "peg"})


def test_metadata_includes_mesh_reference(tmp_path: Path) -> None:
    backend, _ = _make_backend(tmp_path)
    exporter = HardLayerExporter(backend=backend)
    mesh_path = tmp_path / "torso.obj"
    mesh_path.write_text("mesh", encoding="utf-8")

    panel = ShellPanel(name="Torso", label="T-01", mesh=mesh_path)
    exporter.export([panel], tmp_path)

    payload = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert payload["panels"][0]["mesh"] == str(mesh_path)


def test_backend_failure_raises_custom_error(tmp_path: Path) -> None:
    backend = mock.create_autospec(HardLayerCADBackend, instance=True)
    backend.load_panel.return_value = {}

    def failing_export(model: dict[str, str], destination: Path, fmt: str) -> Path:
        raise RuntimeError("boom")

    backend.export.side_effect = failing_export

    exporter = HardLayerExporter(backend=backend)
    panel = ShellPanel(name="Collar", label="COL-1")

    with pytest.raises(HardLayerExportError):
        exporter.export([panel], tmp_path)

    assert not (tmp_path / "collar.stl").exists()
    assert not (tmp_path / "collar.step").exists()
