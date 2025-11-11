from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from pipelines.measurement_inference import MeasurementEstimate, MeasurementReport

if "jsonschema" not in sys.modules:
    jsonschema_stub = ModuleType("jsonschema")

    class _ValidationError(Exception):
        """Placeholder validation error for jsonschema-free testing."""

    jsonschema_stub.Draft202012Validator = object
    jsonschema_stub.ValidationError = _ValidationError
    sys.modules["jsonschema"] = jsonschema_stub

from smii.pipelines.fit_from_measurements import FitResult


def _fake_result(measurements: dict[str, float]) -> FitResult:
    estimates = tuple(
        MeasurementEstimate(
            name=name,
            value=float(value),
            source="measured",
            confidence=1.0,
            variance=0.0,
        )
        for name, value in measurements.items()
    )
    return FitResult(
        betas=np.zeros(10, dtype=float),
        scale=1.0,
        translation=np.zeros(3, dtype=float),
        residual=0.0,
        measurements_used=tuple(sorted(measurements)),
        measurement_report=MeasurementReport(estimates=estimates, coverage=1.0),
    )


def test_afflec_demo_announces_plot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from smii import app

    measurements = {"height": 170.0, "chest_circumference": 95.5}
    called: dict[str, object] = {}

    monkeypatch.setattr(app.importlib_util, "find_spec", lambda name: object())

    def fake_extract(images):
        called["images"] = tuple(images)
        return measurements

    def fake_fit(provided):
        called["measurements"] = provided
        return _fake_result(provided)

    def fake_save(result: FitResult, output_path: Path) -> None:
        called["save_path"] = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")

    def fake_plot(result: FitResult, output_dir: Path) -> Path:
        called["plot_dir"] = output_dir
        plot_path = output_dir / "measurement_report.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_text("plot", encoding="utf-8")
        return plot_path

    monkeypatch.setattr("smii.pipelines.extract_measurements_from_afflec_images", fake_extract)
    monkeypatch.setattr("smii.pipelines.fit_smplx_from_measurements", fake_fit)
    monkeypatch.setattr("smii.pipelines.fit_from_measurements.save_fit", fake_save)
    monkeypatch.setattr("smii.pipelines.fit_from_measurements.plot_measurement_report", fake_plot)

    images = [tmp_path / "front.pgm"]
    output_dir = tmp_path / "afflec"

    result_path = app.run_afflec_fixture_demo(images=images, output_dir=output_dir)

    assert result_path == output_dir / "afflec_measurement_fit.json"
    assert called["images"] == tuple(images)
    assert called["measurements"] == measurements
    assert called["save_path"] == result_path
    assert called["plot_dir"] == output_dir
    assert (output_dir / "measurement_report.png").exists()

    out = capsys.readouterr().out
    assert f"Generated measurement report plot at {output_dir / 'measurement_report.png'}" in out
