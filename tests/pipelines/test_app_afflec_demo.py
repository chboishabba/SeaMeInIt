from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

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
        translation=np.array([0.1, 0.2, 0.3], dtype=float),
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

    class _DummyTensor:
        def __init__(self, array: np.ndarray) -> None:
            self._array = np.asarray(array)

        def detach(self) -> "_DummyTensor":
            return self

        def cpu(self) -> "_DummyTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._array

    class DummyBodyModel:
        def __init__(self, **kwargs: object) -> None:
            called["body_model_kwargs"] = kwargs
            self.model = SimpleNamespace(faces=np.array([[0, 1, 2]], dtype=np.int32))
            self._vertices = _DummyTensor(
                np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32)
            )

        def set_shape(self, betas: np.ndarray) -> None:
            called["betas"] = np.asarray(betas)

        def vertices(self) -> _DummyTensor:
            called["vertices_called"] = True
            return self._vertices

    monkeypatch.setattr("smii.pipelines.extract_measurements_from_afflec_images", fake_extract)
    monkeypatch.setattr("smii.pipelines.fit_smplx_from_measurements", fake_fit)
    monkeypatch.setattr("smii.pipelines.fit_from_measurements.save_fit", fake_save)
    monkeypatch.setattr("smii.pipelines.fit_from_measurements.plot_measurement_report", fake_plot)

    dummy_avatar_module = ModuleType("avatar_model")
    dummy_avatar_module.BodyModel = DummyBodyModel
    monkeypatch.setitem(sys.modules, "avatar_model", dummy_avatar_module)

    images = [tmp_path / "front.pgm"]
    output_dir = tmp_path / "afflec"

    result_path = app.run_afflec_fixture_demo(images=images, output_dir=output_dir)

    assert result_path == output_dir / "afflec_measurement_fit.json"
    assert called["images"] == tuple(images)
    assert called["measurements"] == measurements
    assert called["save_path"] == result_path
    assert called["plot_dir"] == output_dir
    assert (output_dir / "measurement_report.png").exists()
    np.testing.assert_array_equal(called["betas"], np.zeros((1, 10), dtype=float))
    assert called["body_model_kwargs"]["num_betas"] == 10
    assert called["vertices_called"] is True

    mesh_path = output_dir / "afflec_body.npz"
    assert mesh_path.exists()
    archive = np.load(mesh_path)
    try:
        np.testing.assert_allclose(
            archive["vertices"],
            np.array(
                [[0.1, 0.2, 0.3], [1.1, 0.2, 0.3], [0.1, 1.2, 0.3]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(archive["faces"], np.array([[0, 1, 2]], dtype=np.int32))
    finally:
        archive.close()

    out = capsys.readouterr().out
    assert f"Generated measurement report plot at {output_dir / 'measurement_report.png'}" in out
    assert f"Saved fitted body mesh to {mesh_path}" in out
