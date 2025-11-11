"""Tests for plotting measurement reports with uncertainty bars."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

jsonschema_module = types.ModuleType("jsonschema")


class _FakeValidationError(Exception):
    pass


jsonschema_module.Draft202012Validator = object  # type: ignore[assignment]
jsonschema_module.ValidationError = _FakeValidationError  # type: ignore[assignment]
sys.modules.setdefault("jsonschema", jsonschema_module)

from smii.pipelines.fit_from_measurements import (
    DEFAULT_NUM_BETAS,
    FitResult,
    plot_measurement_report,
)
from pipelines.measurement_inference import MeasurementEstimate, MeasurementReport


class _DummyBar:
    def __init__(self, x: float, height: float):
        self._x = x
        self._height = height

    def get_x(self) -> float:  # pragma: no cover - trivial
        return float(self._x)

    def get_width(self) -> float:  # pragma: no cover - trivial
        return 0.8

    def get_height(self) -> float:  # pragma: no cover - trivial
        return float(self._height)


class _DummyAxes:
    def __init__(self) -> None:
        self.captured_yerr: np.ndarray | None = None
        self.captured_capsize: float | None = None
        self.title: str | None = None

    def bar(self, indices, values, *, yerr=None, capsize=None, **_: object):
        self.captured_yerr = np.asarray(yerr)
        self.captured_capsize = capsize
        return [_DummyBar(x, value) for x, value in zip(indices, values)]

    def set_ylabel(self, *_: object, **__: object) -> None:  # pragma: no cover - trivial
        return None

    def set_title(self, title: str) -> None:
        self.title = title

    def set_xticks(self, *_: object, **__: object) -> None:  # pragma: no cover - trivial
        return None

    def grid(self, *_: object, **__: object) -> None:  # pragma: no cover - trivial
        return None

    def legend(self, *_: object, **__: object) -> None:  # pragma: no cover - trivial
        return None

    def annotate(self, *_: object, **__: object) -> None:  # pragma: no cover - trivial
        return None


class _DummyFigure:
    def tight_layout(self) -> None:  # pragma: no cover - trivial
        return None

    def savefig(self, path: Path, **_: object) -> None:  # pragma: no cover - trivial
        path.write_bytes(b"")


class _DummyColors:
    @staticmethod
    def to_rgba(_color: str) -> tuple[float, float, float, float]:  # pragma: no cover - trivial
        return (0.1, 0.2, 0.3, 0.4)


class _DummyPatch:
    def __init__(self, *_, **__):  # pragma: no cover - trivial
        return None


def _install_matplotlib_stubs(monkeypatch) -> _DummyAxes:
    axes = _DummyAxes()

    pyplot_module = types.ModuleType("matplotlib.pyplot")

    def _subplots(*_: object, **__: object):  # pragma: no cover - trivial
        return _DummyFigure(), axes

    pyplot_module.subplots = _subplots
    pyplot_module.close = lambda *args, **kwargs: None  # pragma: no cover - trivial

    colors_module = types.ModuleType("matplotlib.colors")
    colors_module.to_rgba = _DummyColors.to_rgba

    patches_module = types.ModuleType("matplotlib.patches")
    patches_module.Patch = _DummyPatch

    matplotlib_module = types.ModuleType("matplotlib")

    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib_module)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot_module)
    monkeypatch.setitem(sys.modules, "matplotlib.colors", colors_module)
    monkeypatch.setitem(sys.modules, "matplotlib.patches", patches_module)

    return axes


def test_plot_measurement_report_includes_one_sigma_error(monkeypatch, tmp_path):
    axes = _install_matplotlib_stubs(monkeypatch)

    monkeypatch.setattr(
        "smii.pipelines.fit_from_measurements.importlib_util.find_spec",
        lambda name: object() if name == "matplotlib" else None,
    )

    estimates = (
        MeasurementEstimate("measured_positive", 10.0, "measured", 1.0, 4.0),
        MeasurementEstimate("measured_negative", 12.0, "measured", 1.0, -5.0),
        MeasurementEstimate("inferred_positive", 14.0, "inferred", 0.5, 9.0),
        MeasurementEstimate("inferred_zero", 16.0, "inferred", 0.5, 0.0),
    )

    report = MeasurementReport(estimates=estimates, coverage=0.5)
    result = FitResult(
        betas=np.zeros(DEFAULT_NUM_BETAS, dtype=float),
        scale=1.0,
        translation=np.zeros(3, dtype=float),
        residual=0.0,
        measurements_used=tuple(estimate.name for estimate in estimates),
        measurement_report=report,
    )

    output_dir = tmp_path / "plots"
    plot_measurement_report(result, output_dir)

    assert axes.captured_capsize == 4
    np.testing.assert_allclose(axes.captured_yerr, np.array([2.0, 0.0, 3.0, 0.0]))
    assert axes.title == "Measurement report (bars show ±1σ)"
