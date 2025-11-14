from pathlib import Path
import sys
from types import ModuleType

import pytest

from smii.pipelines.fit_from_images import (
    AfflecImageMeasurementExtractor,
    MeasurementExtractionError,
    extract_measurements_from_afflec_images,
    fit_smplx_from_images,
)

if "jsonschema" not in sys.modules:
    jsonschema_stub = ModuleType("jsonschema")

    class _ValidationError(Exception):
        """Placeholder validation error for jsonschema-free testing."""

    jsonschema_stub.Draft202012Validator = object
    jsonschema_stub.ValidationError = _ValidationError
    sys.modules["jsonschema"] = jsonschema_stub

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "afflec"


def test_extract_measurements_from_single_image():
    extractor = AfflecImageMeasurementExtractor()
    path = FIXTURE_DIR / "sample_front.pgm"
    measurements = extractor.parse_measurements(path)
    assert measurements == {
        "height": 170.0,
        "chest_circumference": 96.5,
        "shoulder_width": 42.2,
    }


def test_batch_extract_merges_measurements():
    measurements = extract_measurements_from_afflec_images(
        [FIXTURE_DIR / "sample_front.pgm", FIXTURE_DIR / "sample_side.pgm"]
    )
    assert measurements == {
        "height": 170.0,
        "chest_circumference": 96.5,
        "shoulder_width": 42.2,
        "waist_circumference": 82.3,
        "hip_circumference": 98.1,
    }


def test_batch_extract_raises_on_conflicts(tmp_path: Path):
    conflicting = tmp_path / "conflict.pgm"
    conflicting.write_text(
        """P2\n# measurement:height=165.0\n2 2\n255\n0 0\n0 0\n""",
        encoding="utf-8",
    )
    with pytest.raises(MeasurementExtractionError):
        extract_measurements_from_afflec_images(
            [FIXTURE_DIR / "sample_front.pgm", conflicting]
        )


def test_parse_measurements_requires_metadata(tmp_path: Path):
    missing = tmp_path / "missing.pgm"
    missing.write_text("P2\n2 2\n255\n0 0\n0 0\n", encoding="utf-8")
    extractor = AfflecImageMeasurementExtractor()
    with pytest.raises(MeasurementExtractionError):
        extractor.parse_measurements(missing)


def test_fit_from_images_uses_embedded_metadata_when_regressor_missing(monkeypatch: pytest.MonkeyPatch):
    called: dict[str, object] = {}

    def fake_fit_from_measurements(measurements, **kwargs):
        called["measurements"] = measurements
        called["kwargs"] = kwargs
        return "fit-result"

    monkeypatch.setattr(
        "smii.pipelines.fit_from_measurements.fit_smplx_from_measurements",
        fake_fit_from_measurements,
    )
    monkeypatch.delitem(sys.modules, "pipelines.afflec_regression", raising=False)

    result = fit_smplx_from_images(
        [FIXTURE_DIR / "sample_front.pgm", FIXTURE_DIR / "sample_side.pgm"]
    )

    assert result == "fit-result"
    assert called["measurements"] == {
        "height": 170.0,
        "chest_circumference": 96.5,
        "shoulder_width": 42.2,
        "waist_circumference": 82.3,
        "hip_circumference": 98.1,
    }
    assert called["kwargs"] == {
        "backend": "smplx",
        "schema_path": None,
        "models": None,
        "num_shape_coeffs": None,
        "inference_model": None,
    }


def test_fit_from_images_prefers_regressor_when_available(monkeypatch: pytest.MonkeyPatch):
    called: dict[str, object] = {}

    def fake_regressor(image_paths):
        called["paths"] = tuple(image_paths)
        return {"height": 180.0}

    fake_module = ModuleType("pipelines.afflec_regression")
    fake_module.regress_measurements_from_images = fake_regressor
    monkeypatch.setitem(sys.modules, "pipelines.afflec_regression", fake_module)

    def fake_fit_from_measurements(measurements, **kwargs):
        called["measurements"] = measurements
        called["kwargs"] = kwargs
        return "regressed-fit"

    monkeypatch.setattr(
        "smii.pipelines.fit_from_measurements.fit_smplx_from_measurements",
        fake_fit_from_measurements,
    )

    paths = [FIXTURE_DIR / "sample_front.pgm"]
    result = fit_smplx_from_images(paths)

    assert result == "regressed-fit"
    assert called["paths"] == tuple(paths)
    assert called["measurements"] == {"height": 180.0}
    assert called["kwargs"] == {
        "backend": "smplx",
        "schema_path": None,
        "models": None,
        "num_shape_coeffs": None,
        "inference_model": None,
    }
