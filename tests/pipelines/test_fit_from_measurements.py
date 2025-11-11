from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipelines.measurement_inference import load_default_model
from smii.pipelines.fit_from_measurements import (
    DEFAULT_NUM_BETAS,
    fit_shape_coefficients,
    fit_smplx_from_measurements,
    load_backend_config,
    load_schema,
    validate_measurements,
)

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "data" / "schemas" / "body_measurements.json"


def test_fit_shape_coefficients_at_mean_values_returns_zero_vector() -> None:
    backend_config = load_backend_config()
    measurements = {model.name: model.mean for model in backend_config.models}

    coeffs, used_names, residual = fit_shape_coefficients(
        measurements, models=backend_config.models
    )

    assert used_names == tuple(sorted(measurements))
    assert np.allclose(coeffs, np.zeros_like(coeffs))
    assert residual == pytest.approx(0.0)


def test_fit_smplx_from_measurements_produces_expected_parameters() -> None:
    measurements = {
        "height": 177.5,
        "chest_circumference": 103.0,
        "waist_circumference": 87.0,
        "hip_circumference": 98.0,
    }

    inference_model = load_default_model()
    result = fit_smplx_from_measurements(
        measurements,
        schema_path=SCHEMA_PATH,
        inference_model=inference_model,
    )

    expected_betas = np.array(
        [
            1.55868667,
            1.12056607,
            0.17512983,
            -0.92526525,
            0.43574916,
            2.43199782,
            -0.68443765,
            1.29737847,
            2.5821451,
            -10.93089473,
        ]
    )

    assert result.measurements_used == tuple(sorted(inference_model.names))
    assert result.betas == pytest.approx(expected_betas, rel=1e-6, abs=1e-6)
    assert result.residual == pytest.approx(0.21266870067297797, rel=1e-6)
    assert result.scale == pytest.approx(177.5 / 170.0)
    assert np.allclose(result.translation, np.zeros(3))
    assert result.measurement_report.coverage == pytest.approx(4 / len(inference_model.names))
    inferred = {estimate.name for estimate in result.measurement_report.inferred()}
    assert {"shoulder_width", "arm_length", "inseam_length"} <= inferred


def test_alternative_backend_matches_default_coefficients() -> None:
    measurements = {
        "height": 177.5,
        "chest_circumference": 103.0,
        "waist_circumference": 87.0,
        "hip_circumference": 98.0,
    }

    inference_model = load_default_model()
    default_result = fit_smplx_from_measurements(
        measurements,
        schema_path=SCHEMA_PATH,
        inference_model=inference_model,
    )

    alt_result = fit_smplx_from_measurements(
        measurements,
        backend="smplx_alt",
        schema_path=SCHEMA_PATH,
        inference_model=inference_model,
    )

    assert alt_result.measurements_used == default_result.measurements_used
    np.testing.assert_allclose(alt_result.betas[:DEFAULT_NUM_BETAS], default_result.betas)
    if alt_result.betas.size > DEFAULT_NUM_BETAS:
        np.testing.assert_allclose(alt_result.betas[DEFAULT_NUM_BETAS:], 0.0)
    assert alt_result.scale == pytest.approx(default_result.scale)
    assert alt_result.residual == pytest.approx(default_result.residual)


def test_validate_measurements_reports_missing_entries() -> None:
    schema = load_schema(SCHEMA_PATH)
    incomplete = {
        "height": 170.0,
        "chest_circumference": 95.0,
    }

    with pytest.raises(ValueError) as excinfo:
        validate_measurements(incomplete, schema)

    message = str(excinfo.value)
    assert "waist_circumference" in message
    assert "hip_circumference" in message


def test_fit_shape_coefficients_requires_input_measurements() -> None:
    with pytest.raises(ValueError, match="At least one measurement"):
        fit_shape_coefficients({})
