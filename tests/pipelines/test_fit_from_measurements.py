from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smii.pipelines.fit_from_measurements import (
    MEASUREMENT_MODELS,
    fit_shape_coefficients,
    fit_smplx_from_measurements,
    load_schema,
    validate_measurements,
)

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "data" / "schemas" / "body_measurements.json"


def test_fit_shape_coefficients_at_mean_values_returns_zero_vector() -> None:
    measurements = {model.name: model.mean for model in MEASUREMENT_MODELS}

    coeffs, used_names, residual = fit_shape_coefficients(measurements)

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

    result = fit_smplx_from_measurements(measurements, schema_path=SCHEMA_PATH)

    expected_betas = np.array(
        [
            1.8580307,
            0.60650707,
            -0.26539358,
            -1.3971284,
            5.66967583,
            -1.26106621,
            -3.12215697,
            0.03928729,
            0.68087977,
            0.0,
        ]
    )

    assert result.measurements_used == tuple(sorted(measurements))
    assert result.betas == pytest.approx(expected_betas, rel=1e-6, abs=1e-6)
    assert result.residual == pytest.approx(0.0, abs=1e-12)
    assert result.scale == pytest.approx(177.5 / 170.0)
    assert np.allclose(result.translation, np.zeros(3))


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
