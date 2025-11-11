from __future__ import annotations

import numpy as np
import pytest

from pipelines.measurement_inference import (
    DEFAULT_DATASET_PATH,
    GaussianMeasurementModel,
    MeasurementReport,
    load_default_model,
    load_measurement_samples,
)


def test_load_measurement_samples_returns_expected_shape() -> None:
    names, samples = load_measurement_samples(DEFAULT_DATASET_PATH)
    assert len(names) == samples.shape[1]
    assert samples.shape[0] >= 10


def test_gaussian_model_infers_missing_measurements() -> None:
    model = load_default_model()
    observed = {
        "height": 175.0,
        "chest_circumference": 100.0,
    }
    report = model.infer(observed)

    assert isinstance(report, MeasurementReport)
    values = report.values()
    assert set(values) == set(model.names)
    assert report.coverage == pytest.approx(len(observed) / len(model.names))

    measured = {estimate.name for estimate in report.measured()}
    inferred = {estimate.name for estimate in report.inferred()}
    assert measured == set(observed)
    assert inferred == set(model.names) - set(observed)

    for estimate in report.inferred():
        assert 0.0 <= estimate.confidence <= 1.0
        assert estimate.variance >= 0.0


def test_gaussian_model_handles_empty_observations() -> None:
    names, samples = load_measurement_samples(DEFAULT_DATASET_PATH)
    model = GaussianMeasurementModel.from_samples(names, samples)

    report = model.infer({})
    assert report.coverage == 0.0
    assert all(estimate.source == "inferred" for estimate in report.estimates)
    assert np.allclose(
        np.array([estimate.value for estimate in report.estimates]),
        model.mean,
        atol=1e-6,
    )
