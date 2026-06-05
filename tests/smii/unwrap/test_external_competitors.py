from __future__ import annotations

import json

import numpy as np
import pytest

from smii.unwrap import (
    MEASURED_SPHERE_COMPETITORS,
    adversarial_sphere_fields,
    benchmark_adversarial_sphere_fields,
    benchmark_external_sphere_competitors,
)
import smii.unwrap.external_competitors as external_competitors


def _sample_sphere(xyz: np.ndarray) -> np.ndarray:
    return np.asarray([xyz[0], xyz[1], xyz[2]], dtype=float)


def test_measured_competitor_set_includes_declared_sphere_methods() -> None:
    assert {
        "bt369",
        "equal_area",
        "equirect",
        "cubed_sphere",
        "octahedral",
    }.issubset(MEASURED_SPHERE_COMPETITORS)


def test_benchmark_ranks_measured_competitors_by_score() -> None:
    benchmark = benchmark_external_sphere_competitors(
        _sample_sphere,
        width=6,
        height=4,
        competitors=MEASURED_SPHERE_COMPETITORS,
    )

    assert {receipt.name for receipt in benchmark.measured} == set(MEASURED_SPHERE_COMPETITORS)
    assert all(receipt.available for receipt in benchmark.competitors)
    scores = [receipt.metrics.aggregate_score for receipt in benchmark.competitors]
    assert scores == sorted(scores)
    assert benchmark.winner.name == benchmark.competitors[0].name


def test_optional_competitors_are_unavailable_when_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(external_competitors, "find_spec", lambda name: None)

    benchmark = benchmark_external_sphere_competitors(
        _sample_sphere,
        width=4,
        height=3,
        competitors=("healpix", "xatlas"),
    )

    assert [receipt.name for receipt in benchmark.unavailable] == ["healpix", "xatlas"]
    assert all(not receipt.available for receipt in benchmark.competitors)
    assert all(receipt.reason for receipt in benchmark.competitors)
    assert all(receipt.metrics.aggregate_score is None for receipt in benchmark.competitors)


def test_to_json_is_serializable_receipt_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(external_competitors, "find_spec", lambda name: None)

    payload = benchmark_external_sphere_competitors(
        _sample_sphere,
        width=4,
        height=3,
        competitors=("bt369", "equal_area", "healpix"),
    ).to_json()

    json.dumps(payload)
    assert payload["width"] == 4
    assert payload["height"] == 3
    assert isinstance(payload["winner"], str)
    assert isinstance(payload["competitors"], list)
    assert {receipt["name"] for receipt in payload["competitors"]} == {
        "bt369",
        "equal_area",
        "healpix",
    }
    assert all("metrics" in receipt for receipt in payload["competitors"])
    assert all("certificate" in receipt for receipt in payload["competitors"])


def test_adversarial_sphere_fields_cover_smooth_and_discontinuous_cases() -> None:
    fields = adversarial_sphere_fields()

    assert len(fields) >= 10
    assert {"constant", "linear_xyz", "longitude_seam_stripe", "localized_gaussian_bump"}.issubset(
        {field.name for field in fields}
    )
    assert any(field.discontinuous for field in fields)
    assert any(not field.discontinuous for field in fields)


def test_adversarial_sphere_benchmark_reports_per_field_winners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(external_competitors, "find_spec", lambda name: None)

    suite = benchmark_adversarial_sphere_fields(
        width=5,
        height=4,
        competitors=("bt369", "equal_area", "healpix"),
    )
    payload = suite.to_json()

    json.dumps(payload)
    assert len(suite.results) == len(adversarial_sphere_fields())
    assert sum(suite.winner_histogram.values()) == len(suite.results)
    assert "bt369" in suite.winner_histogram
    assert all(result.benchmark.winner.available for result in suite.results)
    assert all(result.field.name for result in suite.results)
    assert payload["winner_histogram"] == suite.winner_histogram


def test_healpix_is_measured_when_dependency_is_available() -> None:
    if external_competitors.find_spec("healpy") is None:
        pytest.skip("healpy is not installed")

    benchmark = benchmark_external_sphere_competitors(
        _sample_sphere,
        width=8,
        height=4,
        competitors=("healpix",),
    )

    assert benchmark.winner.name == "healpix"
    assert benchmark.winner.available is True
    assert benchmark.winner.certificate["equal_area"] is True
    assert benchmark.winner.metrics.aggregate_score is not None


def test_invalid_width_and_unknown_competitor_raise_value_error() -> None:
    with pytest.raises(ValueError, match="width and height"):
        benchmark_external_sphere_competitors(_sample_sphere, width=0, height=3)

    with pytest.raises(ValueError, match="Unknown unwrap competitor"):
        benchmark_external_sphere_competitors(
            _sample_sphere,
            width=4,
            height=3,
            competitors=("not_a_competitor",),
        )
