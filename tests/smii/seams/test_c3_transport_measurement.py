from __future__ import annotations

import pytest

from smii.seams.c3_transport_measurement import (
    TriadicTransportSample,
    measure_triadic_transport,
)


def _sample(
    source_phase: int,
    source_orientation: int,
    *,
    phase_offset: int = 1,
    orientation_offset: int = 2,
    weight: float = 1.0,
) -> TriadicTransportSample:
    return TriadicTransportSample(
        source_phase=source_phase,
        target_phase=(source_phase + phase_offset) % 3,
        source_orientation_bin=source_orientation,
        target_orientation_bin=(source_orientation + orientation_offset) % 6,
        weight=weight,
    )


def test_exact_offset_transport_has_zero_error() -> None:
    samples = [_sample(phase, 2 * phase) for phase in range(3)]

    measurement, witness = measure_triadic_transport(
        samples,
        threshold=0.0,
        transport_model="bt369_c3",
        address_schema="smii.bt369.triadic_address.v1",
        phase_provenance_hash="phase-sha256",
        orientation_receipt_hash="orientation-sha256",
    )

    assert measurement.fitted_phase_offset == 1
    assert measurement.fitted_orientation_offset == 2
    assert measurement.c3_equivariance_error == 0.0
    assert measurement.passes
    assert witness.c3_equivariance_error == 0.0


def test_one_inconsistent_correspondence_produces_measured_error() -> None:
    samples = [
        _sample(0, 0),
        _sample(1, 2),
        _sample(2, 4, phase_offset=2),
        _sample(0, 1),
    ]

    measurement, witness = measure_triadic_transport(
        samples,
        threshold=0.20,
        transport_model="bt369_c3",
        address_schema="smii.bt369.triadic_address.v1",
        phase_provenance_hash="phase-sha256",
        orientation_receipt_hash="orientation-sha256",
    )

    assert measurement.fitted_phase_offset == 1
    assert measurement.fitted_orientation_offset == 2
    assert measurement.phase_error == pytest.approx(0.25)
    assert measurement.orientation_error == 0.0
    assert measurement.c3_equivariance_error == pytest.approx(0.25)
    assert not measurement.passes
    assert witness.c3_equivariance_error == pytest.approx(0.25)


def test_weighted_mode_and_joint_error_use_sample_weights() -> None:
    samples = [
        _sample(0, 0, weight=4.0),
        _sample(1, 2, weight=4.0),
        _sample(2, 4, phase_offset=2, orientation_offset=1, weight=1.0),
    ]

    measurement, _ = measure_triadic_transport(
        samples,
        threshold=0.12,
        transport_model="bt369_c3",
        address_schema="smii.bt369.triadic_address.v1",
        phase_provenance_hash="phase-sha256",
        orientation_receipt_hash="orientation-sha256",
    )

    assert measurement.total_weight == 9.0
    assert measurement.joint_mismatch_weight == 1.0
    assert measurement.c3_equivariance_error == pytest.approx(1.0 / 9.0)
    assert measurement.passes


def test_invalid_phase_and_orientation_are_rejected() -> None:
    with pytest.raises(ValueError, match="source_phase"):
        TriadicTransportSample(
            source_phase=3,
            target_phase=0,
            source_orientation_bin=0,
            target_orientation_bin=0,
        )

    with pytest.raises(ValueError, match="source_orientation_bin"):
        TriadicTransportSample(
            source_phase=0,
            target_phase=0,
            source_orientation_bin=6,
            target_orientation_bin=0,
        )


def test_empty_measurement_is_not_fabricated() -> None:
    with pytest.raises(ValueError, match="at least one"):
        measure_triadic_transport(
            [],
            threshold=0.05,
            transport_model="bt369_c3",
            address_schema="smii.bt369.triadic_address.v1",
            phase_provenance_hash="phase-sha256",
            orientation_receipt_hash="orientation-sha256",
        )
