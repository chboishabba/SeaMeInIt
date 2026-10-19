"""Tests for in-memory angular reservoir buffering and quality ranking."""

from __future__ import annotations

import math
import numpy as np

from smii.body.multiview.angular_reservoir import (
    AngularReservoir,
    AngularReservoirConfig,
    StreamObservation,
)


def _make_dummy_obs(
    yaw_rad: float,
    frame_idx: int,
    quality: float = 0.85,
    uncertainty: float = 0.10,
) -> StreamObservation:
    landmarks_2d = np.zeros((33, 2), dtype=float)
    landmarks_3d = np.zeros((33, 3), dtype=float)
    landmarks_3d[:, 1] = np.linspace(0.0, 1.7, 33)  # Torso/legs span
    visibility = np.ones(33, dtype=float)

    return StreamObservation(
        timestamp_s=float(frame_idx * 0.033),
        frame_index=frame_idx,
        landmarks_2d=landmarks_2d,
        landmarks_3d=landmarks_3d,
        visibility=visibility,
        yaw_rad=float(yaw_rad),
        yaw_uncertainty_rad=float(uncertainty),
        silhouette_bounds=np.array([0.2, 0.1, 0.8, 0.9]),
        quality_score=float(quality),
        orientation_verified=abs(yaw_rad) < math.pi / 2.0,
    )


def test_bin_index_mapping():
    res = AngularReservoir(AngularReservoirConfig(bins=8))
    # 8 bins of width pi/4:
    # Bin 0: around 0 (-pi/8 to +pi/8)
    assert res.bin_index(0.0) == 0
    assert res.bin_index(0.1) == 0
    assert res.bin_index(-0.1) == 0

    # Bin 1: around pi/4
    assert res.bin_index(math.pi / 4.0) == 1

    # Bin 2: around pi/2 (profile)
    assert res.bin_index(math.pi / 2.0) == 2


def test_reservoir_offer_and_ranking():
    res = AngularReservoir(AngularReservoirConfig(bins=8, keep_secondary=True))

    obs1 = _make_dummy_obs(yaw_rad=0.0, frame_idx=10, quality=0.70)
    obs2 = _make_dummy_obs(
        yaw_rad=0.05, frame_idx=50, quality=0.90
    )  # Better quality, separated in time
    obs3 = _make_dummy_obs(yaw_rad=0.02, frame_idx=80, quality=0.60)  # Worse quality

    assert res.offer(obs1) is True
    assert res.covered_bin_count() == 1
    assert res.selected()[0].quality_score == 0.70

    # obs2 improves primary; obs1 becomes secondary
    assert res.offer(obs2) is True
    assert res.selected()[0].quality_score == 0.90
    assert len(res.secondary_selected()) == 1
    assert res.secondary_selected()[0].quality_score == 0.70

    # obs3 does not beat either primary or secondary
    assert res.offer(obs3) is False


def test_certified_coverage():
    res = AngularReservoir(AngularReservoirConfig(bins=8))
    assert res.is_certified_coverage(min_bins=4) is False

    # Offer front (0 rad), right profile (pi/2), back (pi), left profile (-pi/2)
    res.offer(_make_dummy_obs(0.0, frame_idx=10))
    res.offer(_make_dummy_obs(math.pi / 2.0, frame_idx=30))
    res.offer(_make_dummy_obs(math.pi, frame_idx=60))
    res.offer(_make_dummy_obs(-math.pi / 2.0, frame_idx=90))

    assert res.covered_bin_count() == 4
    assert res.is_certified_coverage(min_bins=4) is True
