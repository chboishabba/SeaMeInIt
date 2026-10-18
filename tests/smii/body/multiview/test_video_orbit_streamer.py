"""Tests for VideoOrbitStreamer online consumption and early stopping."""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from smii.body.multiview.angular_reservoir import StreamObservation
from smii.body.multiview.video_orbit import (
    StopReason,
    StreamConfig,
    VideoOrbitStreamer,
)


class MockExtractor:
    """Mock extractor simulating a person rotating smoothly through 360 degrees."""

    def __init__(self, rotation_duration_frames: int = 120) -> None:
        self.duration = rotation_duration_frames

    def extract(
        self,
        frame: np.ndarray,
        timestamp_s: float,
        frame_index: int,
    ) -> StreamObservation:
        # Yaw progresses continuously from -pi to +pi
        yaw = ((frame_index % self.duration) / self.duration) * 2.0 * math.pi - math.pi

        # 33 landmarks with realistic height and lateral spread
        rng = np.random.default_rng(frame_index)
        lms_3d = rng.normal(0.0, 0.25, size=(33, 3))
        lms_3d[:, 1] = np.linspace(-0.8, 0.8, 33)
        lms_2d = np.zeros((33, 2), dtype=float)
        visibility = np.ones(33, dtype=float)

        return StreamObservation(
            timestamp_s=float(timestamp_s),
            frame_index=frame_index,
            landmarks_2d=lms_2d,
            landmarks_3d=lms_3d,
            visibility=visibility,
            yaw_rad=float(yaw),
            yaw_uncertainty_rad=0.08,
            silhouette_bounds=np.array([0.2, 0.1, 0.8, 0.9]),
            quality_score=0.88,
            orientation_verified=abs(yaw) < math.pi / 2.0,
        )


def _dummy_frames(count: int) -> Iterator[tuple[float, np.ndarray]]:
    for i in range(1, count + 1):
        # 10x10 dummy image
        yield (float(i * 0.033), np.zeros((10, 10, 3), dtype=np.uint8))


def test_streamer_eof():
    cfg = StreamConfig(
        max_frames=500,
        minimum_covered_bins=6,
        convergence_window=50,
        reservoir_bins=8,
    )
    streamer = VideoOrbitStreamer(cfg, extractor=MockExtractor(rotation_duration_frames=60))

    # Feed only 30 frames (half a turn)
    result = streamer.consume(_dummy_frames(30))
    assert result.stop_reason == StopReason.EOF
    assert result.total_frames_processed == 30
    assert result.angular_coverage >= 0.35


def test_streamer_converged_early_stopping():
    cfg = StreamConfig(
        max_frames=1000,
        minimum_covered_bins=6,
        convergence_window=30,  # Low convergence window to trigger early stop
        reservoir_bins=8,
    )
    # 60 frames per full turn
    streamer = VideoOrbitStreamer(cfg, extractor=MockExtractor(rotation_duration_frames=60))

    # Feed 200 frames (more than enough for full turn + 30 stable frames)
    result = streamer.consume(_dummy_frames(200))
    assert result.stop_reason == StopReason.CONVERGED
    assert result.total_frames_processed < 200  # Stopped early!
    assert result.is_certified is True
    assert result.condition_number is not None
    assert result.condition_number <= 10.0
    assert len(result.observations) >= 6


def test_streamer_frame_budget():
    cfg = StreamConfig(
        max_frames=25,
        minimum_covered_bins=6,
        convergence_window=50,
        reservoir_bins=8,
    )
    streamer = VideoOrbitStreamer(cfg, extractor=MockExtractor())
    result = streamer.consume(_dummy_frames(100))
    assert result.stop_reason == StopReason.FRAME_BUDGET
    assert result.total_frames_processed == 25
