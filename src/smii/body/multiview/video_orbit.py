"""Streaming 360-degree video orbit consumer for bespoke body fitting with zero disk overhead."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterator, Protocol, Sequence

import numpy as np

from .angular_reservoir import (
    AngularReservoir,
    AngularReservoirConfig,
    StreamObservation,
)
from .identifiability import analyze_identifiability
from .synthetic_rig import build_circular_rig
from .yaw_tracker import CircularYawTracker


class StopReason(StrEnum):
    CONVERGED = "converged"
    EOF = "eof"
    FRAME_BUDGET = "frame_budget"
    TIME_BUDGET = "time_budget"
    STREAM_ERROR = "stream_error"


@dataclass(frozen=True)
class StreamConfig:
    """Configuration for online video orbit streaming."""

    max_frames: int = 1800  # Default 60 seconds at 30 fps
    max_duration_s: float = 120.0
    convergence_window: int = 90  # Frames without improvement before early exit
    minimum_covered_bins: int = 6
    max_condition_number: float = 10.0
    reservoir_bins: int = 8
    min_quality: float = 0.50
    max_yaw_uncertainty_rad: float = 0.25


@dataclass(frozen=True)
class StreamResult:
    """Outcome of processing a streaming video orbit."""

    stop_reason: StopReason
    total_frames_processed: int
    duration_s: float
    observations: tuple[StreamObservation, ...]
    secondary_observations: tuple[StreamObservation, ...]
    condition_number: float | None
    is_certified: bool
    angular_coverage: float

    def to_dict(self) -> dict[str, object]:
        return {
            "stop_reason": str(self.stop_reason),
            "total_frames_processed": int(self.total_frames_processed),
            "duration_s": float(self.duration_s),
            "observation_count": len(self.observations),
            "secondary_count": len(self.secondary_observations),
            "condition_number": float(self.condition_number)
            if self.condition_number is not None
            else None,
            "is_certified": bool(self.is_certified),
            "angular_coverage": float(self.angular_coverage),
            "observations": [obs.to_dict() for obs in self.observations],
        }


class LandmarkExtractor(Protocol):
    """Protocol for extracting 2D and 3D landmarks from a decoded video frame in RAM."""

    def extract(
        self,
        frame: np.ndarray,
        timestamp_s: float,
        frame_index: int,
    ) -> StreamObservation | None: ...


class MediaPipeLandmarkExtractor:
    """Extracts MediaPipe pose landmarks and updates the circular yaw tracker online."""

    def __init__(self, yaw_tracker: CircularYawTracker | None = None) -> None:
        try:
            import mediapipe as mp

            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except ImportError as err:
            raise ImportError(
                "mediapipe is required for MediaPipeLandmarkExtractor. Install with `pip install mediapipe`."
            ) from err

        self.yaw_tracker = yaw_tracker or CircularYawTracker()

    def extract(
        self,
        frame: np.ndarray,
        timestamp_s: float,
        frame_index: int,
    ) -> StreamObservation | None:
        """Process one RGB/BGR frame, extract landmarks, and discard image pixels."""
        import cv2

        if frame.ndim != 3 or frame.shape[2] != 3:
            return None

        # Convert to RGB if needed (OpenCV default is BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb_frame)

        if not results.pose_landmarks or not results.pose_world_landmarks:
            # Propagate yaw tracker open-loop
            self.yaw_tracker.update(None, timestamp_s=timestamp_s, frame_index=frame_index)
            return None

        # Extract 2D and 3D landmarks
        lms_2d = np.array(
            [[lm.x, lm.y] for lm in results.pose_landmarks.landmark],
            dtype=np.float64,
        )
        lms_3d = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark],
            dtype=np.float64,
        )
        visibility = np.array(
            [lm.visibility for lm in results.pose_landmarks.landmark],
            dtype=np.float64,
        )

        # Estimate yaw
        meas = self.yaw_tracker.estimate_yaw_from_landmarks(lms_3d, visibility)
        yaw_state = self.yaw_tracker.update(meas, timestamp_s=timestamp_s, frame_index=frame_index)

        # Compute silhouette bounds
        sil_bounds = np.array(
            [
                np.min(lms_2d[:, 0]),
                np.min(lms_2d[:, 1]),
                np.max(lms_2d[:, 0]),
                np.max(lms_2d[:, 1]),
            ],
            dtype=np.float64,
        )

        quality = AngularReservoir.compute_quality(
            lms_3d,
            visibility,
            yaw_state.uncertainty_rad,
            has_silhouette=True,
        )

        return StreamObservation(
            timestamp_s=float(timestamp_s),
            frame_index=int(frame_index),
            landmarks_2d=lms_2d,
            landmarks_3d=lms_3d,
            visibility=visibility,
            yaw_rad=float(yaw_state.angle_rad),
            yaw_uncertainty_rad=float(yaw_state.uncertainty_rad),
            silhouette_bounds=sil_bounds,
            quality_score=float(quality),
            orientation_verified=meas.facing_front if meas is not None else False,
        )

    def close(self) -> None:
        if hasattr(self, "_pose") and self._pose is not None:
            self._pose.close()


class VideoOrbitStreamer:
    """Consumes an unknown-length frame stream online with bounded O(K) memory."""

    def __init__(
        self,
        config: StreamConfig | None = None,
        extractor: LandmarkExtractor | None = None,
    ) -> None:
        self.config = config or StreamConfig()
        self.extractor = extractor or MediaPipeLandmarkExtractor()
        self.reservoir = AngularReservoir(
            AngularReservoirConfig(
                bins=self.config.reservoir_bins,
                min_quality=self.config.min_quality,
                max_yaw_uncertainty_rad=self.config.max_yaw_uncertainty_rad,
            )
        )

    def _evaluate_condition_number(
        self,
        observations: Sequence[StreamObservation],
    ) -> float | None:
        """Construct synthetic pinhole cameras from observation yaw angles and check kappa(J_beta)."""
        if len(observations) < 4:
            return None

        azimuths = [-math.degrees(obs.yaw_rad) for obs in observations]
        cameras = build_circular_rig(azimuths)

        mean_pts = observations[0].landmarks_3d
        P = len(mean_pts)
        num_betas = 10
        rng = np.random.default_rng(12345)
        raw_basis = rng.normal(0.0, 0.05, size=(P * 3, num_betas))
        q, _ = np.linalg.qr(raw_basis)
        basis = q.reshape(P, 3, num_betas)

        spectrum = analyze_identifiability(cameras, mean_pts, basis)
        return float(spectrum.condition_number)

    def consume(
        self,
        frames: Iterator[tuple[float, np.ndarray]],
    ) -> StreamResult:
        """Consume an online iterator of (timestamp_s, frame_bgr) with early stopping."""
        frames_processed = 0
        frames_since_improvement = 0
        stop_reason = StopReason.EOF
        last_timestamp = 0.0

        for timestamp_s, frame in frames:
            frames_processed += 1
            last_timestamp = float(timestamp_s)

            # Check time or frame budget limits
            if frames_processed >= self.config.max_frames:
                stop_reason = StopReason.FRAME_BUDGET
                break
            if timestamp_s >= self.config.max_duration_s:
                stop_reason = StopReason.TIME_BUDGET
                break

            # Extract observation in RAM
            obs = self.extractor.extract(frame, timestamp_s, frames_processed)
            if obs is not None:
                accepted = self.reservoir.offer(obs)
                if accepted:
                    frames_since_improvement = 0
                else:
                    frames_since_improvement += 1
            else:
                frames_since_improvement += 1

            # Early stopping check:
            # If coverage is certified and reservoir has stabilized for convergence_window
            if (
                self.reservoir.is_certified_coverage(self.config.minimum_covered_bins)
                and frames_since_improvement >= self.config.convergence_window
            ):
                cond = self._evaluate_condition_number(self.reservoir.selected())
                if cond is not None and cond <= self.config.max_condition_number:
                    stop_reason = StopReason.CONVERGED
                    break

        selected = self.reservoir.selected()
        cond_num = self._evaluate_condition_number(selected) if selected else None
        is_certified = (
            self.reservoir.is_certified_coverage(self.config.minimum_covered_bins)
            and cond_num is not None
            and cond_num <= self.config.max_condition_number
        )

        return StreamResult(
            stop_reason=stop_reason,
            total_frames_processed=frames_processed,
            duration_s=last_timestamp,
            observations=selected,
            secondary_observations=self.reservoir.secondary_selected(),
            condition_number=cond_num,
            is_certified=is_certified,
            angular_coverage=self.reservoir.coverage(),
        )

    def stream_from_source(
        self,
        source: str | Path,
        *,
        step_frames: int = 1,
    ) -> StreamResult:
        """Stream frames from an HTTP/RTSP URL or video file using cv2.VideoCapture."""
        import cv2

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            return StreamResult(
                stop_reason=StopReason.STREAM_ERROR,
                total_frames_processed=0,
                duration_s=0.0,
                observations=(),
                secondary_observations=(),
                condition_number=None,
                is_certified=False,
                angular_coverage=0.0,
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        def frame_generator() -> Iterator[tuple[float, np.ndarray]]:
            idx = 0
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    idx += 1
                    if idx % step_frames == 0:
                        yield (float(idx / fps), frame)
            finally:
                cap.release()

        return self.consume(frame_generator())
