"""Circular alpha-beta yaw filter fusing body orientation cues with wrapped innovation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def wrap_angle_pi(theta: float) -> float:
    """Wrap angle in radians to [-pi, pi)."""
    return float((theta + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass(frozen=True)
class YawMeasurement:
    """Estimated yaw angle and associated uncertainty from a single frame."""

    angle_rad: float
    uncertainty_rad: float
    shoulder_yaw_rad: float
    hip_yaw_rad: float | None
    facing_front: bool
    confidence: float


@dataclass(frozen=True)
class YawState:
    """Current state of the circular yaw tracker."""

    angle_rad: float
    velocity_rad_per_s: float
    uncertainty_rad: float
    timestamp_s: float
    frame_index: int
    is_initialized: bool


class CircularYawTracker:
    """Circular alpha-beta filter for tracking subject yaw angle during 360 rotation."""

    def __init__(
        self,
        *,
        alpha: float = 0.45,
        gamma: float = 0.15,
        max_velocity_rad_s: float = 2.0 * math.pi,  # Max 1 rev/sec
        base_uncertainty_rad: float = 0.15,
    ) -> None:
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.max_velocity = float(max_velocity_rad_s)
        self.base_uncertainty = float(base_uncertainty_rad)

        self._theta: float = 0.0
        self._omega: float = 0.0
        self._uncertainty: float = math.pi
        self._last_timestamp: float | None = None
        self._last_frame_index: int = -1
        self._initialized: bool = False

    @property
    def state(self) -> YawState:
        return YawState(
            angle_rad=self._theta,
            velocity_rad_per_s=self._omega,
            uncertainty_rad=self._uncertainty,
            timestamp_s=self._last_timestamp or 0.0,
            frame_index=self._last_frame_index,
            is_initialized=self._initialized,
        )

    def reset(self) -> None:
        self._theta = 0.0
        self._omega = 0.0
        self._uncertainty = math.pi
        self._last_timestamp = None
        self._last_frame_index = -1
        self._initialized = False

    @staticmethod
    def estimate_yaw_from_landmarks(
        landmarks_3d: np.ndarray,
        visibility: Sequence[float] | np.ndarray,
    ) -> YawMeasurement | None:
        """Estimate yaw angle and uncertainty from 33 MediaPipe pose landmarks in 3D."""
        landmarks_3d = np.asarray(landmarks_3d, dtype=float)
        visibility = np.asarray(visibility, dtype=float)

        if landmarks_3d.shape[0] < 25 or visibility.shape[0] < 25:
            return None

        # MediaPipe Landmark Indices:
        # 0: nose, 2: left_eye, 5: right_eye, 7: left_ear, 8: right_ear
        # 11: left_shoulder, 12: right_shoulder
        # 23: left_hip, 24: right_hip
        vis_ls = float(visibility[11])
        vis_rs = float(visibility[12])
        vis_lh = float(visibility[23])
        vis_rh = float(visibility[24])

        if vis_ls < 0.3 or vis_rs < 0.3:
            return None

        # Camera coords: X right, Y down, Z forward
        p_ls = landmarks_3d[11]
        p_rs = landmarks_3d[12]
        delta_s = p_ls - p_rs
        # Angle of vector from right to left shoulder in XZ plane
        raw_shoulder_angle = math.atan2(delta_s[2], delta_s[0])

        # Check face cues for front vs back disambiguation
        vis_nose = float(visibility[0]) if visibility.size > 0 else 0.0
        has_face = vis_nose > 0.4

        # Facing front corresponds to shoulder angle near 0 (left shoulder at positive X)
        facing_front = has_face and abs(wrap_angle_pi(raw_shoulder_angle)) < math.pi / 2.0

        angle_s = wrap_angle_pi(raw_shoulder_angle)

        # Check hips if visible
        angle_h: float | None = None
        if vis_lh >= 0.3 and vis_rh >= 0.3:
            delta_h = landmarks_3d[23] - landmarks_3d[24]
            angle_h = wrap_angle_pi(math.atan2(delta_h[2], delta_h[0]))

        # Fuse shoulder and hip
        if angle_h is not None:
            diff = abs(wrap_angle_pi(angle_s - angle_h))
            # Average angles circularly
            sin_avg = 0.6 * math.sin(angle_s) + 0.4 * math.sin(angle_h)
            cos_avg = 0.6 * math.cos(angle_s) + 0.4 * math.cos(angle_h)
            fused_angle = wrap_angle_pi(math.atan2(sin_avg, cos_avg))
            uncertainty = max(0.08, diff * 0.5)
            confidence = max(0.1, min(1.0, 1.0 - diff / math.pi))
        else:
            fused_angle = angle_s
            uncertainty = 0.22
            confidence = 0.65

        # If back-facing, add slight uncertainty
        if not facing_front:
            uncertainty += 0.05
            confidence *= 0.85

        return YawMeasurement(
            angle_rad=fused_angle,
            uncertainty_rad=float(uncertainty),
            shoulder_yaw_rad=float(angle_s),
            hip_yaw_rad=float(angle_h) if angle_h is not None else None,
            facing_front=facing_front,
            confidence=float(confidence),
        )

    def update(
        self,
        measurement: YawMeasurement | None,
        *,
        timestamp_s: float,
        frame_index: int,
    ) -> YawState:
        """Update the tracker with a new frame measurement or propagate open-loop."""
        dt = (
            max(1e-4, timestamp_s - self._last_timestamp)
            if self._last_timestamp is not None
            else 0.033
        )
        self._last_timestamp = float(timestamp_s)
        self._last_frame_index = int(frame_index)

        if not self._initialized:
            if measurement is not None:
                self._theta = measurement.angle_rad
                self._omega = 0.0
                self._uncertainty = measurement.uncertainty_rad
                self._initialized = True
            return self.state

        # Prediction step
        pred_theta = wrap_angle_pi(self._theta + self._omega * dt)

        if measurement is None:
            # Open-loop propagation: uncertainty grows
            self._theta = pred_theta
            self._uncertainty = min(math.pi, self._uncertainty + 0.5 * dt)
            return self.state

        # Wrapped innovation: e_t = wrap(z_t - theta_pred)
        innovation = wrap_angle_pi(measurement.angle_rad - pred_theta)

        # Scale gain inversely with measurement uncertainty
        gain_scale = max(
            0.1, min(1.0, self.base_uncertainty / max(1e-4, measurement.uncertainty_rad))
        )
        effective_alpha = self.alpha * gain_scale
        effective_gamma = self.gamma * gain_scale

        # State correction
        self._theta = wrap_angle_pi(pred_theta + effective_alpha * innovation)
        raw_omega = self._omega + (effective_gamma / dt) * innovation
        self._omega = max(-self.max_velocity, min(self.max_velocity, raw_omega))

        # Update uncertainty
        self._uncertainty = (
            1.0 - effective_alpha
        ) * self._uncertainty + effective_alpha * measurement.uncertainty_rad
        return self.state
