"""In-memory angular reservoir buffer storing observation snapshots across 360 degrees."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class StreamObservation:
    """Zero-image observation snapshot containing only lightweight numerical landmarks."""

    timestamp_s: float
    frame_index: int
    landmarks_2d: np.ndarray  # Shape (J, 2) normalized [0, 1]
    landmarks_3d: np.ndarray  # Shape (J, 3) metric meters
    visibility: np.ndarray  # Shape (J,) in [0, 1]
    yaw_rad: float  # In [-pi, pi)
    yaw_uncertainty_rad: float
    silhouette_bounds: np.ndarray | None  # e.g. [min_x, min_y, max_x, max_y]
    quality_score: float
    orientation_verified: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp_s": float(self.timestamp_s),
            "frame_index": int(self.frame_index),
            "yaw_rad": float(self.yaw_rad),
            "yaw_deg": float(math.degrees(self.yaw_rad)),
            "yaw_uncertainty_rad": float(self.yaw_uncertainty_rad),
            "quality_score": float(self.quality_score),
            "orientation_verified": bool(self.orientation_verified),
            "has_silhouette": self.silhouette_bounds is not None,
            "joint_count": int(self.landmarks_2d.shape[0]),
        }


@dataclass(frozen=True)
class AngularReservoirConfig:
    """Configuration for angular reservoir binning."""

    bins: int = 8
    min_quality: float = 0.50
    max_yaw_uncertainty_rad: float = 0.25
    keep_secondary: bool = True
    min_frame_separation_secondary: int = 15


class AngularReservoir:
    """Fixed-size in-memory buffer storing best observation per angular bin (O(K) RAM)."""

    def __init__(self, config: AngularReservoirConfig | None = None) -> None:
        self.config = config or AngularReservoirConfig()
        self.num_bins = int(self.config.bins)
        self.bin_width_rad = (2.0 * math.pi) / self.num_bins

        # Primary and secondary slots per bin
        self._primary: list[StreamObservation | None] = [None] * self.num_bins
        self._secondary: list[StreamObservation | None] = [None] * self.num_bins
        self._total_offers: int = 0
        self._accepted_offers: int = 0

    def bin_index(self, yaw_rad: float) -> int:
        """Map yaw in [-pi, pi) to a discrete bin index in [0, num_bins - 1]."""
        # Center bin 0 at 0 radians (frontal)
        shifted = (yaw_rad + 0.5 * self.bin_width_rad) % (2.0 * math.pi)
        return int(math.floor(shifted / self.bin_width_rad)) % self.num_bins

    def bin_center_rad(self, bin_idx: int) -> float:
        """Return central yaw angle in radians for given bin index."""
        raw = bin_idx * self.bin_width_rad
        return float((raw + math.pi) % (2.0 * math.pi) - math.pi)

    @staticmethod
    def compute_quality(
        landmarks_3d: np.ndarray,
        visibility: Sequence[float] | np.ndarray,
        yaw_uncertainty_rad: float,
        *,
        has_silhouette: bool = False,
    ) -> float:
        """Compute multi-component quality score Q = w_v*Q_v + w_l*Q_l + w_s*Q_s - w_u*Q_u."""
        vis = np.asarray(visibility, dtype=float)
        mean_vis = float(np.mean(vis)) if vis.size > 0 else 0.0

        # Core torso joints (shoulders 11, 12; hips 23, 24)
        torso_vis = float(np.mean(vis[[11, 12, 23, 24]])) if vis.size >= 25 else mean_vis

        # Landmark stability/spread
        pts = np.asarray(landmarks_3d, dtype=float)
        span = float(np.ptp(pts[:, 1])) if pts.shape[0] >= 2 else 0.0
        landmark_score = min(1.0, max(0.0, span / 1.5))  # Expect ~1.5m torso+legs

        sil_bonus = 0.15 if has_silhouette else 0.0
        unc_penalty = min(0.35, yaw_uncertainty_rad * 0.7)

        quality = 0.40 * torso_vis + 0.35 * landmark_score + sil_bonus - unc_penalty
        return max(0.0, min(1.0, float(quality)))

    def offer(self, obs: StreamObservation) -> bool:
        """Offer an observation to the reservoir. Retained only if quality improves the slot."""
        self._total_offers += 1

        if obs.yaw_uncertainty_rad > self.config.max_yaw_uncertainty_rad:
            return False
        if obs.quality_score < self.config.min_quality:
            return False

        idx = self.bin_index(obs.yaw_rad)
        current = self._primary[idx]

        if current is None:
            self._primary[idx] = obs
            self._accepted_offers += 1
            return True

        if obs.quality_score > current.quality_score:
            if (
                self.config.keep_secondary
                and abs(obs.frame_index - current.frame_index)
                >= self.config.min_frame_separation_secondary
            ):
                self._secondary[idx] = current
            self._primary[idx] = obs
            self._accepted_offers += 1
            return True

        # Candidate didn't beat primary; check secondary
        if self.config.keep_secondary:
            sec = self._secondary[idx]
            if abs(
                obs.frame_index - current.frame_index
            ) >= self.config.min_frame_separation_secondary and (
                sec is None or obs.quality_score > sec.quality_score
            ):
                self._secondary[idx] = obs
                return True

        return False

    def coverage(self) -> float:
        """Return fraction of bins occupied by primary observations."""
        occupied = sum(1 for slot in self._primary if slot is not None)
        return float(occupied / self.num_bins)

    def covered_bin_count(self) -> int:
        """Return total number of occupied bins."""
        return sum(1 for slot in self._primary if slot is not None)

    def is_certified_coverage(self, min_bins: int = 6) -> bool:
        """Check if certified coverage is met (sufficient total bins + key views)."""
        if self.covered_bin_count() < min_bins:
            return False

        # Key orthogonal bins: front (bin 0), right profile, back, left profile
        quarter = self.num_bins // 4
        front_idx = 0
        right_idx = quarter
        back_idx = 2 * quarter
        left_idx = 3 * quarter

        # Must have front and at least one lateral profile and either back or second profile
        has_front = self._primary[front_idx] is not None
        has_right = self._primary[right_idx] is not None
        has_back = self._primary[back_idx] is not None
        has_left = self._primary[left_idx] is not None

        return has_front and (has_right or has_left) and (has_back or (has_right and has_left))

    def selected(self) -> tuple[StreamObservation, ...]:
        """Return all certified primary observations ordered by angular bin."""
        return tuple(slot for slot in self._primary if slot is not None)

    def secondary_selected(self) -> tuple[StreamObservation, ...]:
        """Return secondary observations for cross-temporal consistency checks."""
        return tuple(slot for slot in self._secondary if slot is not None)
