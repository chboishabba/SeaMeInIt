"""Quantitative shape identifiability referee via singular value spectrum analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .camera import CameraModel, FloatArray


@dataclass(frozen=True)
class IdentifiabilitySpectrum:
    """Singular value spectrum and condition metrics for a camera configuration."""

    camera_count: int
    landmark_count: int
    shape_components: int
    singular_values: tuple[float, ...]
    condition_number: float
    effective_rank: int
    min_singular_value: float
    max_singular_value: float
    depth_uncertainty: float  # 1 / min_singular_value

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_count": self.camera_count,
            "landmark_count": self.landmark_count,
            "shape_components": self.shape_components,
            "singular_values": [round(float(s), 4) for s in self.singular_values],
            "condition_number": round(float(self.condition_number), 2),
            "effective_rank": self.effective_rank,
            "min_singular_value": round(float(self.min_singular_value), 6),
            "max_singular_value": round(float(self.max_singular_value), 4),
            "depth_uncertainty": round(float(self.depth_uncertainty), 4),
        }


def compute_shape_jacobian(
    cameras: Sequence[CameraModel],
    mean_points: FloatArray,
    shape_bases: FloatArray,
) -> FloatArray:
    """Compute analytical shape Jacobian J_beta = d(2D_pixels) / d(beta).

    Parameters
    ----------
    cameras : Sequence[CameraModel]
        Calibrated camera models.
    mean_points : FloatArray
        (P, 3) canonical landmark coordinates in world space.
    shape_bases : FloatArray
        (P, 3, K) landmark displacement vectors per shape parameter.

    Returns
    -------
    FloatArray
        (2 * len(cameras) * P, K) stacked Jacobian matrix.
    """
    pts = np.asarray(mean_points, dtype=np.float64)
    bases = np.asarray(shape_bases, dtype=np.float64)
    P = len(pts)
    K = bases.shape[2]

    rows: list[np.ndarray] = []

    for cam in cameras:
        R, t = cam.R, cam.t
        fx, fy = cam.K[0, 0], cam.K[1, 1]

        # Camera frame coordinates: X_c = pts @ R.T + t
        pts_cam = pts @ R.T + t

        for p in range(P):
            x_c, y_c, z_c = pts_cam[p, 0], pts_cam[p, 1], pts_cam[p, 2]
            z_c = max(z_c, 1e-4)
            z_sq = z_c * z_c

            # d(u, v) / d(X_cam, Y_cam, Z_cam)
            d_pix_d_cam = np.array(
                [
                    [fx / z_c, 0.0, -fx * x_c / z_sq],
                    [0.0, fy / z_c, -fy * y_c / z_sq],
                ],
                dtype=np.float64,
            )

            # d(X_cam) / d(beta) = R @ bases[p, :, :] -> shape (3, K)
            d_cam_d_beta = R @ bases[p, :, :]

            # d(pix) / d(beta) = d_pix_d_cam @ d_cam_d_beta -> shape (2, K)
            j_p = d_pix_d_cam @ d_cam_d_beta
            rows.append(j_p)

    return np.vstack(rows) if rows else np.empty((0, K), dtype=np.float64)


def analyze_identifiability(
    cameras: Sequence[CameraModel],
    mean_points: FloatArray,
    shape_bases: FloatArray,
    rank_relative_threshold: float = 1e-4,
) -> IdentifiabilitySpectrum:
    """Analyze singular value spectrum, condition number, and depth uncertainty."""
    J = compute_shape_jacobian(cameras, mean_points, shape_bases)
    if J.size == 0 or J.shape[0] < J.shape[1]:
        return IdentifiabilitySpectrum(
            camera_count=len(cameras),
            landmark_count=len(mean_points),
            shape_components=shape_bases.shape[2] if shape_bases.ndim == 3 else 0,
            singular_values=(),
            condition_number=float("inf"),
            effective_rank=0,
            min_singular_value=0.0,
            max_singular_value=0.0,
            depth_uncertainty=float("inf"),
        )

    s = np.linalg.svd(J, compute_uv=False)
    max_s = float(s[0])
    min_s = float(s[-1])
    cond = max_s / max(min_s, 1e-12)
    cutoff = max_s * rank_relative_threshold
    eff_rank = int(np.sum(s > cutoff))
    uncertainty = 1.0 / max(min_s, 1e-12)

    return IdentifiabilitySpectrum(
        camera_count=len(cameras),
        landmark_count=len(mean_points),
        shape_components=int(shape_bases.shape[2]),
        singular_values=tuple(float(v) for v in s),
        condition_number=cond,
        effective_rank=eff_rank,
        min_singular_value=min_s,
        max_singular_value=max_s,
        depth_uncertainty=uncertainty,
    )


__all__ = [
    "IdentifiabilitySpectrum",
    "analyze_identifiability",
    "compute_shape_jacobian",
]
