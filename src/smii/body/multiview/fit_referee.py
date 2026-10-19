"""Empirical shape recovery referee comparing frontal-only vs. multi-view fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .camera import CameraModel, FloatArray
from .identifiability import analyze_identifiability, compute_shape_jacobian


@dataclass(frozen=True)
class RecoveryResult:
    """Shape recovery evaluation metrics for a specific camera configuration."""

    configuration_name: str
    camera_count: int
    beta_ground_truth: tuple[float, ...]
    beta_estimated: tuple[float, ...]
    beta_recovery_rmse: float
    beta_max_abs: float
    green_goblin_detected: bool  # True if beta_max_abs > 5.0 (unphysical morphology)
    condition_number: float
    depth_uncertainty: float

    def to_dict(self) -> dict[str, object]:
        return {
            "configuration_name": self.configuration_name,
            "camera_count": self.camera_count,
            "beta_ground_truth": [round(float(b), 4) for b in self.beta_ground_truth],
            "beta_estimated": [round(float(b), 4) for b in self.beta_estimated],
            "beta_recovery_rmse": round(float(self.beta_recovery_rmse), 4),
            "beta_max_abs": round(float(self.beta_max_abs), 4),
            "green_goblin_detected": self.green_goblin_detected,
            "condition_number": round(float(self.condition_number), 2),
            "depth_uncertainty": round(float(self.depth_uncertainty), 4),
        }


def run_shape_recovery_test(
    configuration_name: str,
    cameras: Sequence[CameraModel],
    mean_points: FloatArray,
    shape_bases: FloatArray,
    beta_ground_truth: FloatArray,
    noise_sigma_px: float = 2.0,
    regularization_lambda: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> RecoveryResult:
    """Evaluate shape recovery accuracy and beta stability for a camera configuration."""
    generator = rng or np.random.default_rng(42)
    pts_mean = np.asarray(mean_points, dtype=np.float64)
    bases = np.asarray(shape_bases, dtype=np.float64)
    beta_true = np.asarray(beta_ground_truth, dtype=np.float64)
    K = len(beta_true)

    # 1. Synthesize ground-truth 3D body landmarks: X_true = mean + sum(beta_k * B_k)
    disp = np.sum(bases * beta_true[np.newaxis, np.newaxis, :], axis=2)
    pts_true = pts_mean + disp

    # 2. Compute 2D projections with measurement noise
    residuals: list[np.ndarray] = []
    for cam in cameras:
        proj_mean = cam.project(pts_mean)
        proj_true = cam.project(pts_true)
        # Add pixel noise to true observation
        noise = generator.normal(0.0, noise_sigma_px, size=proj_true.shape)
        obs = proj_true + noise
        # Residual delta = obs - proj_mean
        residuals.append((obs - proj_mean).ravel())

    delta_y = np.concatenate(residuals)

    # 3. Compute analytical shape Jacobian
    J = compute_shape_jacobian(cameras, pts_mean, bases)

    # 4. Regularized least-squares solve: (J^T J + lambda I) * beta = J^T delta_y
    A = J.T @ J + regularization_lambda * np.eye(K)
    b = J.T @ delta_y
    beta_hat = np.linalg.solve(A, b)

    # 5. Compute referee metrics
    rmse = float(np.sqrt(np.mean((beta_hat - beta_true) ** 2)))
    max_abs = float(np.max(np.abs(beta_hat)))
    # Green goblin threshold: beta_max_abs > 5.0 (SMPL-X standard Gaussian prior is N(0, 1))
    goblin = max_abs > 5.0

    spectrum = analyze_identifiability(cameras, pts_mean, bases)

    return RecoveryResult(
        configuration_name=configuration_name,
        camera_count=len(cameras),
        beta_ground_truth=tuple(float(b) for b in beta_true),
        beta_estimated=tuple(float(b) for b in beta_hat),
        beta_recovery_rmse=rmse,
        beta_max_abs=max_abs,
        green_goblin_detected=goblin,
        condition_number=spectrum.condition_number,
        depth_uncertainty=spectrum.depth_uncertainty,
    )


__all__ = [
    "RecoveryResult",
    "run_shape_recovery_test",
]
