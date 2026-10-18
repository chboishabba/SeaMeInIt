"""Synthetic multi-view camera rig generator with zero disk storage overhead."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .camera import CameraModel, FloatArray


@dataclass(frozen=True)
class SyntheticRigConfig:
    """Configuration for circular camera rig placement."""

    radius: float = 2.5  # Distance from subject center in metres
    focal_length: float = 1200.0
    image_size: tuple[int, int] = (1920, 1080)
    target_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_height: float = 0.0  # Camera Y height relative to target


def build_circular_rig(
    azimuth_angles_deg: Sequence[float],
    config: SyntheticRigConfig | None = None,
) -> tuple[CameraModel, ...]:
    """Place cameras in a circle around the target at specified azimuth angles."""
    cfg = config or SyntheticRigConfig()
    target = np.array(cfg.target_point, dtype=np.float64)
    cameras: list[CameraModel] = []

    for idx, angle_deg in enumerate(azimuth_angles_deg):
        rad = np.radians(angle_deg)
        # In world coordinates: Z is depth (front +Z, back -Z), X is lateral (left +X, right -X)
        # Azimuth 0 = Front (+Z), 90 = Left (+X), 180 = Back (-Z), 270 = Right (-X)
        eye_x = float(target[0] + cfg.radius * np.sin(rad))
        eye_z = float(target[2] + cfg.radius * np.cos(rad))
        eye_y = float(target[1] + cfg.camera_height)
        eye = np.array([eye_x, eye_y, eye_z], dtype=np.float64)

        cam = CameraModel.look_at(
            camera_id=f"cam_{idx:02d}_{int(angle_deg)}deg",
            eye=eye,
            target=target,
            focal_length=cfg.focal_length,
            image_size=cfg.image_size,
        )
        cameras.append(cam)

    return tuple(cameras)


def build_frontal_rig(
    num_cameras: int = 3,
    angular_spread_deg: float = 15.0,
    config: SyntheticRigConfig | None = None,
) -> tuple[CameraModel, ...]:
    """Build a narrow-baseline frontal camera cluster simulating red-carpet/press photos."""
    if num_cameras <= 1:
        angles = [0.0]
    else:
        angles = list(np.linspace(-angular_spread_deg * 0.5, angular_spread_deg * 0.5, num_cameras))
    return build_circular_rig(angles, config)


def build_four_view_orthogonal_rig(
    config: SyntheticRigConfig | None = None,
) -> tuple[CameraModel, ...]:
    """Build standard 4-view orthogonal camera rig (Front, Left Profile, Back, Right Profile)."""
    angles = (0.0, 90.0, 180.0, 270.0)
    return build_circular_rig(angles, config)


def build_eight_view_ring_rig(
    config: SyntheticRigConfig | None = None,
) -> tuple[CameraModel, ...]:
    """Build full 360-degree 8-view ring rig at 45-degree intervals."""
    angles = tuple(float(i * 45) for i in range(8))
    return build_circular_rig(angles, config)


def project_multiview(
    cameras: Sequence[CameraModel],
    points_3d: FloatArray,
    noise_sigma_px: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[str, FloatArray]:
    """Project 3D landmarks into 2D pixel coordinates across all cameras with optional noise."""
    projections: dict[str, FloatArray] = {}
    generator = rng or np.random.default_rng(0)

    for cam in cameras:
        pts_2d = cam.project(points_3d)
        if noise_sigma_px > 0.0:
            noise = generator.normal(0.0, noise_sigma_px, size=pts_2d.shape)
            pts_2d = pts_2d + noise
        projections[cam.camera_id] = pts_2d

    return projections


__all__ = [
    "SyntheticRigConfig",
    "build_circular_rig",
    "build_eight_view_ring_rig",
    "build_four_view_orthogonal_rig",
    "build_frontal_rig",
    "project_multiview",
]
