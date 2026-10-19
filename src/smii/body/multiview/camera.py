"""Calibrated pinhole camera model with intrinsics, extrinsics, and ray projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CameraModel:
    """Calibrated pinhole camera model."""

    camera_id: str
    K: FloatArray  # 3x3 intrinsic matrix
    R: FloatArray  # 3x3 rotation matrix (world to camera)
    t: FloatArray  # 3-vector translation (world to camera)
    distortion: FloatArray  # (k1, k2, p1, p2, k3)
    image_size: tuple[int, int]  # (width, height)

    @classmethod
    def from_intrinsics_extrinsics(
        cls,
        camera_id: str,
        focal_length: float,
        image_size: tuple[int, int],
        R: FloatArray,
        t: FloatArray,
        distortion: FloatArray | None = None,
    ) -> CameraModel:
        """Construct a camera with principal point at image center."""
        w, h = image_size
        cx = w * 0.5
        cy = h * 0.5
        K = np.array(
            [
                [focal_length, 0.0, cx],
                [0.0, focal_length, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist = (
            np.zeros(5, dtype=np.float64)
            if distortion is None
            else np.asarray(distortion, dtype=np.float64)
        )
        return cls(
            camera_id=camera_id,
            K=K,
            R=np.asarray(R, dtype=np.float64),
            t=np.asarray(t, dtype=np.float64),
            distortion=dist,
            image_size=image_size,
        )

    @classmethod
    def look_at(
        cls,
        camera_id: str,
        eye: FloatArray,
        target: FloatArray,
        up: FloatArray = np.array([0.0, 1.0, 0.0], dtype=np.float64),
        focal_length: float = 1200.0,
        image_size: tuple[int, int] = (1920, 1080),
        distortion: FloatArray | None = None,
    ) -> CameraModel:
        """Construct a camera positioned at `eye` pointing towards `target`."""
        eye_pt = np.asarray(eye, dtype=np.float64)
        tgt_pt = np.asarray(target, dtype=np.float64)
        up_vec = np.asarray(up, dtype=np.float64)

        forward = tgt_pt - eye_pt
        f_norm = float(np.linalg.norm(forward))
        if f_norm <= 1e-12:
            raise ValueError("Camera eye and target cannot be coincident.")
        z_c = forward / f_norm

        right = np.cross(z_c, up_vec)
        r_norm = float(np.linalg.norm(right))
        if r_norm <= 1e-12:
            # Degenerate up vector, fallback to X axis
            right = np.cross(z_c, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            r_norm = float(np.linalg.norm(right)) or 1.0
        x_c = right / r_norm

        # Orthogonal up vector in camera frame (y points down for computer vision image convention)
        y_c = np.cross(z_c, x_c)

        R = np.vstack([x_c, -y_c, z_c])  # Standard OpenCV / vision coordinate system
        t = -R @ eye_pt

        return cls.from_intrinsics_extrinsics(
            camera_id=camera_id,
            focal_length=focal_length,
            image_size=image_size,
            R=R,
            t=t,
            distortion=distortion,
        )

    def project(self, points_3d: FloatArray) -> FloatArray:
        """Project (N, 3) 3D world points to (N, 2) 2D pixel coordinates."""
        pts = np.asarray(points_3d, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points_3d must have shape (N, 3), got {pts.shape}")
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float64)

        # Transform to camera frame: X_c = pts @ R.T + t
        pts_cam = pts @ self.R.T + self.t
        z = np.maximum(pts_cam[:, 2], 1e-6)  # Prevent divide-by-zero behind camera

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        u = fx * (pts_cam[:, 0] / z) + cx
        v = fy * (pts_cam[:, 1] / z) + cy

        return np.column_stack([u, v])

    def unproject_rays(self, points_2d: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Compute ray origins and unit directions in world space for (N, 2) pixel coordinates."""
        pts = np.asarray(points_2d, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points_2d must have shape (N, 2), got {pts.shape}")

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        x_norm = (pts[:, 0] - cx) / fx
        y_norm = (pts[:, 1] - cy) / fy
        z_norm = np.ones_like(x_norm)

        dirs_cam = np.column_stack([x_norm, y_norm, z_norm])
        # World direction = dirs_cam @ R
        dirs_world = dirs_cam @ self.R
        norms = np.linalg.norm(dirs_world, axis=1, keepdims=True)
        dirs_world /= np.maximum(norms, 1e-12)

        # Ray origin is camera optical center in world space: eye = -R.T @ t
        eye = -self.R.T @ self.t
        origins_world = np.tile(eye, (len(pts), 1))

        return origins_world, dirs_world


__all__ = [
    "CameraModel",
    "FloatArray",
]
