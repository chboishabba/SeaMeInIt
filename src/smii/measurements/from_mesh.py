"""Geometry-derived body measurements inferred directly from a mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "MeshMeasurementConfig",
    "infer_measurements_from_mesh",
]


@dataclass(frozen=True)
class MeshMeasurementConfig:
    """Configuration for mesh-derived measurement sampling."""

    axis_up: int = 1  # y-up for SMPL/SMPL-X
    chest_level: float = 0.80
    waist_level: float = 0.55
    hip_level: float = 0.45
    shoulder_level: float = 0.82
    slice_band: float = 0.02  # +/- percentage of height around the target slice


def _axis_slices(vertices: np.ndarray, cfg: MeshMeasurementConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    axis = int(cfg.axis_up)
    up = vertices[:, axis]
    height = float(np.max(up) - np.min(up))
    if height <= 0:
        raise ValueError("Mesh height must be positive to derive measurements.")
    return up, height, vertices


def _slice_points(vertices: np.ndarray, up: np.ndarray, *, center: float, band: float, cfg: MeshMeasurementConfig) -> np.ndarray:
    mask = np.abs(up - center) <= band
    if not np.any(mask):
        return vertices[:, [0, 2]]
    return vertices[mask][:, [0, 2]]


def _convex_hull_perimeter(points: np.ndarray) -> float:
    """Monotonic chain convex hull perimeter for 2D points."""

    pts = np.unique(points, axis=0)
    if len(pts) < 2:
        return 0.0
    if len(pts) == 2:
        return float(np.linalg.norm(pts[0] - pts[1]) * 2)

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1])
    rolled = np.roll(hull, -1, axis=0)
    return float(np.sum(np.linalg.norm(hull - rolled, axis=1)))


def _slice_perimeter(vertices: np.ndarray, up: np.ndarray, *, level: float, band: float, cfg: MeshMeasurementConfig) -> float:
    band = max(band, 1e-6)
    center = float(np.min(up) + level * (np.max(up) - np.min(up)))
    pts = _slice_points(vertices, up, center=center, band=band, cfg=cfg)
    return _convex_hull_perimeter(pts)


def _joint_distance(joints: Mapping[str, Sequence[float]] | None, name_a: str, name_b: str) -> float | None:
    if not joints:
        return None
    if name_a not in joints or name_b not in joints:
        return None
    a = np.asarray(joints[name_a], dtype=float)
    b = np.asarray(joints[name_b], dtype=float)
    if a.shape != (3,) or b.shape != (3,):
        return None
    return float(np.linalg.norm(a - b))


def infer_measurements_from_mesh(
    vertices: np.ndarray,
    joints: Mapping[str, Sequence[float]] | None = None,
    *,
    config: MeshMeasurementConfig | None = None,
) -> dict[str, float]:
    """Infer a small set of anthropometric measurements from a SMPL mesh.

    The measurements are coarse but deterministic and purely geometry-based.
    """

    cfg = config or MeshMeasurementConfig()
    up, height, verts = _axis_slices(np.asarray(vertices, dtype=float), cfg)
    band = float(height * cfg.slice_band)

    shoulder_width = _joint_distance(joints, "left_shoulder", "right_shoulder")
    hip_width = _joint_distance(joints, "left_hip", "right_hip")

    if shoulder_width is None:
        top_slice = _slice_points(verts, up, center=float(np.min(up) + cfg.shoulder_level * height), band=band, cfg=cfg)
        shoulder_width = float(np.max(top_slice[:, 0]) - np.min(top_slice[:, 0]))
    if hip_width is None:
        hip_slice = _slice_points(verts, up, center=float(np.min(up) + cfg.hip_level * height), band=band, cfg=cfg)
        hip_width = float(np.max(hip_slice[:, 0]) - np.min(hip_slice[:, 0]))

    def _limb_length(names: tuple[str, str]) -> float | None:
        d = _joint_distance(joints, names[0], names[1])
        return d

    arm_length = None
    upper = _limb_length(("left_shoulder", "left_elbow"))
    lower = _limb_length(("left_elbow", "left_wrist"))
    if upper is not None and lower is not None:
        arm_length = upper + lower

    leg_length = None
    thigh = _limb_length(("left_hip", "left_knee"))
    shin = _limb_length(("left_knee", "left_ankle"))
    if thigh is not None and shin is not None:
        leg_length = thigh + shin

    torso_length = None
    hip_to_shoulder_left = _joint_distance(joints, "left_hip", "left_shoulder")
    hip_to_shoulder_right = _joint_distance(joints, "right_hip", "right_shoulder")
    if hip_to_shoulder_left is not None and hip_to_shoulder_right is not None:
        torso_length = 0.5 * (hip_to_shoulder_left + hip_to_shoulder_right)

    measurements = {
        "height": height,
        "shoulder_width": float(shoulder_width),
        "hip_width": float(hip_width),
        "chest_circumference": _slice_perimeter(verts, up, level=cfg.chest_level, band=band, cfg=cfg),
        "waist_circumference": _slice_perimeter(verts, up, level=cfg.waist_level, band=band, cfg=cfg),
        "hip_circumference": _slice_perimeter(verts, up, level=cfg.hip_level, band=band, cfg=cfg),
    }

    if arm_length is not None:
        measurements["arm_length"] = float(arm_length)
    if leg_length is not None:
        measurements["leg_length"] = float(leg_length)
    if torso_length is not None:
        measurements["torso_length"] = float(torso_length)

    return measurements
