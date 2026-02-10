"""Joint flex accumulation and projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from smii.rom.sampler_real import PoseSample

__all__ = ["FlexStats", "accumulate_flex_stats", "project_flex_to_vertices"]


@dataclass(frozen=True, slots=True)
class FlexStats:
    joint_max_abs: Mapping[str, float]
    joint_mean_abs: Mapping[str, float]


def accumulate_flex_stats(samples: Iterable[PoseSample]) -> FlexStats:
    """Accumulate per-joint absolute angle stats from schedule metadata."""
    maxima: dict[str, float] = {}
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for sample in samples:
        meta = sample.metadata or {}
        if meta.get("level") != "L0":
            continue
        joint = meta.get("joint")
        axis = meta.get("axis")
        angle = meta.get("angle_deg")
        if joint is None or axis is None or angle is None:
            continue
        key = f"{joint}:{axis}"
        angle_abs = abs(float(angle))
        maxima[key] = max(maxima.get(key, 0.0), angle_abs)
        totals[key] = totals.get(key, 0.0) + angle_abs
        counts[key] = counts.get(key, 0) + 1

    mean_abs = {k: totals[k] / max(counts.get(k, 1), 1) for k in maxima}
    return FlexStats(joint_max_abs=maxima, joint_mean_abs=mean_abs)


def project_flex_to_vertices(
    stats: FlexStats,
    *,
    joint_vertices: Mapping[str, int],
    vertex_count: int,
    mode: str = "max",
) -> np.ndarray:
    """Project joint flex stats to a per-vertex scalar field using a joint→vertex map."""

    field = np.zeros(vertex_count, dtype=float)
    values = stats.joint_max_abs if mode == "max" else stats.joint_mean_abs
    for joint_axis, val in values.items():
        joint, _axis = joint_axis.split(":")
        if joint not in joint_vertices:
            continue
        idx = joint_vertices[joint]
        if 0 <= idx < vertex_count:
            field[idx] = max(field[idx], float(val))
    return field


def nearest_joint_vertices(joint_centers: Mapping[str, np.ndarray], vertices: np.ndarray) -> dict[str, int]:
    """Map joint centers to nearest vertex indices."""
    verts = np.asarray(vertices, dtype=float)
    out: dict[str, int] = {}
    for name, center in joint_centers.items():
        center_arr = np.asarray(center, dtype=float).reshape(3)
        if verts.size == 0:
            out[name] = 0
            continue
        dists = np.linalg.norm(verts - center_arr[None, :], axis=1)
        out[name] = int(np.argmin(dists))
    return out
