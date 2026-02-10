"""Lightweight pose legality scoring via joint spheres/capsules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = ["LegalityConfig", "LegalityResult", "score_legality", "DEFAULT_RADII"]

# Default joint sphere radii (meters) keyed by SMPL-X body joint names.
DEFAULT_RADII: Mapping[str, float] = {
    "head": 0.10,
    "neck": 0.08,
    "spine1": 0.09,
    "spine2": 0.09,
    "spine3": 0.09,
    "left_shoulder": 0.11,
    "right_shoulder": 0.11,
    "left_elbow": 0.09,
    "right_elbow": 0.09,
    "left_wrist": 0.07,
    "right_wrist": 0.07,
    "left_hip": 0.12,
    "right_hip": 0.12,
    "left_knee": 0.10,
    "right_knee": 0.10,
    "left_ankle": 0.08,
    "right_ankle": 0.08,
}


@dataclass(frozen=True, slots=True)
class LegalityConfig:
    """Configuration for collision scoring."""

    radii: Mapping[str, float]
    # Names of joint pairs to ignore (parent-child neighbors).
    ignore_pairs: Sequence[tuple[str, str]] = (
        ("spine1", "spine2"),
        ("spine2", "spine3"),
        ("spine3", "neck"),
        ("neck", "head"),
        ("left_shoulder", "left_elbow"),
        ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_elbow", "right_wrist"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"),
        ("right_knee", "right_ankle"),
    )
    alpha: float = 1.0  # slack factor on summed radii

    def __init__(
        self,
        radii: Mapping[str, float] | None = None,
        ignore_pairs: Sequence[tuple[str, str]] | None = None,
        alpha: float = 1.0,
    ) -> None:
        object.__setattr__(self, "radii", dict(radii) if radii is not None else dict(DEFAULT_RADII))
        default_pairs = (
            ("spine1", "spine2"),
            ("spine2", "spine3"),
            ("spine3", "neck"),
            ("neck", "head"),
            ("left_shoulder", "left_elbow"),
            ("right_shoulder", "right_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_elbow", "right_wrist"),
            ("left_hip", "left_knee"),
            ("right_hip", "right_knee"),
            ("left_knee", "left_ankle"),
            ("right_knee", "right_ankle"),
        )
        object.__setattr__(self, "ignore_pairs", tuple(ignore_pairs) if ignore_pairs is not None else default_pairs)
        object.__setattr__(self, "alpha", float(alpha))


@dataclass(frozen=True, slots=True)
class LegalityResult:
    score: float
    violations: int
    worst_pair: tuple[str, str] | None
    worst_penetration: float


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def score_legality(joint_positions: Mapping[str, np.ndarray], config: LegalityConfig | None = None) -> LegalityResult:
    """Compute a cheap collision-based illegality score.

    joint_positions: mapping joint name -> (3,) array in meters.
    """
    cfg = config or LegalityConfig()
    score = 0.0
    violations = 0
    worst_penetration = 0.0
    worst_pair: tuple[str, str] | None = None

    names = list(joint_positions.keys())
    ignore = {_pair_key(a, b) for a, b in cfg.ignore_pairs}

    for i, a in enumerate(names):
        pa = np.asarray(joint_positions[a], dtype=float).reshape(3)
        ra = float(cfg.radii.get(a, 0.08))
        for b in names[i + 1 :]:
            key = _pair_key(a, b)
            if key in ignore:
                continue
            pb = np.asarray(joint_positions[b], dtype=float).reshape(3)
            rb = float(cfg.radii.get(b, 0.08))
            if np.linalg.norm(pa) < 1e-6 and np.linalg.norm(pb) < 1e-6:
                continue
            required = cfg.alpha * (ra + rb)
            dist = float(np.linalg.norm(pa - pb))
            penetration = max(0.0, required - dist)
            if penetration > 0:
                violations += 1
                worst_penetration = max(worst_penetration, penetration)
                worst_pair = key
                score += penetration

    return LegalityResult(score=score, violations=violations, worst_pair=worst_pair, worst_penetration=worst_penetration)
