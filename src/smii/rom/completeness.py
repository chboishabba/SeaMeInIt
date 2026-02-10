"""ROM envelope and completeness utilities."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

__all__ = ["Envelope", "build_envelope", "compare_envelopes", "spearman_rank"]


Envelope = Mapping[str, Any]


class HasPoseMeta(Protocol):
    pose_id: str
    weight: float
    metadata: Mapping[str, Any] | None


def build_envelope(samples: Iterable[HasPoseMeta]) -> Envelope:
    """Aggregate envelope statistics from schedule-annotated samples."""
    l0: dict[str, dict[str, float]] = {}
    l1: dict[str, dict[str, float]] = {}
    counts = {"L0": 0, "L1": 0}
    for sample in samples:
        meta = sample.metadata or {}
        level = meta.get("level")
        if level == "L0":
            key = (meta.get("joint"), meta.get("axis"), meta.get("side"))
            if None in key:
                continue
            angle = float(meta.get("angle_deg", 0.0))
            bucket = l0.setdefault(str(key), {"min_deg": angle, "max_deg": angle, "count": 0})
            bucket["min_deg"] = min(bucket["min_deg"], angle)
            bucket["max_deg"] = max(bucket["max_deg"], angle)
            bucket["count"] += 1
            counts["L0"] += 1
        elif level == "L1":
            pair = meta.get("pair")
            if pair is None:
                continue
            bucket = l1.setdefault(str(pair), {"samples": 0})
            bucket["samples"] += 1
            counts["L1"] += 1
    return {"L0": l0, "L1": l1, "counts": counts}


def compare_envelopes(prev: Envelope, curr: Envelope, *, angle_tol: float = 0.5) -> dict[str, Any]:
    """Compute envelope deltas; angle_tol in degrees."""
    deltas: dict[str, Any] = {"L0": [], "L1": []}
    for key, bucket in curr.get("L0", {}).items():
        prev_bucket = prev.get("L0", {}).get(key)
        if not prev_bucket:
            deltas["L0"].append({"key": key, "delta": "new"})
            continue
        d_min = bucket["min_deg"] - prev_bucket["min_deg"]
        d_max = bucket["max_deg"] - prev_bucket["max_deg"]
        changed = abs(d_min) > angle_tol or abs(d_max) > angle_tol
        deltas["L0"].append(
            {
                "key": key,
                "delta_min_deg": d_min,
                "delta_max_deg": d_max,
                "changed": changed,
            }
        )
    for key, bucket in curr.get("L1", {}).items():
        prev_bucket = prev.get("L1", {}).get(key)
        if not prev_bucket:
            deltas["L1"].append({"key": key, "delta": "new"})
            continue
        delta_samples = bucket["samples"] - prev_bucket.get("samples", 0)
        deltas["L1"].append({"key": key, "delta_samples": delta_samples})
    return deltas


def spearman_rank(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman rank correlation without scipy; returns None if undefined."""
    if len(a) != len(b) or len(a) < 2:
        return None
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return None
    a_ranks = np.argsort(np.argsort(a))
    b_ranks = np.argsort(np.argsort(b))
    a_mean = np.mean(a_ranks)
    b_mean = np.mean(b_ranks)
    num = np.sum((a_ranks - a_mean) * (b_ranks - b_mean))
    den = np.sqrt(np.sum((a_ranks - a_mean) ** 2) * np.sum((b_ranks - b_mean) ** 2))
    if den == 0:
        return None
    return float(num / den)
