"""Fail-closed mapping of receipted dart vertex supports into UV construction marks."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np

from smii.garment.pattern_model import (
    FullSuitPattern,
    MarkKind,
    PatternChart,
    PatternFeature,
    PatternMark,
)


def _uv_for_source(chart: PatternChart, source_id: int) -> np.ndarray:
    matches = np.flatnonzero(chart.source_vertex_ids == source_id)
    if len(matches) != 1:
        raise ValueError(
            f"Source vertex {source_id} has {len(matches)} UV copies on {chart.chart_id}; "
            "explicit cut-side correspondence required"
        )
    return np.asarray(chart.uv_mm[int(matches[0])], dtype=float)


def materialize_dart(
    chart: PatternChart,
    *,
    feature_id: str,
    operator_id: str,
    apex_vertex: int,
    mouth_left_vertex: int,
    mouth_right_vertex: int,
    max_leg_mismatch_mm: float = 1.0,
    drill_backoff_mm: float = 5.0,
) -> PatternFeature:
    """Produce a sewn (uncut) dart from three unambiguous source vertices."""
    apex = _uv_for_source(chart, apex_vertex)
    left = _uv_for_source(chart, mouth_left_vertex)
    right = _uv_for_source(chart, mouth_right_vertex)
    ll = float(np.linalg.norm(left - apex))
    lr = float(np.linalg.norm(right - apex))
    if min(ll, lr) <= drill_backoff_mm:
        raise ValueError("Dart legs too short for safe drill backoff")
    if abs(ll - lr) > max_leg_mismatch_mm:
        raise ValueError(f"Dart leg length mismatch: {abs(ll - lr):.3f} mm")
    unit_left = (left - apex) / ll
    unit_right = (right - apex) / lr
    bisector = unit_left + unit_right
    norm = float(np.linalg.norm(bisector))
    if norm <= 1e-8:
        raise ValueError("Degenerate dart bisector")
    bisector /= norm
    mouth_mid = (left + right) / 2
    drill = apex + drill_backoff_mm * bisector

    def points(*coords: np.ndarray) -> tuple[tuple[float, float], ...]:
        return tuple((float(p[0]), float(p[1])) for p in coords)

    return PatternFeature(
        feature_id=feature_id,
        operator_id=operator_id,
        kind="dart",
        marks=(
            PatternMark(MarkKind.SEW, points(left, apex)),
            PatternMark(MarkKind.SEW, points(right, apex)),
            PatternMark(MarkKind.FOLD, points(mouth_mid, apex)),
            PatternMark(MarkKind.DRILL, points(drill)),
            PatternMark(MarkKind.NOTCH, points(left, right)),
        ),
    )


def attach_dart_features(
    pattern: FullSuitPattern,
    corrections: Sequence[Mapping[str, object]],
) -> FullSuitPattern:
    """Attach explicit, verified dart definitions; never infer geometry from hints.

    Each entry requires chart_id, feature_id, operator_id, apex_vertex,
    mouth_left_vertex and mouth_right_vertex. Upstream promotion and source
    hash verification must occur at the receipt ingestion boundary.
    """
    by_chart = {chart.chart_id: chart for chart in pattern.charts}
    features: dict[str, list[PatternFeature]] = {
        chart.chart_id: list(chart.features) for chart in pattern.charts
    }
    for entry in corrections:
        if str(entry.get("kind")) != "dart":
            raise ValueError("Only explicit dart geometry is supported by this bridge")
        chart_id = str(entry["chart_id"])
        chart = by_chart[chart_id]
        feature = materialize_dart(
            chart,
            feature_id=str(entry["feature_id"]),
            operator_id=str(entry["operator_id"]),
            apex_vertex=int(entry["apex_vertex"]),
            mouth_left_vertex=int(entry["mouth_left_vertex"]),
            mouth_right_vertex=int(entry["mouth_right_vertex"]),
        )
        if any(existing.feature_id == feature.feature_id for existing in features[chart_id]):
            raise ValueError(f"Duplicate feature ID: {feature.feature_id}")
        features[chart_id].append(feature)
    return replace(
        pattern,
        charts=tuple(
            replace(chart, features=tuple(features[chart.chart_id]))
            for chart in pattern.charts
        ),
    )


__all__ = ["attach_dart_features", "materialize_dart"]
