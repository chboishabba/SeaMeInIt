"""Tests for R6-lite spline fitting behavior."""

from __future__ import annotations

import math
from dataclasses import replace

from suit import NEOPRENE_DEFAULT_BUDGETS
from suit.panel_boundary_regularization import regularize_boundary


def _close(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if points and points[0] != points[-1]:
        points.append(points[0])
    return points


def _make_circle(*, radius_mm: float, step_mm: float) -> list[tuple[float, float]]:
    radius = radius_mm / 1000.0
    circumference = 2.0 * math.pi * radius
    count = max(12, int(circumference / (step_mm / 1000.0)))
    points = [
        (
            radius * math.cos(2.0 * math.pi * idx / count),
            radius * math.sin(2.0 * math.pi * idx / count),
        )
        for idx in range(count)
    ]
    return _close(points)


def _make_wave_circle(
    *,
    base_radius_mm: float,
    amplitude_mm: float,
    waves: int,
    step_mm: float,
) -> list[tuple[float, float]]:
    base_radius = base_radius_mm / 1000.0
    amplitude = amplitude_mm / 1000.0
    circumference = 2.0 * math.pi * base_radius
    count = max(12, int(circumference / (step_mm / 1000.0)))
    points: list[tuple[float, float]] = []
    for idx in range(count):
        theta = 2.0 * math.pi * idx / count
        radius = base_radius + amplitude * math.sin(waves * theta)
        points.append((radius * math.cos(theta), radius * math.sin(theta)))
    return _close(points)


def _codes(issues: list) -> set[str]:
    return {issue.code for issue in issues}


def test_spline_fit_applies_on_clean_outline() -> None:
    boundary = _make_circle(radius_mm=100.0, step_mm=5.0)
    budgets = replace(NEOPRENE_DEFAULT_BUDGETS, min_feature_size=None)

    _, issues = regularize_boundary(boundary, budgets)

    assert "SPLINE_FIT_APPLIED" in _codes(issues)


def test_spline_fit_skipped_when_split_suggested() -> None:
    boundary = _make_wave_circle(
        base_radius_mm=40.0,
        amplitude_mm=8.0,
        waves=12,
        step_mm=3.0,
    )

    _, issues = regularize_boundary(boundary, NEOPRENE_DEFAULT_BUDGETS)
    codes = _codes(issues)

    assert "SUGGEST_SPLIT" in codes
    assert "SPLINE_FIT_APPLIED" not in codes
