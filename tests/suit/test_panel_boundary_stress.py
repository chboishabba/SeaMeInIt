"""Stress tests for boundary regularization edge cases."""

from __future__ import annotations

import math
from dataclasses import replace

from suit import NEOPRENE_DEFAULT_BUDGETS
from suit.panel_boundary_regularization import regularize_boundary, summarize_panel_issues


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


def _make_spike() -> list[tuple[float, float]]:
    base = [
        (0.0, 0.0),
        (0.05, 0.0),
        (0.051, 0.02),
        (0.052, 0.0),
        (0.1, 0.0),
        (0.1, 0.1),
        (0.0, 0.1),
    ]
    return _close(base)


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


def _has_code(issues: list) -> set[str]:
    return {issue.code for issue in issues}


def test_almost_circle_is_clean() -> None:
    boundary = _make_circle(radius_mm=100.0, step_mm=5.0)
    budgets = replace(NEOPRENE_DEFAULT_BUDGETS, min_feature_size=None)

    _, issues = regularize_boundary(boundary, budgets)

    assert issues == []


def test_spike_triggers_curvature_error() -> None:
    boundary = _make_spike()

    _, issues = regularize_boundary(boundary, NEOPRENE_DEFAULT_BUDGETS)

    assert "CURVATURE_EXCEEDED" in _has_code(issues)
    summary = summarize_panel_issues(issues)
    assert summary.sewable is False


def test_dense_oscillation_suggests_split() -> None:
    boundary = _make_wave_circle(
        base_radius_mm=40.0,
        amplitude_mm=8.0,
        waves=12,
        step_mm=3.0,
    )

    _, issues = regularize_boundary(boundary, NEOPRENE_DEFAULT_BUDGETS)

    assert "SUGGEST_SPLIT" in _has_code(issues)


def test_tiny_teeth_are_suppressed() -> None:
    boundary = _make_wave_circle(
        base_radius_mm=30.0,
        amplitude_mm=3.0,
        waves=20,
        step_mm=2.0,
    )

    _, issues = regularize_boundary(boundary, NEOPRENE_DEFAULT_BUDGETS)
    codes = _has_code(issues)

    assert "MIN_FEATURE_VIOLATION" in codes
    assert "FEATURE_SUPPRESSED" in codes
