"""Tests for boundary regularization stages R1 and R2."""

from __future__ import annotations

import math
from dataclasses import replace

from suit import NEOPRENE_DEFAULT_BUDGETS, WOVEN_DEFAULT_BUDGETS
from suit.panel_boundary_regularization import (
    PanelIssue,
    regularize_boundary,
    summarize_panel_issues,
    suggest_split_issue,
    suppress_min_features,
)
from suit.panel_model import PanelBudgets


def _closed_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def _make_zigzag(
    *, step_mm: float = 5.0, amplitude_mm: float = 10.0, segments: int = 12
) -> list[tuple[float, float]]:
    step = step_mm / 1000.0
    amplitude = amplitude_mm / 1000.0
    points: list[tuple[float, float]] = []
    x = 0.0
    sign = 1.0
    for _ in range(max(3, segments)):
        points.append((x, sign * amplitude))
        x += step
        sign *= -1.0
    if points and points[0] != points[-1]:
        points.append(points[0])
    return points


def _make_soft_wave(
    *,
    step_mm: float = 5.0,
    amplitude_mm: float = 8.0,
    base_radius_mm: float = 60.0,
    waves: int = 2,
) -> list[tuple[float, float]]:
    step = step_mm / 1000.0
    amplitude = amplitude_mm / 1000.0
    base_radius = base_radius_mm / 1000.0
    circumference = 2.0 * math.pi * base_radius
    count = max(12, int(circumference / step))
    points: list[tuple[float, float]] = []
    for idx in range(count):
        theta = 2.0 * math.pi * idx / count
        radius = base_radius + amplitude * math.sin(waves * theta)
        points.append((radius * math.cos(theta), radius * math.sin(theta)))
    return points


def _make_circle(*, radius_mm: float = 50.0, step_mm: float = 5.0) -> list[tuple[float, float]]:
    radius = radius_mm / 1000.0
    circumference = 2.0 * math.pi * radius
    count = max(12, int(circumference / (step_mm / 1000.0)))
    return [
        (
            radius * math.cos(2.0 * math.pi * idx / count),
            radius * math.sin(2.0 * math.pi * idx / count),
        )
        for idx in range(count)
    ]


def test_resample_preserves_length() -> None:
    boundary = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    budgets = PanelBudgets(
        distortion_max=8.0,
        curvature_min_radius=0.1,
        turning_max_per_length=10.0,
        min_feature_size=0.03,
    )

    resampled, issues = regularize_boundary(boundary, budgets)

    assert math.isclose(_closed_length(resampled), _closed_length(boundary), rel_tol=1e-2)


def test_spike_is_flagged_or_smoothed() -> None:
    boundary = [(0.0, 0.0), (1.0, 0.0), (1.02, 0.4), (1.04, 0.0), (2.0, 0.0)]
    budgets = PanelBudgets(
        distortion_max=8.0,
        curvature_min_radius=0.2,
        turning_max_per_length=10.0,
        min_feature_size=0.01,
    )

    resampled, issues = regularize_boundary(boundary, budgets)

    assert issues or resampled != boundary


def test_woven_flags_where_neoprene_allows() -> None:
    boundary = _make_circle(radius_mm=50.0, step_mm=5.0)
    neoprene_budgets = replace(NEOPRENE_DEFAULT_BUDGETS, min_feature_size=None)
    woven_budgets = replace(WOVEN_DEFAULT_BUDGETS, min_feature_size=None)

    neoprene_outline, neoprene_issues = regularize_boundary(
        boundary,
        neoprene_budgets,
    )
    woven_outline, woven_issues = regularize_boundary(
        boundary,
        woven_budgets,
    )

    assert neoprene_outline
    assert woven_outline
    assert not neoprene_issues
    assert any(issue.code == "TURNING_BUDGET_EXCEEDED" for issue in woven_issues)


def test_turning_budget_detects_zigzag() -> None:
    boundary = _make_zigzag(step_mm=5.0, amplitude_mm=10.0)

    _, issues = regularize_boundary(boundary, NEOPRENE_DEFAULT_BUDGETS)

    assert any(issue.code == "TURNING_BUDGET_EXCEEDED" for issue in issues)
    assert sum(issue.code == "TURNING_BUDGET_EXCEEDED" for issue in issues) == 1


def test_turning_budget_material_sensitive() -> None:
    boundary = _make_soft_wave(step_mm=5.0, amplitude_mm=8.0)

    _, neoprene_issues = regularize_boundary(boundary, NEOPRENE_DEFAULT_BUDGETS)
    _, woven_issues = regularize_boundary(boundary, WOVEN_DEFAULT_BUDGETS)

    assert not any(issue.code == "TURNING_BUDGET_EXCEEDED" for issue in neoprene_issues)
    assert any(issue.code == "TURNING_BUDGET_EXCEEDED" for issue in woven_issues)


def test_min_feature_detects_tiny_teeth() -> None:
    boundary = _make_zigzag(step_mm=3.0, amplitude_mm=6.0, segments=20)

    _, issues = regularize_boundary(boundary, NEOPRENE_DEFAULT_BUDGETS)

    assert any(issue.code == "MIN_FEATURE_VIOLATION" for issue in issues)


def test_min_feature_material_sensitive() -> None:
    boundary = _make_soft_wave(
        step_mm=3.0,
        amplitude_mm=2.0,
        base_radius_mm=8.0,
        waves=4,
    )

    _, neoprene_issues = regularize_boundary(boundary, NEOPRENE_DEFAULT_BUDGETS)
    _, woven_issues = regularize_boundary(boundary, WOVEN_DEFAULT_BUDGETS)

    assert not any(issue.code == "MIN_FEATURE_VIOLATION" for issue in neoprene_issues)
    assert any(issue.code == "MIN_FEATURE_VIOLATION" for issue in woven_issues)


def test_zigzag_helper_is_closed() -> None:
    boundary = _make_zigzag(step_mm=5.0, amplitude_mm=10.0, segments=12)

    assert boundary[0] == boundary[-1]


def test_min_feature_none_budget_is_noop() -> None:
    boundary = _make_soft_wave(step_mm=5.0, amplitude_mm=8.0)
    budgets = replace(NEOPRENE_DEFAULT_BUDGETS, min_feature_size=None)

    _, issues = regularize_boundary(boundary, budgets)

    assert not any(issue.code == "MIN_FEATURE_VIOLATION" for issue in issues)


def test_suppress_min_features_removes_short_runs() -> None:
    boundary = _make_zigzag(step_mm=3.0, amplitude_mm=6.0, segments=20)
    base = boundary[:-1] if boundary[0] == boundary[-1] else boundary

    suppressed, _ = suppress_min_features(boundary, 3.0, NEOPRENE_DEFAULT_BUDGETS)

    assert len(suppressed) < len(base)


def test_severity_escalation_blocks_sewable() -> None:
    issues = [PanelIssue.from_code("CURVATURE_EXCEEDED")]

    summary = summarize_panel_issues(issues)

    assert summary.sewable is False


def test_split_suggestion_emitted_on_repeated_turning() -> None:
    issues = [
        PanelIssue.from_code("TURNING_BUDGET_EXCEEDED", index=10),
        PanelIssue.from_code("TURNING_BUDGET_EXCEEDED", index=120),
    ]

    suggestion = suggest_split_issue(
        issues,
        turning_violation_indices=[10, 120],
        min_feature_persisted=False,
    )

    assert suggestion is not None
    assert suggestion.code == "SUGGEST_SPLIT"
