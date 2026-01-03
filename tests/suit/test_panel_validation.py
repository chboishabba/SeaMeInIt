"""Tests for panel validation helpers."""

from __future__ import annotations

from suit import (
    Panel,
    PanelBudgets,
    SurfacePatch,
    combine_results,
    validate_panel_budgets,
    validate_panel_curvature,
)


def test_validate_panel_budgets_requires_budgets() -> None:
    result = validate_panel_budgets(Panel(panel_id="front"))

    assert not result.ok
    assert result.issues
    assert result.issues[0].code == "missing_budgets"


def test_validate_panel_budgets_accepts_positive_values() -> None:
    panel = Panel(
        panel_id="front",
        budgets=PanelBudgets(
            distortion_max=0.1,
            curvature_min_radius=0.01,
            turning_max_per_length=0.5,
            min_feature_size=0.02,
        ),
    )

    result = validate_panel_budgets(panel)

    assert result.ok
    assert result.issues == ()


def test_validate_panel_budgets_flags_invalid_values() -> None:
    panel = Panel(
        panel_id="front",
        budgets=PanelBudgets(
            distortion_max=-1.0,
            curvature_min_radius=0.0,
            turning_max_per_length=-0.1,
            min_feature_size=0.0,
        ),
    )

    result = validate_panel_budgets(panel)

    assert not result.ok
    assert {issue.code for issue in result.issues} == {
        "distortion_max_missing",
        "curvature_min_radius_missing",
        "turning_max_per_length_missing",
        "min_feature_size_missing",
    }


def test_validate_panel_curvature_checks_budget() -> None:
    panel = Panel(
        panel_id="front",
        surface_patch=SurfacePatch(metadata={"panel_curvature": 200.0}),
        budgets=PanelBudgets(
            distortion_max=0.1,
            curvature_min_radius=0.01,
            turning_max_per_length=0.5,
            min_feature_size=0.02,
        ),
    )

    result = validate_panel_curvature(panel)

    assert not result.ok
    assert result.issues[0].code == "curvature_budget_exceeded"


def test_combine_results_merges_issues() -> None:
    panel = Panel(panel_id="front")

    combined = combine_results(
        validate_panel_budgets(panel),
        validate_panel_curvature(panel),
    )

    assert not combined.ok
    assert combined.issues
