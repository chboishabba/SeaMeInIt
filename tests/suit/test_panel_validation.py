"""Tests for panel validation gate aggregation."""

from __future__ import annotations

from suit import SuitMaterial
from suit.panel_validation import gate_panel_validation


def test_gate_panel_validation_ok() -> None:
    result = gate_panel_validation(
        budget_issue_codes=[],
        regularization_issues=[],
        material=SuitMaterial.NEOPRENE,
    )

    assert result.status == "ok"
    assert result.blocking_codes == ()
    assert result.advisory_codes == ()


def test_gate_panel_validation_warning() -> None:
    result = gate_panel_validation(
        budget_issue_codes=[],
        regularization_issues=[{"code": "TURNING_BUDGET_EXCEEDED", "severity": "warning"}],
        material=SuitMaterial.NEOPRENE,
    )

    assert result.status == "warning"
    assert "TURNING_BUDGET_EXCEEDED" in result.advisory_codes
    assert result.blocking_codes == ()


def test_gate_panel_validation_error() -> None:
    result = gate_panel_validation(
        budget_issue_codes=[],
        regularization_issues=[{"code": "SEAM_MISMATCH", "severity": "error"}],
        material=SuitMaterial.NEOPRENE,
    )

    assert result.status == "error"
    assert "SEAM_MISMATCH" in result.blocking_codes
