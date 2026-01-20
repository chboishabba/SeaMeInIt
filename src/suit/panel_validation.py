"""Validation helpers for panel sewability budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .panel_boundary_regularization import severity_for_code
from .panel_defaults import SuitMaterial
from .panel_model import Panel


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """Single validation issue with a stable code."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Aggregate validation result for a panel."""

    ok: bool
    issues: tuple[ValidationIssue, ...] = ()


def _append_issue(issues: list[ValidationIssue], code: str, message: str) -> None:
    issues.append(ValidationIssue(code=code, message=message))


def _validate_positive(
    issues: list[ValidationIssue],
    name: str,
    value: float | None,
    *,
    code: str,
) -> None:
    if value is None:
        _append_issue(issues, code, f"{name} is required.")
        return
    if value <= 0:
        _append_issue(issues, code, f"{name} must be positive.")


def validate_panel_budgets(
    panel: Panel,
    *,
    require_budgets: bool = True,
) -> ValidationResult:
    """Validate numeric sewability budgets for a panel."""

    budgets = panel.budgets
    if budgets is None:
        if require_budgets:
            return ValidationResult(
                ok=False,
                issues=(
                    ValidationIssue(
                        code="missing_budgets",
                        message="Panel budgets are required for sewability checks.",
                    ),
                ),
            )
        return ValidationResult(ok=True)

    issues: list[ValidationIssue] = []
    _validate_positive(
        issues,
        "distortion_max",
        budgets.distortion_max,
        code="distortion_max_missing",
    )
    _validate_positive(
        issues,
        "curvature_min_radius",
        budgets.curvature_min_radius,
        code="curvature_min_radius_missing",
    )
    _validate_positive(
        issues,
        "turning_max_per_length",
        budgets.turning_max_per_length,
        code="turning_max_per_length_missing",
    )
    _validate_positive(
        issues,
        "min_feature_size",
        budgets.min_feature_size,
        code="min_feature_size_missing",
    )

    return ValidationResult(ok=not issues, issues=tuple(issues))

def validate_panel_curvature(
    panel: Panel,
    *,
    metadata_key: str = "panel_curvature",
) -> ValidationResult:
    """Validate curvature metadata against the minimum radius budget."""

    budgets = panel.budgets
    if budgets is None or budgets.curvature_min_radius is None:
        return ValidationResult(ok=True)

    surface = panel.surface_patch
    if surface is None:
        return ValidationResult(
            ok=False,
            issues=(
                ValidationIssue(
                    code="missing_surface_patch",
                    message="Panel surface_patch is required for curvature validation.",
                ),
            ),
        )

    curvature = surface.metadata.get(metadata_key)
    if curvature is None:
        return ValidationResult(
            ok=False,
            issues=(
                ValidationIssue(
                    code="missing_curvature_metric",
                    message=f"Panel metadata missing '{metadata_key}'.",
                ),
            ),
        )

    curvature_value = float(curvature)
    if curvature_value <= 0.0:
        return ValidationResult(
            ok=False,
            issues=(
                ValidationIssue(
                    code="invalid_curvature_metric",
                    message=f"Panel metadata '{metadata_key}' must be positive.",
                ),
            ),
        )

    max_curvature = 1.0 / float(budgets.curvature_min_radius)
    if curvature_value > max_curvature:
        return ValidationResult(
            ok=False,
            issues=(
                ValidationIssue(
                    code="curvature_budget_exceeded",
                    message="Panel curvature exceeds minimum radius budget.",
                ),
            ),
        )

    return ValidationResult(ok=True)


def combine_results(*results: ValidationResult) -> ValidationResult:
    """Combine multiple validation results into one."""

    issues: list[ValidationIssue] = []
    for result in results:
        if not result.ok:
            issues.extend(result.issues)
    return ValidationResult(ok=not issues, issues=tuple(issues))


@dataclass(frozen=True, slots=True)
class PanelGateResult:
    """Final panel acceptance decision based on validation issues."""

    status: str
    blocking_codes: tuple[str, ...] = ()
    advisory_codes: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        return {
            "status": self.status,
            "blocking_codes": list(self.blocking_codes),
            "advisory_codes": list(self.advisory_codes),
        }


def gate_panel_validation(
    *,
    budget_issue_codes: Sequence[str] | None,
    regularization_issues: Sequence[Mapping[str, Any]] | None,
    material: SuitMaterial | str | None = None,
) -> PanelGateResult:
    """Aggregate budget and regularization issues into an accept/warn/reject gate."""

    blocking: set[str] = set()
    advisory: set[str] = set()

    for code in budget_issue_codes or []:
        blocking.add(str(code))

    for issue in regularization_issues or []:
        code = str(issue.get("code", ""))
        if not code:
            continue
        severity = str(issue.get("severity") or severity_for_code(code))
        if severity == "error":
            blocking.add(code)
        elif severity in {"warning", "info"}:
            advisory.add(code)

    material_value = _normalize_material(material)
    if material_value == SuitMaterial.WOVEN:
        for code in _WOVEN_WARNING_ESCALATIONS:
            if code in advisory:
                advisory.remove(code)
                blocking.add(code)

    if blocking:
        status = "error"
    elif advisory:
        status = "warning"
    else:
        status = "ok"

    return PanelGateResult(
        status=status,
        blocking_codes=tuple(sorted(blocking)),
        advisory_codes=tuple(sorted(advisory)),
    )


def _normalize_material(material: SuitMaterial | str | None) -> SuitMaterial | None:
    if material is None:
        return None
    if isinstance(material, SuitMaterial):
        return material
    try:
        return SuitMaterial(str(material))
    except ValueError:
        return None


_WOVEN_WARNING_ESCALATIONS: set[str] = {
    "TURNING_BUDGET_EXCEEDED",
    "MIN_FEATURE_VIOLATION",
}


__all__ = [
    "PanelGateResult",
    "ValidationIssue",
    "ValidationResult",
    "combine_results",
    "gate_panel_validation",
    "validate_panel_budgets",
    "validate_panel_curvature",
]
