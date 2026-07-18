from __future__ import annotations

import numpy as np

from smii.pipelines.refinement_authority import build_receipt, solve_bounded_refinement
from smii.pipelines.refinement_policy import RefinementPolicy


class MeasurementRow:
    def __init__(self, name: str, weights: tuple[float, ...]) -> None:
        self.name = name
        self.mean = 0.0
        self.std = 1.0
        self.weights = weights


def policy(**overrides: object) -> RefinementPolicy:
    settings: dict[str, object] = {
        "beta_lower": -2.0,
        "beta_upper": 2.0,
        "prior_weight": 0.1,
        "anchor_weight": 1.0,
        "max_beta_shift": 3.0,
        "max_measurement_residual": 10.0,
        "max_residual_degradation": 10.0,
        "solver_tolerance": 1e-12,
        "solver_max_iterations": 1000,
    }
    settings.update(overrides)
    return RefinementPolicy.from_effective_config(
        backend="test",
        num_betas=2,
        scale_measurement="m0",
        models=(
            MeasurementRow("m0", (1.0, 0.0)),
            MeasurementRow("m1", (0.0, 1.0)),
        ),
        settings=settings,
    )


def test_policy_hash_changes_with_effective_settings() -> None:
    assert policy().policy_hash != policy(anchor_weight=2.0).policy_hash


def test_solver_enforces_bounds_during_optimization() -> None:
    result = solve_bounded_refinement(
        np.eye(2),
        np.array([100.0, -100.0]),
        np.zeros(2),
        policy(),
    )

    np.testing.assert_allclose(result.betas, [2.0, -2.0])
    assert result.converged


def test_reference_warning_abstains_and_preserves_input_hash() -> None:
    refinement_policy = policy()
    anchor = np.zeros(2)
    result = solve_bounded_refinement(np.eye(2), anchor, anchor, refinement_policy)

    receipt = build_receipt(
        policy=refinement_policy,
        measurements={"m0": 0.0, "m1": 0.0},
        names=("m0", "m1"),
        anchor_betas=anchor,
        solution=result,
        warnings=("WARN:low_view_diversity",),
        severity="warn",
    )

    assert receipt.decision == "abstain"
    assert receipt.selected_output_hash == receipt.input_hash


def test_unrelated_warning_stays_visible_without_veto() -> None:
    refinement_policy = policy()
    anchor = np.zeros(2)
    result = solve_bounded_refinement(np.eye(2), anchor, anchor, refinement_policy)

    receipt = build_receipt(
        policy=refinement_policy,
        measurements={"m0": 0.0, "m1": 0.0},
        names=("m0", "m1"),
        anchor_betas=anchor,
        solution=result,
        warnings=("WARN:diagnostic_only",),
        severity="warn",
    )

    assert receipt.decision == "promote"
    assert receipt.warnings == ("WARN:diagnostic_only",)
