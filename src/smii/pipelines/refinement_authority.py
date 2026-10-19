"""Bounded refinement solver and hash-linked authority receipt."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np

from .refinement_policy import RefinementPolicy, canonical_hash

Decision = Literal["promote", "abstain", "reject"]
Severity = Literal["pass", "warn", "fail"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RefinementSolution:
    betas: np.ndarray
    measurement_residual: float
    anchor_measurement_residual: float
    beta_shift: float
    prior_cost: float
    anchor_cost: float
    active_lower: tuple[int, ...]
    active_upper: tuple[int, ...]
    iterations: int
    converged: bool
    kkt_residual: float

    def evidence(self) -> dict[str, object]:
        return {
            "candidate_betas": np.asarray(self.betas, dtype=float).reshape(-1).tolist(),
            "measurement_residual": self.measurement_residual,
            "anchor_measurement_residual": self.anchor_measurement_residual,
            "residual_delta": self.measurement_residual - self.anchor_measurement_residual,
            "beta_shift": self.beta_shift,
            "prior_cost": self.prior_cost,
            "anchor_cost": self.anchor_cost,
            "active_lower": list(self.active_lower),
            "active_upper": list(self.active_upper),
            "iterations": self.iterations,
            "converged": self.converged,
            "kkt_residual": self.kkt_residual,
        }


def _kkt(
    hessian: np.ndarray,
    linear: np.ndarray,
    betas: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float,
) -> float:
    gradient = hessian @ betas - linear
    at_lower = betas <= lower + tolerance
    at_upper = betas >= upper - tolerance
    free = ~(at_lower | at_upper)
    violations = np.zeros_like(gradient)
    violations[free] = np.abs(gradient[free])
    violations[at_lower] = np.maximum(0.0, -gradient[at_lower])
    violations[at_upper] = np.maximum(0.0, gradient[at_upper])
    return float(np.max(violations)) if violations.size else 0.0


def solve_bounded_refinement(
    matrix: np.ndarray,
    target: np.ndarray,
    anchor_betas: np.ndarray,
    policy: RefinementPolicy,
) -> RefinementSolution:
    matrix = np.asarray(matrix, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)
    anchor = np.asarray(anchor_betas, dtype=float).reshape(-1)
    if matrix.shape != (target.size, policy.num_betas) or anchor.size != policy.num_betas:
        raise ValueError("Refinement matrix, target and anchor dimensions do not match policy")
    if not all(np.all(np.isfinite(value)) for value in (matrix, target, anchor)):
        raise ValueError("Refinement inputs must be finite")

    lower = np.asarray(policy.beta_lower)
    upper = np.asarray(policy.beta_upper)
    hessian = matrix.T @ matrix + (
        policy.prior_weight + policy.anchor_weight
    ) * np.eye(policy.num_betas)
    linear = matrix.T @ target + policy.anchor_weight * anchor
    betas = np.clip(anchor, lower, upper)
    converged = False
    iterations = 0

    for iteration in range(1, policy.solver_max_iterations + 1):
        max_delta = 0.0
        for index in range(policy.num_betas):
            diagonal = float(hessian[index, index])
            if diagonal <= 0 or not math.isfinite(diagonal):
                raise ValueError("Refinement Hessian is not positive on its diagonal")
            numerator = float(
                linear[index] - np.dot(hessian[index], betas) + diagonal * betas[index]
            )
            next_value = min(upper[index], max(lower[index], numerator / diagonal))
            max_delta = max(max_delta, abs(next_value - betas[index]))
            betas[index] = next_value
        iterations = iteration
        kkt = _kkt(hessian, linear, betas, lower, upper, policy.solver_tolerance)
        if max_delta <= policy.solver_tolerance and kkt <= policy.solver_tolerance * 10:
            converged = True
            break

    candidate_residual = float(np.sqrt(np.mean((matrix @ betas - target) ** 2)))
    anchor_residual = float(np.sqrt(np.mean((matrix @ anchor - target) ** 2)))
    tolerance = policy.solver_tolerance * 10
    return RefinementSolution(
        betas=betas.copy(),
        measurement_residual=candidate_residual,
        anchor_measurement_residual=anchor_residual,
        beta_shift=float(np.linalg.norm(betas - anchor)),
        prior_cost=float(policy.prior_weight * np.dot(betas, betas)),
        anchor_cost=float(policy.anchor_weight * np.dot(betas - anchor, betas - anchor)),
        active_lower=tuple(int(i) for i in np.flatnonzero(np.isclose(betas, lower, atol=tolerance))),
        active_upper=tuple(int(i) for i in np.flatnonzero(np.isclose(betas, upper, atol=tolerance))),
        iterations=iterations,
        converged=converged,
        kkt_residual=_kkt(hessian, linear, betas, lower, upper, policy.solver_tolerance),
    )


def decide(
    solution: RefinementSolution,
    policy: RefinementPolicy,
    warnings: Sequence[str],
    severity: Severity,
) -> tuple[Decision, tuple[str, ...]]:
    reject: list[str] = []
    if severity == "fail":
        reject.append("diagnostic_failure")
    if not solution.converged:
        reject.append("solver_not_converged")
    if not np.all(np.isfinite(solution.betas)):
        reject.append("candidate_betas_non_finite")
    if reject:
        return "reject", tuple(reject)

    abstain: list[str] = []
    if solution.beta_shift > policy.max_beta_shift:
        abstain.append("beta_shift_exceeds_policy")
    if solution.measurement_residual > policy.max_measurement_residual:
        abstain.append("measurement_residual_exceeds_policy")
    if (
        solution.measurement_residual - solution.anchor_measurement_residual
        > policy.max_residual_degradation
    ):
        abstain.append("measurement_residual_degrades_beyond_policy")
    if any(warning in policy.abstain_on_warnings for warning in warnings):
        abstain.append("reference_quality_insufficient_for_refinement")
    return ("abstain", tuple(dict.fromkeys(abstain))) if abstain else ("promote", ())


@dataclass(frozen=True, slots=True)
class RefinementReceipt:
    policy_hash: str
    input_hash: str
    candidate_hash: str
    selected_output_hash: str
    decision: Decision
    diagnostic_severity: Severity
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    input_evidence: Mapping[str, object]
    candidate_evidence: Mapping[str, object]
    schema_version: str = "smii.body_refinement_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "smii.body_refinement_receipt.v1":
            raise ValueError("Unsupported refinement receipt schema")
        if any(_SHA256.fullmatch(value) is None for value in (
            self.policy_hash, self.input_hash, self.candidate_hash, self.selected_output_hash
        )):
            raise ValueError("Refinement receipt hashes must be lowercase SHA-256 digests")
        expected = self.candidate_hash if self.decision == "promote" else self.input_hash
        if self.selected_output_hash != expected:
            raise ValueError("Receipt output selection does not match its decision")
        if self.decision == "promote" and self.blockers:
            raise ValueError("Promoted refinement cannot contain blockers")
        if self.decision != "promote" and not self.blockers:
            raise ValueError("Abstained or rejected refinement requires blockers")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_hash": self.policy_hash,
            "input_hash": self.input_hash,
            "candidate_hash": self.candidate_hash,
            "selected_output_hash": self.selected_output_hash,
            "decision": self.decision,
            "diagnostic_severity": self.diagnostic_severity,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "input_evidence": dict(self.input_evidence),
            "candidate_evidence": dict(self.candidate_evidence),
        }

    @property
    def receipt_hash(self) -> str:
        return canonical_hash(self.to_dict())


def build_receipt(
    *,
    policy: RefinementPolicy,
    measurements: Mapping[str, float],
    names: Sequence[str],
    anchor_betas: np.ndarray,
    solution: RefinementSolution,
    warnings: Sequence[str] = (),
    severity: Severity = "pass",
    input_context: Mapping[str, object] | None = None,
    candidate_context: Mapping[str, object] | None = None,
) -> RefinementReceipt:
    input_evidence: dict[str, object] = {
        "measurements": {name: float(measurements[name]) for name in sorted(names)},
        "measurements_used": list(names),
        "anchor_betas": np.asarray(anchor_betas, dtype=float).reshape(-1).tolist(),
    }
    candidate_evidence = solution.evidence()
    if input_context:
        input_evidence["context"] = dict(input_context)
    if candidate_context:
        candidate_evidence["context"] = dict(candidate_context)
    warning_values = tuple(dict.fromkeys(str(item) for item in warnings))
    decision, blockers = decide(solution, policy, warning_values, severity)
    input_hash = canonical_hash(input_evidence)
    candidate_hash = canonical_hash(candidate_evidence)
    return RefinementReceipt(
        policy_hash=policy.policy_hash,
        input_hash=input_hash,
        candidate_hash=candidate_hash,
        selected_output_hash=candidate_hash if decision == "promote" else input_hash,
        decision=decision,
        diagnostic_severity=severity,
        blockers=blockers,
        warnings=warning_values,
        input_evidence=input_evidence,
        candidate_evidence=candidate_evidence,
    )
