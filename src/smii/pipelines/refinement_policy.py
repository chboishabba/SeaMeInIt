"""Normalized, hashable policy for image-anchored body refinement."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounds(value: object, size: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (_finite(value, name),) * size
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be numeric or a sequence")
    result = tuple(_finite(item, name) for item in value)
    if len(result) != size:
        raise ValueError(f"{name} must contain {size} values")
    return result


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a string sequence")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} must contain only strings")
    return tuple(dict.fromkeys(value))


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Canonical JSON cannot contain non-finite values")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported canonical JSON value {type(value)!r}")


def canonical_hash(payload: Mapping[str, Any]) -> str:
    data = json.dumps(
        _json_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class RefinementPolicy:
    backend: str
    num_betas: int
    scale_measurement: str
    measurement_models: tuple[dict[str, object], ...]
    beta_lower: tuple[float, ...]
    beta_upper: tuple[float, ...]
    prior_weight: float
    anchor_weight: float
    max_beta_shift: float
    max_measurement_residual: float
    max_residual_degradation: float
    solver_tolerance: float
    solver_max_iterations: int
    abstain_on_warnings: tuple[str, ...]
    schema_version: str = "smii.body_refinement_policy.v1"
    solver: str = "box_coordinate_descent"

    def __post_init__(self) -> None:
        if self.schema_version != "smii.body_refinement_policy.v1":
            raise ValueError("Unsupported refinement policy schema")
        if not self.backend or not self.scale_measurement or self.num_betas <= 0:
            raise ValueError("Policy backend, scale measurement and beta count are required")
        if len(self.beta_lower) != self.num_betas or len(self.beta_upper) != self.num_betas:
            raise ValueError("Beta bounds must match num_betas")
        if any(lower >= upper for lower, upper in zip(self.beta_lower, self.beta_upper)):
            raise ValueError("Every beta lower bound must be below its upper bound")
        if self.prior_weight < 0 or self.anchor_weight < 0:
            raise ValueError("Prior and anchor weights must be non-negative")
        if self.prior_weight + self.anchor_weight <= 0:
            raise ValueError("At least one regularization weight must be positive")
        if self.solver_tolerance <= 0 or self.solver_max_iterations <= 0:
            raise ValueError("Solver tolerance and iteration limit must be positive")
        if any(
            value < 0
            for value in (
                self.max_beta_shift,
                self.max_measurement_residual,
                self.max_residual_degradation,
            )
        ):
            raise ValueError("Refinement thresholds must be non-negative")

    @classmethod
    def from_effective_config(
        cls,
        *,
        backend: str,
        num_betas: int,
        scale_measurement: str,
        models: Sequence[object],
        settings: Mapping[str, object] | None = None,
    ) -> "RefinementPolicy":
        settings = settings or {}
        rows: list[dict[str, object]] = []
        for model in models:
            weights = tuple(float(item) for item in getattr(model, "weights"))
            if len(weights) < num_betas:
                raise ValueError(f"Measurement model {getattr(model, 'name')!r} has too few weights")
            rows.append(
                {
                    "name": str(getattr(model, "name")),
                    "mean": _finite(getattr(model, "mean"), "measurement mean"),
                    "std": _finite(getattr(model, "std"), "measurement std"),
                    "weights": list(weights[:num_betas]),
                }
            )
        return cls(
            backend=backend,
            num_betas=num_betas,
            scale_measurement=scale_measurement,
            measurement_models=tuple(rows),
            beta_lower=_bounds(settings.get("beta_lower", -5.0), num_betas, "beta_lower"),
            beta_upper=_bounds(settings.get("beta_upper", 5.0), num_betas, "beta_upper"),
            prior_weight=_finite(settings.get("prior_weight", 0.05), "prior_weight"),
            anchor_weight=_finite(settings.get("anchor_weight", 0.5), "anchor_weight"),
            max_beta_shift=_finite(settings.get("max_beta_shift", 4.0), "max_beta_shift"),
            max_measurement_residual=_finite(
                settings.get("max_measurement_residual", 0.75), "max_measurement_residual"
            ),
            max_residual_degradation=_finite(
                settings.get("max_residual_degradation", 0.25), "max_residual_degradation"
            ),
            solver_tolerance=_finite(settings.get("solver_tolerance", 1e-10), "solver_tolerance"),
            solver_max_iterations=int(settings.get("solver_max_iterations", 5000)),
            abstain_on_warnings=_strings(
                settings.get(
                    "abstain_on_warnings",
                    ("WARN:low_view_diversity", "WARN:long_lens_flattening_risk"),
                ),
                "abstain_on_warnings",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "num_betas": self.num_betas,
            "scale_measurement": self.scale_measurement,
            "measurement_models": [dict(row) for row in self.measurement_models],
            "beta_lower": list(self.beta_lower),
            "beta_upper": list(self.beta_upper),
            "prior_weight": self.prior_weight,
            "anchor_weight": self.anchor_weight,
            "max_beta_shift": self.max_beta_shift,
            "max_measurement_residual": self.max_measurement_residual,
            "max_residual_degradation": self.max_residual_degradation,
            "solver": self.solver,
            "solver_tolerance": self.solver_tolerance,
            "solver_max_iterations": self.solver_max_iterations,
            "abstain_on_warnings": list(self.abstain_on_warnings),
        }

    @property
    def policy_hash(self) -> str:
        return canonical_hash(self.to_dict())
