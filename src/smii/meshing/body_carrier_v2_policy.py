"""Policy primitives for final-export body authorization."""

from __future__ import annotations

import math
import re
from typing import Literal, Mapping

BodyDecision = Literal["promote", "abstain", "reject"]
CanonicalSource = Literal["raw_image_fit", "refined_candidate"]
Severity = Literal["pass", "warn", "fail"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_BLOCKED_CONSUMERS = (
    "undersuit",
    "hard_shell",
    "panel_transfer",
    "pattern_export",
    "manufacturing_export",
)


def validate_hash(value: object, name: str) -> str:
    result = str(value)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def finite_nonnegative(
    value: object,
    name: str,
    *,
    maximum: float | None = None,
) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def validate_topology(value: Mapping[str, object], name: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for key in ("vertex_count", "face_count"):
        try:
            count = int(value[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{name}.{key} must be a positive integer") from exc
        if count <= 0:
            raise ValueError(f"{name}.{key} must be positive")
        result[key] = count
    return result


def decide_body_authorization(
    *,
    trust_level: str,
    severity: Severity,
    confidence: float,
    skull_residual: float,
    geometry_finite: bool,
    topology_valid: bool,
    confidence_threshold: float = 0.75,
    skull_threshold: float = 0.35,
) -> tuple[BodyDecision, tuple[str, ...]]:
    """Authorize the consumed final artifact, not merely its pre-repair source."""

    rejected = []
    if severity == "fail":
        rejected.append("diagnostic_failure")
    if not geometry_finite:
        rejected.append("final_export_non_finite")
    if not topology_valid:
        rejected.append("final_export_topology_invalid")
    if rejected:
        return "reject", tuple(rejected)

    abstained = []
    if trust_level != "high":
        abstained.append("body_trust_below_high")
    if confidence < confidence_threshold:
        abstained.append("body_fit_confidence_below_threshold")
    if skull_residual > skull_threshold:
        abstained.append("final_skull_residual_exceeds_threshold")
    return ("abstain", tuple(abstained)) if abstained else ("promote", ())
