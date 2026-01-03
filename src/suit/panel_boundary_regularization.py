"""Boundary regularization helpers for flattened panel outlines."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .panel_model import PanelBudgets


SEVERITY_BY_CODE: dict[str, str] = {
    "CURVATURE_EXCEEDED": "error",
    "MISSING_BUDGET": "error",
    "TURNING_BUDGET_EXCEEDED": "warning",
    "MIN_FEATURE_VIOLATION": "warning",
    "FEATURE_SUPPRESSED": "info",
    "SPLINE_FIT_APPLIED": "info",
    "SUGGEST_SPLIT": "info",
}
DEFAULT_SEVERITY = "warning"


def severity_for_code(code: str) -> str:
    return SEVERITY_BY_CODE.get(code, DEFAULT_SEVERITY)


@dataclass(frozen=True, slots=True)
class PanelIssue:
    code: str
    severity: str
    index: int | None = None
    value: float | None = None
    limit: float | None = None
    message: str | None = None

    @classmethod
    def from_code(
        cls,
        code: str,
        *,
        index: int | None = None,
        value: float | None = None,
        limit: float | None = None,
        message: str | None = None,
    ) -> "PanelIssue":
        return cls(
            code=code,
            severity=severity_for_code(code),
            index=index,
            value=value,
            limit=limit,
            message=message,
        )


@dataclass(frozen=True, slots=True)
class PanelIssueSummary:
    sewable: bool
    action: str


def summarize_panel_issues(issues: list[PanelIssue]) -> PanelIssueSummary:
    has_error = any(issue.severity == "error" for issue in issues)
    has_warning = any(issue.severity == "warning" for issue in issues)
    should_split = any(issue.code == "SUGGEST_SPLIT" for issue in issues)
    if should_split:
        action = "split"
    elif has_error or has_warning:
        action = "review"
    else:
        action = "ok"
    return PanelIssueSummary(sewable=not has_error, action=action)


def panel_issue_to_mapping(issue: PanelIssue) -> dict[str, object]:
    mapping: dict[str, object] = {
        "code": issue.code,
        "severity": issue.severity,
    }
    if issue.index is not None:
        mapping["index"] = issue.index
    if issue.value is not None:
        mapping["value"] = issue.value
    if issue.limit is not None:
        mapping["limit"] = issue.limit
    if issue.message:
        mapping["message"] = issue.message
    return mapping


def _euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _resample_by_arclength(
    points: list[tuple[float, float]],
    step: float,
) -> list[tuple[float, float]]:
    if len(points) < 3:
        return list(points)
    if step <= 0:
        return list(points)

    count = len(points)
    lengths = [_euclidean_distance(points[i], points[(i + 1) % count]) for i in range(count)]
    total_length = sum(lengths)
    if total_length <= 0:
        return list(points)

    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)

    samples: list[tuple[float, float]] = []
    distance = 0.0
    segment_index = 0
    while distance < total_length:
        while segment_index < count - 1 and distance > cumulative[segment_index + 1]:
            segment_index += 1
        segment_start = points[segment_index]
        segment_length = lengths[segment_index]
        if segment_length <= 0:
            distance += step
            continue
        seg_start_distance = cumulative[segment_index]
        t = (distance - seg_start_distance) / segment_length
        next_point = points[(segment_index + 1) % count]
        x = segment_start[0] + (next_point[0] - segment_start[0]) * t
        y = segment_start[1] + (next_point[1] - segment_start[1]) * t
        samples.append((x, y))
        distance += step

    return samples if len(samples) >= 3 else list(points)


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    norm_a = math.hypot(ax, ay)
    norm_b = math.hypot(bx, by)
    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return 0.0
    dot = (ax * bx + ay * by) / (norm_a * norm_b)
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


def _signed_turning_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


def _catmull_rom_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        2.0 * p1[0]
        + (-p0[0] + p2[0]) * t
        + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
        + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        2.0 * p1[1]
        + (-p0[1] + p2[1]) * t
        + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
        + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
    )
    return (x, y)


def _turning_event_indices(
    points: list[tuple[float, float]],
    eps: float,
) -> list[int]:
    count = len(points)
    dtheta: list[float] = []
    for idx in range(count):
        prev_point = points[(idx - 1) % count]
        curr_point = points[idx]
        next_point = points[(idx + 1) % count]
        v0 = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
        v1 = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])
        dtheta.append(_signed_turning_angle(v0, v1))

    turning_idxs: list[int] = []
    for idx in range(count):
        prev_theta = abs(dtheta[(idx - 1) % count])
        curr_theta = abs(dtheta[idx])
        next_theta = abs(dtheta[(idx + 1) % count])
        if curr_theta <= eps:
            continue
        if curr_theta > prev_theta and curr_theta > next_theta:
            turning_idxs.append(idx)

    if len(turning_idxs) < 2:
        nonzero = [idx for idx, theta in enumerate(dtheta) if abs(theta) > eps]
        for pos in range(1, len(nonzero)):
            prev_idx = nonzero[pos - 1]
            curr_idx = nonzero[pos]
            if dtheta[prev_idx] * dtheta[curr_idx] < 0:
                turning_idxs.append(curr_idx)

    turning_idxs.sort()
    return turning_idxs


def _curvature_at(points: list[tuple[float, float]], index: int) -> float:
    count = len(points)
    prev_point = points[(index - 1) % count]
    curr_point = points[index % count]
    next_point = points[(index + 1) % count]
    v1 = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
    v2 = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])
    step = 0.5 * (
        _euclidean_distance(prev_point, curr_point) + _euclidean_distance(curr_point, next_point)
    )
    if step <= 1e-9:
        return 0.0
    theta = _angle_between(v1, v2)
    return abs(theta) / step


def fit_spline_boundary(
    points_xy: list[tuple[float, float]],
    step_mm: float,
    budgets: PanelBudgets,
) -> tuple[list[tuple[float, float]], bool]:
    if len(points_xy) < 4:
        return list(points_xy), False
    if step_mm <= 0:
        return list(points_xy), False

    points = points_xy
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 4:
        return list(points_xy), False

    samples_per_segment = 3
    count = len(points)
    sampled: list[tuple[float, float]] = []
    for idx in range(count):
        p0 = points[(idx - 1) % count]
        p1 = points[idx]
        p2 = points[(idx + 1) % count]
        p3 = points[(idx + 2) % count]
        for sample_idx in range(samples_per_segment):
            t = sample_idx / samples_per_segment
            sampled.append(_catmull_rom_point(p0, p1, p2, p3, t))

    step = step_mm / 1000.0
    resampled = _resample_by_arclength(sampled, step)
    if len(resampled) < 3:
        return list(points_xy), False

    if budgets.curvature_min_radius is not None:
        kappa_max = 1.0 / float(budgets.curvature_min_radius)
        for idx in range(len(resampled)):
            if _curvature_at(resampled, idx) > kappa_max:
                return list(points_xy), False

    return resampled, True


def detect_turning_budget(
    points_xy: list[tuple[float, float]],
    step_mm: float,
    budgets: PanelBudgets,
) -> list[PanelIssue]:
    issues: list[PanelIssue] = []
    if len(points_xy) < 3:
        return issues
    if budgets.turning_max_per_length is None:
        return issues
    if step_mm <= 0:
        return issues

    points = points_xy
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        return issues

    count = len(points)
    dtheta: list[float] = []
    for idx in range(count):
        prev_point = points[(idx - 1) % count]
        curr_point = points[idx]
        next_point = points[(idx + 1) % count]
        v0 = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
        v1 = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])
        dtheta.append(abs(_signed_turning_angle(v0, v1)))

    window_mm = 40.0
    window_len = max(1, int(window_mm / step_mm))
    window_len = min(window_len, len(dtheta))
    # turning_max_per_length is radians per meter; convert 40mm window to radians.
    max_deg = math.degrees(budgets.turning_max_per_length * 0.04)
    extended = dtheta + dtheta[: window_len - 1]
    for idx in range(len(dtheta)):
        window_sum = sum(extended[idx : idx + window_len])
        window_deg = math.degrees(window_sum)
        if window_deg > max_deg:
            center_idx = idx + window_len // 2
            issues.append(
                PanelIssue.from_code(
                    "TURNING_BUDGET_EXCEEDED",
                    index=center_idx % count,
                    value=window_deg,
                    limit=max_deg,
                )
            )
            break

    return issues


def _turning_budget_violation_indices(
    points_xy: list[tuple[float, float]],
    step_mm: float,
    budgets: PanelBudgets,
) -> list[int]:
    if len(points_xy) < 3:
        return []
    if budgets.turning_max_per_length is None:
        return []
    if step_mm <= 0:
        return []

    points = points_xy
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        return []

    count = len(points)
    dtheta: list[float] = []
    for idx in range(count):
        prev_point = points[(idx - 1) % count]
        curr_point = points[idx]
        next_point = points[(idx + 1) % count]
        v0 = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
        v1 = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])
        dtheta.append(abs(_signed_turning_angle(v0, v1)))

    window_mm = 40.0
    window_len = max(1, int(window_mm / step_mm))
    window_len = min(window_len, len(dtheta))
    max_deg = math.degrees(budgets.turning_max_per_length * 0.04)
    extended = dtheta + dtheta[: window_len - 1]

    indices: list[int] = []
    for idx in range(len(dtheta)):
        window_sum = sum(extended[idx : idx + window_len])
        window_deg = math.degrees(window_sum)
        if window_deg > max_deg:
            indices.append(idx)

    if not indices:
        return []

    distinct = [indices[0]]
    for idx in indices[1:]:
        if idx - distinct[-1] >= window_len:
            distinct.append(idx)
    return distinct


def detect_min_feature_size(
    points_xy: list[tuple[float, float]],
    step_mm: float,
    budgets: PanelBudgets,
) -> list[PanelIssue]:
    issues: list[PanelIssue] = []
    if len(points_xy) < 4:
        return issues
    if budgets.min_feature_size is None:
        return issues
    if step_mm <= 0:
        return issues

    points = points_xy
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 4:
        return issues

    eps = math.radians(2.0)
    turning_idxs = _turning_event_indices(points, eps)

    if len(turning_idxs) < 2:
        return issues

    min_feature_mm = budgets.min_feature_size * 1000.0
    count = len(points)
    wrapped = turning_idxs + [turning_idxs[0] + count]
    for idx, next_idx in zip(wrapped, wrapped[1:]):
        feature_len_mm = (next_idx - idx) * step_mm
        if feature_len_mm < min_feature_mm:
            issues.append(
                PanelIssue.from_code(
                    "MIN_FEATURE_VIOLATION",
                    index=idx,
                    value=feature_len_mm,
                    limit=min_feature_mm,
                )
            )
            break

    return issues


def suppress_min_features(
    points_xy: list[tuple[float, float]],
    step_mm: float,
    budgets: PanelBudgets,
) -> tuple[list[tuple[float, float]], bool, int | None, float | None]:
    if len(points_xy) < 4:
        return list(points_xy), False, None, None
    if budgets.min_feature_size is None:
        return list(points_xy), False, None, None
    if step_mm <= 0:
        return list(points_xy), False, None, None

    points = points_xy
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 4:
        return list(points), False, None, None

    eps = math.radians(2.0)
    turning_idxs = _turning_event_indices(points, eps)
    if len(turning_idxs) < 2:
        return list(points), False, None, None

    min_feature_mm = budgets.min_feature_size * 1000.0
    count = len(points)
    remove: set[int] = set()
    wrapped = turning_idxs + [turning_idxs[0] + count]
    suppression_index: int | None = None
    suppression_span_mm: float | None = None
    for idx, next_idx in zip(wrapped, wrapped[1:]):
        feature_len_mm = (next_idx - idx) * step_mm
        if feature_len_mm >= min_feature_mm:
            continue
        start = idx + 1
        end = next_idx
        for remove_idx in range(start, end + 1):
            remove.add(remove_idx % count)
        suppression_index = (idx + next_idx) // 2 % count
        suppression_span_mm = feature_len_mm
        break

    if not remove:
        return list(points), False, None, None

    suppressed = [point for idx, point in enumerate(points) if idx not in remove]
    if len(suppressed) >= 3:
        return suppressed, True, suppression_index, suppression_span_mm
    return list(points_xy), False, None, None


def suggest_split_issue(
    issues: list[PanelIssue],
    turning_violation_indices: list[int],
    *,
    min_feature_persisted: bool,
) -> PanelIssue | None:
    reasons: list[str] = []
    codes = {issue.code for issue in issues}
    if "CURVATURE_EXCEEDED" in codes and "FEATURE_SUPPRESSED" in codes:
        reasons.append("curvature+suppression")
    if len(turning_violation_indices) >= 2:
        reasons.append("repeated_turning")
    if min_feature_persisted:
        reasons.append("min_feature_persisted")
    if not reasons:
        return None

    index: int | None = None
    if turning_violation_indices:
        index = turning_violation_indices[0]
    else:
        for issue in issues:
            if issue.index is not None:
                index = issue.index
                break

    return PanelIssue.from_code(
        "SUGGEST_SPLIT",
        index=index,
        message="; ".join(reasons),
    )


def regularize_boundary(
    boundary_xy: list[tuple[float, float]],
    budgets: PanelBudgets,
) -> tuple[list[tuple[float, float]], list[PanelIssue]]:
    """Apply arc-length resampling and curvature clamping to a 2D boundary."""

    if not boundary_xy:
        return [], []
    if budgets.curvature_min_radius is None:
        return list(boundary_xy), [PanelIssue.from_code("MISSING_BUDGET")]

    # Fallback to a 3mm step when no min feature size is provided.
    min_feature = budgets.min_feature_size if budgets.min_feature_size is not None else 0.009
    step = min(min_feature / 3.0, 0.003)
    resampled = _resample_by_arclength(list(boundary_xy), step)
    if len(resampled) < 3:
        return resampled, []

    issues: list[PanelIssue] = []
    kappa_max = 1.0 / float(budgets.curvature_min_radius)

    for idx in range(len(resampled)):
        kappa = _curvature_at(resampled, idx)
        if kappa <= kappa_max:
            continue
        issues.append(
            PanelIssue.from_code(
                "CURVATURE_EXCEEDED",
                index=idx,
                value=kappa,
                limit=kappa_max,
            )
        )
        prev_point = resampled[(idx - 1) % len(resampled)]
        next_point = resampled[(idx + 1) % len(resampled)]
        resampled[idx] = (
            0.5 * (prev_point[0] + next_point[0]),
            0.5 * (prev_point[1] + next_point[1]),
        )

    step_mm = step * 1000.0
    issues.extend(detect_turning_budget(resampled, step_mm, budgets))
    issues.extend(detect_min_feature_size(resampled, step_mm, budgets))
    resampled, suppressed, suppression_index, suppression_span_mm = suppress_min_features(
        resampled,
        step_mm,
        budgets,
    )
    if suppressed:
        issues.append(
            PanelIssue.from_code(
                "FEATURE_SUPPRESSED",
                index=suppression_index,
                value=suppression_span_mm,
                limit=budgets.min_feature_size * 1000.0 if budgets.min_feature_size else None,
                message="Suppressed features below min feature size.",
            )
        )
    min_feature_persisted = bool(detect_min_feature_size(resampled, step_mm, budgets))
    turning_violation_indices = _turning_budget_violation_indices(
        resampled,
        step_mm,
        budgets,
    )
    suggestion = suggest_split_issue(
        issues,
        turning_violation_indices,
        min_feature_persisted=min_feature_persisted,
    )
    if suggestion is not None:
        issues.append(suggestion)

    summary = summarize_panel_issues(issues)
    has_split = any(issue.code == "SUGGEST_SPLIT" for issue in issues)
    if summary.action == "ok" and not has_split:
        resampled, applied = fit_spline_boundary(resampled, step_mm, budgets)
        if applied:
            issues.append(
                PanelIssue.from_code(
                    "SPLINE_FIT_APPLIED",
                    message="Applied spline fitting to panel outline.",
                )
            )

    return resampled, issues


__all__ = [
    "PanelIssue",
    "PanelIssueSummary",
    "SEVERITY_BY_CODE",
    "panel_issue_to_mapping",
    "regularize_boundary",
    "severity_for_code",
    "suggest_split_issue",
    "summarize_panel_issues",
]
