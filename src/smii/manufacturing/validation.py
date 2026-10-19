"""Geometric referee and Gate-7 manufacturing validation for garment pattern layouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, Polygon

from smii.garment.pattern_model import MarkKind

from smii.garment.pattern_model import FullSuitPattern
from .nesting import NestingResult


@dataclass(frozen=True)
class ManufacturingValidationReport:
    """Quantitative empirical report certifying geometric manufacturing validity."""

    is_valid: bool
    total_charts: int
    placed_charts: int
    unplaced_count: int
    collision_count: int
    self_intersection_count: int
    max_seam_error_relative: float
    gate7_promoted: int  # 1 for accepted, 0 for blocked
    blocking_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "total_charts": self.total_charts,
            "placed_charts": self.placed_charts,
            "unplaced_count": self.unplaced_count,
            "collision_count": self.collision_count,
            "self_intersection_count": self.self_intersection_count,
            "max_seam_error_relative": self.max_seam_error_relative,
            "gate7_promoted": self.gate7_promoted,
            "blocking_reasons": list(self.blocking_reasons),
        }


def validate_manufacturing_layout(
    pattern: FullSuitPattern,
    nesting: NestingResult,
    max_permitted_seam_error: float = 0.50,
) -> ManufacturingValidationReport:
    """Validate cutting contours, seam alignment, and nesting layout for Gate 7."""
    blockers: list[str] = []
    total_charts = len(pattern.charts)
    placed_charts = len(nesting.placements)
    unplaced_count = len(nesting.unplaced_chart_ids)

    if unplaced_count > 0:
        blockers.append(
            f"{unplaced_count} charts could not fit on fabric roll width {nesting.fabric_width_mm}mm."
        )

    # 1. Collision count
    collision_count = nesting.collision_count
    if collision_count > 0:
        blockers.append(f"Detected {collision_count} overlapping pieces in nested fabric layout.")

    # 2. Polygon self-intersection validation
    self_intersection_count = 0
    for p in nesting.placements:
        poly = Polygon(p.offset_contour_mm)
        if not poly.is_valid:
            self_intersection_count += 1
            blockers.append(f"Chart '{p.chart_id}' cutting contour has self-intersections.")

    # 3. Fail closed on unresolved sew-line length verification.
    max_seam_err = 0.0
    chart_map = {chart.chart_id: chart for chart in pattern.charts}
    from smii.garment.seam_topology import SeamTopologyExtractor

    for seam in pattern.seams:
        left = chart_map.get(seam.left.chart_id)
        right = chart_map.get(seam.right.chart_id)
        if left is None or right is None:
            blockers.append(f"Missing chart for seam {seam.seam_id}.")
            continue
        err = SeamTopologyExtractor.compute_uv_seam_error(seam, left, right)
        max_seam_err = max(max_seam_err, err)
        if err > max_permitted_seam_error:
            blockers.append(f"UV sew-line mismatch for {seam.seam_id}: {err:.6f}.")

    # 4. Physical dart construction marks are mandatory when a dart is present.
    for chart in pattern.charts:
        if not chart.features:
            continue
        if not chart.boundary_loops:
            blockers.append(f"Chart '{chart.chart_id}' has no boundary for feature validation.")
            continue
        polygon = Polygon(chart.uv_mm[chart.boundary_loops[0]])
        if not polygon.is_valid:
            blockers.append(f"Chart '{chart.chart_id}' has invalid UV feature boundary.")
            continue
        for feature in chart.features:
            if feature.kind != "dart":
                continue
            sew = [m for m in feature.marks if m.kind == MarkKind.SEW]
            fold = [m for m in feature.marks if m.kind == MarkKind.FOLD]
            drills = [m for m in feature.marks if m.kind == MarkKind.DRILL]
            if len(sew) != 2 or len(fold) != 1 or len(drills) != 1:
                blockers.append(f"Incomplete dart feature '{feature.feature_id}'.")
                continue
            leg_lengths = [
                float(np.linalg.norm(np.diff(np.asarray(m.points_mm), axis=0), axis=1).sum())
                for m in sew
            ]
            if abs(leg_lengths[0] - leg_lengths[1]) > 1.0:
                blockers.append(f"Dart '{feature.feature_id}' leg mismatch exceeds 1 mm.")
            if not np.allclose(sew[0].points_mm[-1], sew[1].points_mm[-1], atol=1e-6):
                blockers.append(f"Dart '{feature.feature_id}' sew legs do not share apex.")
            drill = Point(drills[0].points_mm[0])
            if not polygon.contains(drill) or polygon.exterior.distance(drill) <= 3.0:
                blockers.append(f"Dart '{feature.feature_id}' drill violates 3 mm clearance.")

    is_valid = len(blockers) == 0
    gate7_promoted = 1 if is_valid else 0

    return ManufacturingValidationReport(
        is_valid=is_valid,
        total_charts=total_charts,
        placed_charts=placed_charts,
        unplaced_count=unplaced_count,
        collision_count=collision_count,
        self_intersection_count=self_intersection_count,
        max_seam_error_relative=max_seam_err,
        gate7_promoted=gate7_promoted,
        blocking_reasons=tuple(blockers),
    )


__all__ = [
    "ManufacturingValidationReport",
    "validate_manufacturing_layout",
]
