"""Geometric referee and Gate-7 manufacturing validation for garment pattern layouts."""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon

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
