"""Deterministic 2D sheet nesting for garment cutting layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from shapely.geometry import box

from smii.garment.pattern_model import PatternChart
from .allowances import compute_seam_allowance_offset


@dataclass(frozen=True)
class NestingConfig:
    fabric_width_mm: float = 1500.0  # 150 cm fabric standard width
    clearance_mm: float = 10.0
    allowance_mm: float = 10.0
    allowed_rotations_deg: tuple[int, ...] = (0, 180)
    allow_mirroring: bool = False
    deterministic_seed: int = 0


@dataclass(frozen=True)
class Placement:
    chart_id: str
    x_mm: float
    y_mm: float
    rotation_deg: int
    mirrored: bool
    bounds_mm: tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    offset_contour_mm: np.ndarray  # Shifted to placed coordinate


@dataclass(frozen=True)
class NestingResult:
    placements: tuple[Placement, ...]
    fabric_length_mm: float
    fabric_width_mm: float
    utilization: float
    collision_count: int
    unplaced_chart_ids: tuple[str, ...]


def nest_charts_deterministic(
    charts: Sequence[PatternChart],
    config: NestingConfig | None = None,
) -> NestingResult:
    """Pack pattern charts onto a continuous fabric roll using deterministic shelf packing."""
    cfg = config or NestingConfig()
    fabric_w = cfg.fabric_width_mm
    clearance = cfg.clearance_mm
    allowance = cfg.allowance_mm

    # Extract cut outline (with seam allowance) for each chart
    chart_data = []
    total_chart_area = 0.0

    for chart in charts:
        if len(chart.boundary_loops) > 0 and len(chart.uv_mm) > 0:
            loop_indices = chart.boundary_loops[0]
            pts = chart.uv_mm[loop_indices]
            try:
                cut_pts = compute_seam_allowance_offset(pts, allowance)
            except Exception:
                cut_pts = pts
        else:
            cut_pts = chart.uv_mm

        min_x, min_y = float(cut_pts[:, 0].min()), float(cut_pts[:, 1].min())
        max_x, max_y = float(cut_pts[:, 0].max()), float(cut_pts[:, 1].max())
        w = max(max_x - min_x, 10.0)
        h = max(max_y - min_y, 10.0)

        # Normalize piece origin to (0, 0)
        norm_pts = cut_pts.copy()
        norm_pts[:, 0] -= min_x
        norm_pts[:, 1] -= min_y

        # Approximate piece area via shoelace formula
        area = 0.5 * abs(
            float(
                np.dot(norm_pts[:, 0], np.roll(norm_pts[:, 1], 1))
                - np.dot(norm_pts[:, 1], np.roll(norm_pts[:, 0], 1))
            )
        )
        total_chart_area += area

        chart_data.append(
            {
                "chart": chart,
                "norm_pts": norm_pts,
                "width": w,
                "height": h,
                "area": area,
            }
        )

    # Sort descending by height for optimal shelf packing
    sorted_items = sorted(chart_data, key=lambda item: item["height"], reverse=True)

    curr_x = clearance
    curr_y = clearance
    shelf_height = 0.0
    placements: list[Placement] = []
    unplaced: list[str] = []

    for item in sorted_items:
        w = item["width"]
        h = item["height"]
        cid = item["chart"].chart_id

        # Check if exceeds fabric width
        if w + 2 * clearance > fabric_w:
            unplaced.append(cid)
            continue

        # Check if needs new shelf
        if curr_x + w + clearance > fabric_w:
            curr_x = clearance
            curr_y += shelf_height + clearance
            shelf_height = 0.0

        # Place piece at (curr_x, curr_y)
        placed_pts = item["norm_pts"].copy()
        placed_pts[:, 0] += curr_x
        placed_pts[:, 1] += curr_y

        p_min_x = curr_x
        p_min_y = curr_y
        p_max_x = curr_x + w
        p_max_y = curr_y + h

        placements.append(
            Placement(
                chart_id=cid,
                x_mm=curr_x,
                y_mm=curr_y,
                rotation_deg=0,
                mirrored=False,
                bounds_mm=(p_min_x, p_min_y, p_max_x, p_max_y),
                offset_contour_mm=placed_pts,
            )
        )

        curr_x += w + clearance
        shelf_height = max(shelf_height, h)

    total_length = curr_y + shelf_height + clearance if placements else 0.0
    total_sheet_area = fabric_w * total_length if total_length > 0 else 1.0
    utilization = min(1.0, max(0.0, total_chart_area / total_sheet_area))

    # Collision test using Shapely bounding boxes
    collision_count = 0
    boxes = [box(*p.bounds_mm) for p in placements]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].intersection(boxes[j]).area > 1e-4:
                collision_count += 1

    return NestingResult(
        placements=tuple(placements),
        fabric_length_mm=total_length,
        fabric_width_mm=fabric_w,
        utilization=utilization,
        collision_count=collision_count,
        unplaced_chart_ids=tuple(unplaced),
    )


__all__ = [
    "NestingConfig",
    "NestingResult",
    "Placement",
    "nest_charts_deterministic",
]
