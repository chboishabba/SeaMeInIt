"""Seam allowance geometry construction and outward polygon buffering."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

ALLOWANCE_BY_METHOD: dict[str, float] = {
    "home_sewing": 15.0,  # 15 mm
    "overlock": 10.0,  # 10 mm
    "flatlock": 8.0,  # 8 mm
    "bonded": 5.0,  # 5 mm
    "welded": 3.0,  # 3 mm
    "laser_cut": 2.0,  # 2 mm
}


def compute_seam_allowance_offset(
    boundary_points: np.ndarray,
    allowance_mm: float,
    join_style: str = "mitre",
    mitre_limit: float = 2.0,
) -> np.ndarray:
    """Compute exact outward polygon offset with seam allowance using Shapely.

    Parameters
    ----------
    boundary_points : np.ndarray
        Array of shape (N, 2) defining the ordered boundary loop in mm.
    allowance_mm : float
        Outward seam allowance offset distance in mm.
    join_style : str
        Join style ('mitre', 'round', or 'bevel').
    mitre_limit : float
        Maximum ratio of mitre distance to offset distance.

    Returns
    -------
    np.ndarray
        Array of shape (M, 2) containing the offset cutting boundary loop.
    """
    pts = np.asarray(boundary_points, dtype=np.float64)
    if len(pts) < 3:
        raise ValueError("Boundary loop must contain at least 3 points.")

    poly = Polygon(pts)
    if not poly.is_valid:
        # Buffer(0) cleans minor self-intersections
        poly = poly.buffer(0)

    if allowance_mm <= 0.0:
        if poly.exterior is not None:
            return np.asarray(poly.exterior.coords[:-1], dtype=np.float64)
        return pts

    # Outward buffer
    offset_poly = poly.buffer(
        allowance_mm,
        join_style=join_style,
        mitre_limit=mitre_limit,
    )

    if offset_poly.is_empty:
        raise ValueError("Seam allowance offset resulted in empty geometry.")

    # In case buffering creates a MultiPolygon, select the largest component by area
    if offset_poly.geom_type == "MultiPolygon":
        largest = max(offset_poly.geoms, key=lambda g: g.area)
        coords = np.asarray(largest.exterior.coords[:-1], dtype=np.float64)
    else:
        coords = np.asarray(offset_poly.exterior.coords[:-1], dtype=np.float64)

    return coords


__all__ = [
    "ALLOWANCE_BY_METHOD",
    "compute_seam_allowance_offset",
]
