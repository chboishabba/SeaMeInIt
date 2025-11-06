"""Hard-shell clearance analysis utilities."""

from .clearance import (
    ClearanceResult,
    ContactPoint,
    Mesh,
    PoseClearance,
    analyze_clearance,
    interpolate_poses,
)

__all__ = [
    "analyze_clearance",
    "ClearanceResult",
    "ContactPoint",
    "Mesh",
    "PoseClearance",
    "interpolate_poses",
]
