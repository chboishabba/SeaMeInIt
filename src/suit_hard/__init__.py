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
"""Hard-shell suit segmentation utilities."""

from .segmentation import (
    ArticulationDefinition,
    HardShellSegmentation,
    HardShellSegmentationOptions,
    HardShellSegmenter,
    SegmentedPanel,
)

__all__ = [
    "ArticulationDefinition",
    "HardShellSegmentation",
    "HardShellSegmentationOptions",
    "HardShellSegmenter",
    "SegmentedPanel",
]
"""Rigid shell generation utilities."""

from .shell_generator import ShellGenerationResult, ShellGenerator, ShellOptions

__all__ = ["ShellGenerationResult", "ShellGenerator", "ShellOptions"]
