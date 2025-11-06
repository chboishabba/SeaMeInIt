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
