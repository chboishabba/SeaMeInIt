"""Hard-shell attachments, clearance, segmentation, and shell generation."""

from .attachments import (
    Attachment,
    AttachmentLayout,
    AttachmentPlanner,
    AttachmentRouting,
    PanelSegment,
)
from .clearance import (
    ClearanceResult,
    ContactPoint,
    Mesh,
    PoseClearance,
    analyze_clearance,
    interpolate_poses,
)
from .segmentation import (
    ArticulationDefinition,
    HardShellSegmentation,
    HardShellSegmentationOptions,
    HardShellSegmenter,
    SegmentedPanel,
)
from .shell_generator import ShellGenerationResult, ShellGenerator, ShellOptions

__all__ = [
    "Attachment",
    "AttachmentLayout",
    "AttachmentPlanner",
    "AttachmentRouting",
    "PanelSegment",
    "ClearanceResult",
    "ContactPoint",
    "Mesh",
    "PoseClearance",
    "analyze_clearance",
    "interpolate_poses",
    "ArticulationDefinition",
    "HardShellSegmentation",
    "HardShellSegmentationOptions",
    "HardShellSegmenter",
    "SegmentedPanel",
    "ShellGenerationResult",
    "ShellGenerator",
    "ShellOptions",
]
