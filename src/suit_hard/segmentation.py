"""Motion-aware hard shell segmentation around articulated joints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "ArticulationDefinition",
    "HardShellSegmentation",
    "HardShellSegmentationOptions",
    "HardShellSegmenter",
    "SegmentedPanel",
]


Vector = np.ndarray


@dataclass(slots=True, frozen=True)
class ArticulationDefinition:
    """Description of an articulation driving a rigid panel cut line."""

    name: str
    proximal_joint: str
    distal_joint: str
    panel_name: str
    cut_ratio: float = 0.35
    up_hint: Vector = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=float))


@dataclass(slots=True, frozen=True)
class HardShellSegmentationOptions:
    """Runtime knobs influencing hard-shell segmentation output."""

    hinge_allowance: float = 0.004
    panel_width_scale: float = 0.3
    panel_height_scale: float = 0.22
    hinge_extension_scale: float = 0.08
    boundary_points: int = 24


@dataclass(slots=True, frozen=True)
class SegmentedPanel:
    """Panel generated around a single articulation."""

    name: str
    joint_name: str
    cut_point: Vector
    cut_normal: Vector
    hinge_line: np.ndarray
    boundary: np.ndarray
    allowance: float
    motion_axis: Vector
    limb_length: float

    def as_dict(self) -> dict[str, object]:
        """Serialise the panel to a JSON-friendly mapping."""

        return {
            "name": self.name,
            "joint_name": self.joint_name,
            "cut_point": self.cut_point.tolist(),
            "cut_normal": self.cut_normal.tolist(),
            "hinge_line": self.hinge_line.tolist(),
            "boundary": self.boundary.tolist(),
            "allowance": self.allowance,
            "motion_axis": self.motion_axis.tolist(),
            "limb_length": self.limb_length,
        }


@dataclass(slots=True, frozen=True)
class HardShellSegmentation:
    """Segmentation output containing all computed panels."""

    panels: tuple[SegmentedPanel, ...]

    def panels_for_joint(self, joint_name: str) -> tuple[SegmentedPanel, ...]:
        """Return every panel belonging to ``joint_name``."""

        return tuple(panel for panel in self.panels if panel.joint_name == joint_name)

    def as_dict(self) -> dict[str, object]:
        """Serialise the segmentation for persistence."""

        return {"panels": [panel.as_dict() for panel in self.panels]}


class HardShellSegmenter:
    """Compute motion-aware segmentation curves from avatar joints."""

    #: Default articulation set targeting major limb joints on SMPL-X.
    DEFAULT_ARTICULATIONS: tuple[ArticulationDefinition, ...] = (
        ArticulationDefinition("left_shoulder", "left_shoulder", "left_elbow", "left_shoulder_panel", 0.32),
        ArticulationDefinition("left_elbow", "left_elbow", "left_wrist", "left_elbow_panel", 0.46),
        ArticulationDefinition("left_hip", "left_hip", "left_knee", "left_hip_panel", 0.4),
        ArticulationDefinition("left_knee", "left_knee", "left_ankle", "left_knee_panel", 0.47),
        ArticulationDefinition("right_shoulder", "right_shoulder", "right_elbow", "right_shoulder_panel", 0.32),
        ArticulationDefinition("right_elbow", "right_elbow", "right_wrist", "right_elbow_panel", 0.46),
        ArticulationDefinition("right_hip", "right_hip", "right_knee", "right_hip_panel", 0.4),
        ArticulationDefinition("right_knee", "right_knee", "right_ankle", "right_knee_panel", 0.47),
    )

    #: Common fallbacks for unnamed joints when records omit ``joint_names``.
    DEFAULT_JOINT_ORDER: tuple[str, ...] = (
        "pelvis",
        "left_hip",
        "right_hip",
        "spine",
        "left_knee",
        "right_knee",
        "spine1",
        "left_ankle",
        "right_ankle",
        "spine2",
        "left_foot",
        "right_foot",
        "neck",
        "left_shoulder",
        "right_shoulder",
        "head",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    )

    def __init__(
        self,
        articulations: Sequence[ArticulationDefinition] | None = None,
    ) -> None:
        self._articulations = tuple(articulations) if articulations is not None else self.DEFAULT_ARTICULATIONS

    def segment(
        self,
        joints: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | np.ndarray,
        *,
        options: HardShellSegmentationOptions | None = None,
        joint_names: Sequence[str] | None = None,
    ) -> HardShellSegmentation:
        """Compute panels from a set of joints."""

        opts = options or HardShellSegmentationOptions()
        joint_positions = self._normalise_joint_map(joints, joint_names=joint_names)
        panels = [
            panel
            for definition in self._articulations
            if (panel := self._build_panel(definition, joint_positions, opts)) is not None
        ]
        return HardShellSegmentation(tuple(panels))

    def _normalise_joint_map(
        self,
        joints: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | np.ndarray,
        *,
        joint_names: Sequence[str] | None,
    ) -> Mapping[str, np.ndarray]:
        if isinstance(joints, Mapping):
            return {name: np.asarray(value, dtype=float) for name, value in joints.items()}

        joint_array = np.asarray(joints, dtype=float)
        if joint_array.ndim != 2 or joint_array.shape[1] != 3:
            raise ValueError("Joint arrays must have shape (N, 3).")
        if joint_names is None:
            if joint_array.shape[0] > len(self.DEFAULT_JOINT_ORDER):
                msg = "joint_names must be supplied when more joints than defaults are provided"
                raise ValueError(msg)
            names = self.DEFAULT_JOINT_ORDER[: joint_array.shape[0]]
        else:
            names = tuple(joint_names)
        return {name: joint_array[idx] for idx, name in enumerate(names)}

    def _build_panel(
        self,
        definition: ArticulationDefinition,
        joint_positions: Mapping[str, np.ndarray],
        options: HardShellSegmentationOptions,
    ) -> SegmentedPanel | None:
        try:
            proximal = joint_positions[definition.proximal_joint]
            distal = joint_positions[definition.distal_joint]
        except KeyError:
            return None

        limb_vector = distal - proximal
        limb_length = float(np.linalg.norm(limb_vector))
        if limb_length < 1e-6:
            return None
        axis = limb_vector / limb_length

        cut_point = proximal + limb_vector * np.clip(definition.cut_ratio, 0.05, 0.95)

        up_hint = definition.up_hint
        up_hint = up_hint / np.linalg.norm(up_hint)
        if abs(np.dot(up_hint, axis)) > 0.92:
            fallback = np.array([1.0, 0.0, 0.0], dtype=float)
            up_hint = fallback / np.linalg.norm(fallback)

        tangent = np.cross(up_hint, axis)
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            fallback = np.array([0.0, 0.0, 1.0], dtype=float)
            tangent = np.cross(fallback, axis)
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-6:
                return None
        tangent /= tangent_norm

        binormal = np.cross(axis, tangent)
        binormal /= np.linalg.norm(binormal)

        width = options.panel_width_scale * limb_length
        height = options.panel_height_scale * limb_length
        radius_major = max(width, 1e-4)
        radius_minor = max(height, 1e-4)

        boundary = self._ellipse(cut_point, tangent, binormal, radius_major, radius_minor, options.boundary_points)
        hinge_extension = options.hinge_extension_scale * limb_length
        hinge_line = np.stack(
            (
                cut_point - tangent * (radius_major + hinge_extension),
                cut_point + tangent * (radius_major + hinge_extension),
            )
        )

        return SegmentedPanel(
            name=definition.panel_name,
            joint_name=definition.name,
            cut_point=cut_point,
            cut_normal=axis,
            hinge_line=hinge_line,
            boundary=boundary,
            allowance=options.hinge_allowance,
            motion_axis=axis,
            limb_length=limb_length,
        )

    @staticmethod
    def _ellipse(
        centre: Vector,
        tangent: Vector,
        binormal: Vector,
        radius_major: float,
        radius_minor: float,
        samples: int,
    ) -> np.ndarray:
        steps = max(int(samples), 8)
        angles = np.linspace(0.0, 2 * np.pi, num=steps, endpoint=False)
        points = [
            centre
            + np.cos(theta) * radius_major * tangent
            + np.sin(theta) * radius_minor * binormal
            for theta in angles
        ]
        points.append(points[0])
        return np.asarray(points, dtype=float)
