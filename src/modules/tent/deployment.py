"""Tent deployment planning logic leveraging suit landmarks and seam templates."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

__all__ = [
    "AttachmentAnchor",
    "FoldPath",
    "FoldStep",
    "FoldingSequence",
    "DeploymentKinematics",
    "default_attachment_anchors",
    "generate_fold_paths",
    "build_deployment_kinematics",
    "load_canopy_template",
    "load_canopy_seams",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ASSET_ROOT = _REPO_ROOT / "assets" / "modules" / "tent"


@dataclass(slots=True)
class AttachmentAnchor:
    """Anchor tying the canopy to a particular suit landmark."""

    name: str
    landmark: str
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    description: str

    @classmethod
    def from_landmark(
        cls,
        *,
        name: str,
        landmark: str,
        landmark_positions: Mapping[str, Sequence[float]],
        offset: Sequence[float],
        description: str,
    ) -> "AttachmentAnchor":
        """Create an anchor offset from a suit landmark."""

        if landmark not in landmark_positions:
            msg = f"Landmark '{landmark}' required for anchor '{name}' was not provided."
            raise KeyError(msg)

        base = tuple(float(value) for value in landmark_positions[landmark][:3])
        if len(base) != 3:
            msg = f"Landmark '{landmark}' must provide three coordinates"
            raise ValueError(msg)

        offset_vec = tuple(float(value) for value in offset[:3])
        position = tuple(component + delta for component, delta in zip(base, offset_vec))
        magnitude = math.sqrt(sum(component**2 for component in offset_vec))
        if magnitude == 0.0:
            normal = (0.0, 1.0, 0.0)
        else:
            normal = tuple(component / magnitude for component in offset_vec)
        return cls(
            name=name,
            landmark=landmark,
            position=position,
            normal=normal,
            description=description,
        )


@dataclass(slots=True)
class FoldPath:
    """Geometric description of a fold path between two anchors."""

    name: str
    anchors: tuple[str, str]
    points: tuple[tuple[float, float, float], ...]
    description: str
    order: int

    def validate(self, anchors: Mapping[str, AttachmentAnchor]) -> None:
        """Validate path anchors exist and the path has at least two points."""

        start, end = self.anchors
        if start not in anchors:
            msg = f"Fold path '{self.name}' references missing anchor '{start}'"
            raise ValueError(msg)
        if end not in anchors:
            msg = f"Fold path '{self.name}' references missing anchor '{end}'"
            raise ValueError(msg)
        if len(self.points) < 2:
            msg = f"Fold path '{self.name}' must contain at least two points"
            raise ValueError(msg)


@dataclass(slots=True)
class FoldStep:
    """Single instruction referencing a fold path."""

    name: str
    path_name: str
    instruction: str
    dwell_time: float | None = None


@dataclass(slots=True)
class FoldingSequence:
    """Ordered set of fold steps describing canopy deployment."""

    name: str
    steps: tuple[FoldStep, ...]

    def path_names(self) -> tuple[str, ...]:
        return tuple(step.path_name for step in self.steps)


@dataclass(slots=True)
class DeploymentKinematics:
    """Complete deployment description referencing anchors and fold paths."""

    anchors: Mapping[str, AttachmentAnchor]
    fold_paths: Mapping[str, FoldPath]
    sequence: FoldingSequence

    def validate(self) -> None:
        for anchor in self.anchors.values():
            if len(anchor.position) != 3:
                msg = f"Anchor '{anchor.name}' must define a three-dimensional position"
                raise ValueError(msg)
        for path in self.fold_paths.values():
            path.validate(self.anchors)
        for step in self.sequence.steps:
            if step.path_name not in self.fold_paths:
                msg = f"Fold step '{step.name}' references unknown path '{step.path_name}'"
                raise ValueError(msg)


def default_attachment_anchors(
    landmarks: Mapping[str, Sequence[float]],
) -> Mapping[str, AttachmentAnchor]:
    """Generate default canopy anchors using common suit landmarks."""

    plan = [
        {
            "name": "dorsal_mount",
            "landmark": "c7_vertebra",
            "offset": (0.0, 0.08, -0.04),
            "description": "Primary dorsal attachment centred near C7 for load transfer.",
        },
        {
            "name": "ventral_mount",
            "landmark": "sternum",
            "offset": (0.0, 0.06, 0.05),
            "description": "Ventral harness anchor tying the canopy to the sternum plate.",
        },
        {
            "name": "left_shoulder_mount",
            "landmark": "left_acromion",
            "offset": (-0.09, 0.02, -0.02),
            "description": "Left shoulder anchor to stabilise the canopy ridge.",
        },
        {
            "name": "right_shoulder_mount",
            "landmark": "right_acromion",
            "offset": (0.09, 0.02, -0.02),
            "description": "Right shoulder anchor to stabilise the canopy ridge.",
        },
    ]

    anchors: MutableMapping[str, AttachmentAnchor] = {}
    for entry in plan:
        anchor = AttachmentAnchor.from_landmark(
            name=entry["name"],
            landmark=entry["landmark"],
            landmark_positions=landmarks,
            offset=entry["offset"],
            description=entry["description"],
        )
        anchors[anchor.name] = anchor
    return anchors


def generate_fold_paths(
    anchors: Mapping[str, AttachmentAnchor],
    seam_plan: Mapping[str, Mapping[str, object]],
) -> Mapping[str, FoldPath]:
    """Create fold paths by combining seam metadata with anchor definitions."""

    fold_paths: MutableMapping[str, FoldPath] = {}
    for seam in seam_plan.values():
        fold_payloads = seam.get("fold_paths", [])
        if fold_payloads is None:
            continue
        if not isinstance(fold_payloads, Iterable):
            msg = "seam fold_paths entries must be iterable"
            raise TypeError(msg)
        for payload in fold_payloads:
            name = str(payload.get("name"))
            if not name:
                msg = "Fold path entries must define a name"
                raise ValueError(msg)
            if name in fold_paths:
                msg = f"Fold path '{name}' defined multiple times"
                raise ValueError(msg)
            anchor_pair = payload.get("anchors", [])
            if not isinstance(anchor_pair, Sequence) or len(anchor_pair) != 2:
                msg = f"Fold path '{name}' must define two anchors"
                raise ValueError(msg)
            anchor_tuple = (str(anchor_pair[0]), str(anchor_pair[1]))
            points_raw = payload.get("points", [])
            if not isinstance(points_raw, Iterable):
                msg = f"Fold path '{name}' points must be iterable"
                raise TypeError(msg)
            normalised_points: list[tuple[float, float, float]] = []
            for coord_tuple in points_raw:
                if not isinstance(coord_tuple, Sequence) or len(coord_tuple) < 3:
                    msg = f"Fold path '{name}' points must contain xyz coordinates"
                    raise ValueError(msg)
                normalised_points.append(
                    (
                        float(coord_tuple[0]),
                        float(coord_tuple[1]),
                        float(coord_tuple[2]),
                    )
                )
            points = tuple(normalised_points)
            description = str(payload.get("description", ""))
            order_raw = payload.get("order")
            if order_raw is None:
                order = len(fold_paths) + 1
            else:
                order = int(order_raw)
            fold_path = FoldPath(
                name=name,
                anchors=anchor_tuple,
                points=points,
                description=description,
                order=order,
            )
            fold_path.validate(anchors)
            fold_paths[name] = fold_path
    return fold_paths


def build_deployment_kinematics(
    landmarks: Mapping[str, Sequence[float]],
    seam_plan: Mapping[str, Mapping[str, object]],
) -> DeploymentKinematics:
    """Build deployment plan from suit landmarks and seam metadata."""

    anchors = default_attachment_anchors(landmarks)
    fold_paths = generate_fold_paths(anchors, seam_plan)
    ordered_paths = sorted(fold_paths.values(), key=lambda path: path.order)
    steps: list[FoldStep] = []
    for index, path in enumerate(ordered_paths, start=1):
        instruction = f"Step {index}: track along {path.name} between {path.anchors[0]} and {path.anchors[1]}."
        steps.append(
            FoldStep(
                name=f"fold_{index}",
                path_name=path.name,
                instruction=instruction,
                dwell_time=payload_dwell_time(seam_plan, path.name),
            )
        )
    sequence = FoldingSequence(name="tent_canopy_fold", steps=tuple(steps))
    deployment = DeploymentKinematics(anchors=anchors, fold_paths=fold_paths, sequence=sequence)
    deployment.validate()
    return deployment


def payload_dwell_time(
    seam_plan: Mapping[str, Mapping[str, object]], path_name: str
) -> float | None:
    """Look up an optional dwell time for a given fold path."""

    for seam in seam_plan.values():
        for payload in seam.get("fold_paths", []):
            if payload.get("name") == path_name:
                dwell = payload.get("dwell_time")
                if dwell is None:
                    return None
                return float(dwell)
    return None


def load_canopy_template() -> Mapping[str, object]:
    """Load the baseline canopy geometry used for deployment planning."""

    path = _ASSET_ROOT / "canopy_template.json"
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_canopy_seams() -> Mapping[str, Mapping[str, object]]:
    """Load seam metadata describing fold paths and allowances."""

    path = _ASSET_ROOT / "seams.json"
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return {str(key): value for key, value in payload.items()}
