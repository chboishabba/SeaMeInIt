"""Utilities for fitting anatomical centreline axes on body meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = ["AxisPath", "BodyAxes", "fit_body_axes"]


ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True, slots=True)
class AxisPath:
    """Represents a single centreline axis for a limb or spine."""

    name: str
    points: np.ndarray
    parameters: np.ndarray
    length: float
    measurement_name: str | None
    confidence: float

    def sample(self, values: float | Sequence[float] | np.ndarray) -> np.ndarray:
        """Interpolate points along the axis for the provided parameter values."""

        values_arr = np.asarray(values, dtype=float)
        single_value = values_arr.ndim == 0
        if single_value:
            values_arr = values_arr[None]

        if not len(self.points):
            result = np.zeros((values_arr.size, 3), dtype=float)
        elif len(self.points) == 1:
            result = np.repeat(self.points, values_arr.size, axis=0)
        else:
            values_arr = np.clip(values_arr, 0.0, 1.0)
            unique_mask = _unique_parameter_mask(self.parameters)
            base_parameters = self.parameters[unique_mask]
            base_points = self.points[unique_mask]
            result = np.stack(
                [np.interp(values_arr, base_parameters, base_points[:, axis]) for axis in range(3)],
                axis=-1,
            )

        if single_value:
            return result[0]
        return result

    def to_metadata(self) -> Mapping[str, object]:
        """Return a JSON-serialisable representation of the axis."""

        return {
            "name": self.name,
            "length": float(self.length),
            "measurement": self.measurement_name,
            "confidence": float(self.confidence),
            "points": self.points.tolist(),
            "parameters": self.parameters.tolist(),
        }


@dataclass(frozen=True, slots=True)
class BodyAxes:
    """Collection of all fitted body axes for a mesh."""

    axes: Mapping[str, AxisPath]

    def __iter__(self) -> Iterable[AxisPath]:
        return iter(self.axes.values())

    def axis(self, name: str) -> AxisPath:
        return self.axes[name]

    def to_metadata(self) -> Mapping[str, Mapping[str, object]]:
        return {name: axis.to_metadata() for name, axis in self.axes.items()}


@dataclass(frozen=True, slots=True)
class _AxisSpec:
    name: str
    landmarks: tuple[str, ...]
    measurement: str | None
    fallback_axis: int | None
    side: str | None = None


_AXIS_SPECS: tuple[_AxisSpec, ...] = (
    _AxisSpec(
        name="left_arm",
        landmarks=("left_shoulder", "left_elbow", "left_wrist", "left_hand"),
        measurement="arm_length",
        fallback_axis=0,
        side="negative",
    ),
    _AxisSpec(
        name="right_arm",
        landmarks=("right_shoulder", "right_elbow", "right_wrist", "right_hand"),
        measurement="arm_length",
        fallback_axis=0,
        side="positive",
    ),
    _AxisSpec(
        name="left_leg",
        landmarks=("left_hip", "left_knee", "left_ankle", "left_foot"),
        measurement="inseam_length",
        fallback_axis=2,
        side="negative",
    ),
    _AxisSpec(
        name="right_leg",
        landmarks=("right_hip", "right_knee", "right_ankle", "right_foot"),
        measurement="inseam_length",
        fallback_axis=2,
        side="positive",
    ),
    _AxisSpec(
        name="spine",
        landmarks=("pelvis", "spine", "spine1", "spine2", "neck", "head"),
        measurement="height",
        fallback_axis=2,
        side=None,
    ),
)


def fit_body_axes(
    vertices: ArrayLike | np.ndarray,
    *,
    joints: Mapping[str, ArrayLike | np.ndarray] | Sequence[Mapping[str, ArrayLike | np.ndarray]] | None = None,
    measurements: Mapping[str, float] | None = None,
    measurement_confidences: Mapping[str, float] | None = None,
) -> BodyAxes:
    """Fit centreline axes for key limbs and torso regions."""

    vertices_arr = _ensure_vertices(vertices)
    joint_lookup = _normalise_joints(joints)
    measurements = measurements or {}
    confidences = measurement_confidences or {}

    axes: dict[str, AxisPath] = {}
    for spec in _AXIS_SPECS:
        points = _points_from_landmarks(spec.landmarks, joint_lookup)
        if points is None:
            points = _fallback_points(vertices_arr, spec)

        path = _build_axis_path(
            spec,
            points,
            measurements.get(spec.measurement) if spec.measurement else None,
            confidences.get(spec.measurement, 1.0) if spec.measurement else 0.0,
        )
        axes[spec.name] = path

    return BodyAxes(axes)


def _ensure_vertices(vertices: ArrayLike | np.ndarray) -> np.ndarray:
    vertices_arr = np.asarray(vertices, dtype=float)
    if vertices_arr.ndim != 2 or vertices_arr.shape[1] != 3:
        raise ValueError("Vertices must be an array of shape (N, 3).")
    return vertices_arr


def _normalise_joints(
    joints: Mapping[str, ArrayLike | np.ndarray]
    | Sequence[Mapping[str, ArrayLike | np.ndarray]]
    | None,
) -> Mapping[str, np.ndarray]:
    if joints is None:
        return {}

    if isinstance(joints, Mapping):
        iter_items = joints.items()
    else:
        lookup: dict[str, ArrayLike | np.ndarray] = {}
        for entry in joints:
            if isinstance(entry, Mapping) and "name" in entry and "position" in entry:
                lookup[str(entry["name"])] = entry["position"]
        iter_items = lookup.items()

    normalised: dict[str, np.ndarray] = {}
    for name, value in iter_items:
        arr = np.asarray(value, dtype=float)
        if arr.shape != (3,):
            continue
        normalised[str(name)] = arr
    return normalised


def _points_from_landmarks(
    names: Sequence[str],
    joints: Mapping[str, np.ndarray],
) -> np.ndarray | None:
    points: list[np.ndarray] = []
    for name in names:
        if name in joints:
            points.append(joints[name])
    if len(points) < 2:
        return None
    stacked = np.vstack(points)
    return _deduplicate_points(stacked)


def _fallback_points(vertices: np.ndarray, spec: _AxisSpec) -> np.ndarray:
    center = vertices.mean(axis=0)
    if spec.fallback_axis is None:
        axis_idx = int(np.argmax(np.var(vertices, axis=0)))
    else:
        axis_idx = spec.fallback_axis

    coords = vertices[:, axis_idx]
    subset = vertices
    if spec.side == "negative":
        subset_mask = coords <= np.median(coords)
        if np.any(subset_mask):
            subset = vertices[subset_mask]
    elif spec.side == "positive":
        subset_mask = coords >= np.median(coords)
        if np.any(subset_mask):
            subset = vertices[subset_mask]

    if subset.size == 0:
        subset = vertices

    start = center
    if spec.side == "negative":
        end = subset[np.argmin(subset[:, axis_idx])]
    elif spec.side == "positive":
        end = subset[np.argmax(subset[:, axis_idx])]
    else:
        deviations = np.abs(subset[:, axis_idx] - center[axis_idx])
        end = subset[np.argmax(deviations)]

    points = np.vstack([start, end])
    return _deduplicate_points(points)


def _build_axis_path(
    spec: _AxisSpec,
    points: np.ndarray,
    measurement: float | None,
    confidence: float,
) -> AxisPath:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Axis points must be shaped (N, 3).")
    if len(points) == 0:
        points = np.zeros((1, 3), dtype=float)

    points = _deduplicate_points(points)
    if len(points) == 1:
        parameters = np.zeros(1, dtype=float)
        length = 0.0
        return AxisPath(spec.name, points, parameters, length, spec.measurement, float(confidence))

    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    geom_length = float(segment_lengths.sum())

    target_length = _blend_length(geom_length, measurement, confidence)
    scaled_points = _scale_polyline(points, geom_length, target_length)

    scaled_diffs = np.diff(scaled_points, axis=0)
    scaled_lengths = np.linalg.norm(scaled_diffs, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(scaled_lengths)))
    total = float(cumulative[-1])
    parameters = cumulative / total if total > 0.0 else np.zeros_like(cumulative)

    return AxisPath(
        spec.name,
        scaled_points,
        parameters,
        float(total),
        spec.measurement,
        float(np.clip(confidence, 0.0, 1.0) if spec.measurement else 0.0),
    )


def _deduplicate_points(points: np.ndarray, *, atol: float = 1e-9) -> np.ndarray:
    if len(points) <= 1:
        return points
    filtered = [points[0]]
    for point in points[1:]:
        if not np.allclose(point, filtered[-1], atol=atol):
            filtered.append(point)
    return np.vstack(filtered)


def _blend_length(geom_length: float, measurement: float | None, confidence: float) -> float:
    if measurement is None or geom_length == 0.0:
        return measurement if measurement is not None else geom_length
    weight = float(np.clip(confidence, 0.0, 1.0))
    return geom_length + weight * (measurement - geom_length)


def _scale_polyline(points: np.ndarray, geom_length: float, target_length: float | None) -> np.ndarray:
    if target_length is None or geom_length == 0.0:
        return points
    if target_length <= 0.0:
        return np.repeat(points[:1], len(points), axis=0)

    scale = target_length / geom_length
    scaled = [points[0]]
    for delta in np.diff(points, axis=0):
        scaled.append(scaled[-1] + delta * scale)
    return np.vstack(scaled)


def _unique_parameter_mask(parameters: np.ndarray, *, atol: float = 1e-9) -> np.ndarray:
    if parameters.size <= 1:
        return np.ones_like(parameters, dtype=bool)
    mask = np.ones_like(parameters, dtype=bool)
    prev = parameters[0]
    for idx in range(1, parameters.size):
        if abs(parameters[idx] - prev) <= atol:
            mask[idx] = False
        else:
            prev = parameters[idx]
    return mask
