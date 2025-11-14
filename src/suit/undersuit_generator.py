"""Generate watertight undersuit layers from body model outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from .body_axes import fit_body_axes
from .measurement_loops import solve_measurement_loops

LayerArray = np.ndarray

__all__ = [
    "LAYER_REFERENCE_MEASUREMENTS",
    "MeshLayer",
    "UnderSuitOptions",
    "UnderSuitResult",
    "UnderSuitGenerator",
]


LAYER_REFERENCE_MEASUREMENTS: Mapping[str, float] = {
    "chest_circumference": 95.0,
    "waist_circumference": 80.0,
    "hip_circumference": 98.0,
    "neck_circumference": 36.0,
    "thigh_circumference": 55.0,
}
"""Reference anthropometric values used to compute scaling adjustments."""


@dataclass(frozen=True, slots=True)
class MeshLayer:
    """Container describing a generated undersuit layer."""

    name: str
    vertices: LayerArray
    faces: LayerArray
    thickness: float
    surface_area: float


@dataclass(frozen=True, slots=True)
class UnderSuitOptions:
    """Configuration flags that control undersuit layer synthesis."""

    base_thickness: float = 0.0015
    insulation_thickness: float = 0.003
    comfort_liner_thickness: float = 0.001
    include_insulation: bool = True
    include_comfort_liner: bool = True
    ease_percent: float = 0.03
    measurement_weights: Mapping[str, float] | None = None

    def layer_sequence(self) -> Iterable[tuple[str, float]]:
        yield ("base", self.base_thickness)
        if self.include_insulation and self.insulation_thickness > 0.0:
            yield ("insulation", self.insulation_thickness)
        if self.include_comfort_liner and self.comfort_liner_thickness > 0.0:
            yield ("comfort", self.comfort_liner_thickness)


@dataclass(frozen=True, slots=True)
class UnderSuitResult:
    """Result payload returned by :class:`UnderSuitGenerator`."""

    base_layer: MeshLayer
    insulation_layer: MeshLayer | None
    comfort_layer: MeshLayer | None
    metadata: Mapping[str, object]

    def layers(self) -> tuple[MeshLayer, ...]:
        entries: list[MeshLayer] = [self.base_layer]
        if self.insulation_layer is not None:
            entries.append(self.insulation_layer)
        if self.comfort_layer is not None:
            entries.append(self.comfort_layer)
        return tuple(entries)


class UnderSuitGenerator:
    """Generate watertight undersuit meshes based on SMPL-X outputs."""

    def __init__(self, seam_tolerance: float = 1e-5) -> None:
        self.seam_tolerance = float(seam_tolerance)

    def generate(
        self,
        body_output: Mapping[str, LayerArray] | object,
        *,
        options: UnderSuitOptions | None = None,
        measurements: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
    ) -> UnderSuitResult:
        """Produce undersuit layers given a SMPL-X body model output."""

        options = options or UnderSuitOptions()
        measurement_values, measurement_confidences = _parse_measurements(measurements)
        vertices, faces = _extract_vertices_and_faces(body_output)
        normals = _vertex_normals(vertices, faces)
        measurement_scale = _measurement_scale(measurement_values, options.measurement_weights)
        ease_scale = 1.0 + float(options.ease_percent)
        scale_factor = measurement_scale * ease_scale

        centroid = vertices.mean(axis=0, keepdims=True)
        scaled_vertices = _scale_points(vertices, centroid, scale_factor)

        raw_joints = None
        if isinstance(body_output, Mapping):
            raw_joints = body_output.get("joints")
        else:
            raw_joints = getattr(body_output, "joints", None)

        body_axes = fit_body_axes(
            scaled_vertices,
            joints=_scale_joint_payload(raw_joints, centroid, scale_factor),
            measurements=measurement_values,
            measurement_confidences=measurement_confidences,
        )

        layer_metadata: MutableMapping[str, object] = {
            "scale_factor": scale_factor,
            "measurement_scale": measurement_scale,
            "ease_percent": options.ease_percent,
            "watertight": _is_watertight(faces, tolerance=self.seam_tolerance),
            "body_axes": body_axes.to_metadata(),
        }

        if not layer_metadata["watertight"]:
            raise ValueError("Body mesh must be watertight to generate undersuit layers.")

        base_layer: MeshLayer | None = None
        insulation_layer: MeshLayer | None = None
        comfort_layer: MeshLayer | None = None
        previous_vertices = scaled_vertices

        for name, thickness in options.layer_sequence():
            layer_vertices = previous_vertices + normals * thickness
            area = _surface_area(layer_vertices, faces)
            layer = MeshLayer(
                name=name,
                vertices=layer_vertices,
                faces=faces.copy(),
                thickness=thickness,
                surface_area=area,
            )
            if name == "base":
                base_layer = layer
            elif name == "insulation":
                insulation_layer = layer
            elif name == "comfort":
                comfort_layer = layer
            previous_vertices = layer_vertices

        assert base_layer is not None, "Base layer must always be generated."

        layer_metadata["layers"] = {
            layer.name: {
                "thickness": layer.thickness,
                "surface_area": layer.surface_area,
            }
            for layer in (layer for layer in (base_layer, insulation_layer, comfort_layer) if layer is not None)
        }
        layer_metadata["seam_max_deviation"] = _seam_deviation(base_layer.vertices, faces)

        if measurements:
            measurement_loops = solve_measurement_loops(
                base_layer.vertices,
                base_layer.faces,
                measurements,
            )
            if measurement_loops:
                layer_metadata["measurement_loops"] = {
                    name: {
                        "target": loop.target,
                        "perimeter": loop.perimeter,
                        "relative_error": loop.relative_error,
                        "vertices": list(loop.vertices),
                    }
                    for name, loop in measurement_loops.items()
                }

        return UnderSuitResult(
            base_layer=base_layer,
            insulation_layer=insulation_layer,
            comfort_layer=comfort_layer,
            metadata=dict(layer_metadata),
        )


def _extract_vertices_and_faces(body_output: Mapping[str, LayerArray] | object) -> tuple[LayerArray, LayerArray]:
    if isinstance(body_output, Mapping):
        vertices = body_output.get("vertices")
        faces = body_output.get("faces")
    else:
        vertices = getattr(body_output, "vertices", None)
        faces = getattr(body_output, "faces", None)

    if vertices is None or faces is None:
        raise TypeError("Body output must provide 'vertices' and 'faces'.")

    vertices_arr = np.asarray(vertices, dtype=float)
    faces_arr = np.asarray(faces, dtype=int)

    if vertices_arr.ndim == 3:
        vertices_arr = vertices_arr[0]
    if faces_arr.ndim == 3:
        faces_arr = faces_arr[0]

    if vertices_arr.ndim != 2 or vertices_arr.shape[1] != 3:
        raise ValueError("Vertices must be shaped (N, 3).")
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
        raise ValueError("Faces must be shaped (M, 3).")

    return vertices_arr, faces_arr


def _scale_about_centroid(vertices: LayerArray, scale_factor: float) -> LayerArray:
    center = vertices.mean(axis=0, keepdims=True)
    return _scale_points(vertices, center, scale_factor)


def _scale_points(points: LayerArray, center: np.ndarray, scale_factor: float) -> LayerArray:
    return (points - center) * scale_factor + center


def _measurement_scale(
    measurements: Mapping[str, float] | None,
    weights: Mapping[str, float] | None,
) -> float:
    if not measurements:
        return 1.0
    weighted: list[float] = []
    total_weight = 0.0
    weighting = weights or {}
    for name, reference in LAYER_REFERENCE_MEASUREMENTS.items():
        if name in measurements:
            weight = weighting.get(name, 1.0)
            weighted.append((float(measurements[name]) / reference) * weight)
            total_weight += weight
    if not weighted:
        return 1.0
    return float(np.sum(weighted) / total_weight)


def _parse_measurements(
    measurements: Mapping[str, object] | Sequence[Mapping[str, object]] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    if measurements is None:
        return {}, {}

    values: dict[str, float] = {}
    confidences: dict[str, float] = {}

    if isinstance(measurements, Mapping):
        items = measurements.items()
    else:
        items = []
        for entry in measurements:
            if isinstance(entry, Mapping) and "name" in entry:
                items.append((entry["name"], entry))

    for name, raw_value in items:
        if isinstance(raw_value, Mapping):
            if "value" in raw_value:
                values[name] = float(raw_value["value"])
            if "confidence" in raw_value:
                confidences[name] = float(raw_value["confidence"])
        else:
            values[name] = float(raw_value)

    return values, confidences


def _scale_joint_payload(
    joints: Mapping[str, object] | Sequence[Mapping[str, object]] | None,
    center: np.ndarray,
    scale_factor: float,
) -> Mapping[str, object] | Sequence[Mapping[str, object]] | None:
    if joints is None:
        return None

    center_vec = np.asarray(center, dtype=float).reshape(1, 3)

    if isinstance(joints, Mapping):
        scaled: dict[str, np.ndarray] = {}
        for name, value in joints.items():
            arr = np.asarray(value, dtype=float)
            if arr.shape == (3,):
                scaled[name] = _scale_points(arr.reshape(1, 3), center_vec, scale_factor)[0]
        return scaled

    scaled_entries: list[Mapping[str, object]] = []
    for entry in joints:
        if isinstance(entry, Mapping) and "position" in entry:
            arr = np.asarray(entry["position"], dtype=float)
            if arr.shape == (3,):
                scaled_entry = dict(entry)
                scaled_entry["position"] = _scale_points(arr.reshape(1, 3), center_vec, scale_factor)[0]
                scaled_entries.append(scaled_entry)
                continue
        scaled_entries.append(entry)
    return scaled_entries


def _vertex_normals(vertices: LayerArray, faces: LayerArray) -> LayerArray:
    normals = np.zeros_like(vertices)
    tri_vertices = vertices[faces]
    tri_normals = np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0])
    for idx, face in enumerate(faces):
        normals[face] += tri_normals[idx]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return normals / norms


def _surface_area(vertices: LayerArray, faces: LayerArray) -> float:
    tri_vertices = vertices[faces]
    cross = np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def _is_watertight(faces: LayerArray, *, tolerance: float) -> bool:
    edge_counts: dict[tuple[int, int], int] = {}
    for tri in faces:
        for idx in range(3):
            edge = (int(tri[idx]), int(tri[(idx + 1) % 3]))
            key = tuple(sorted(edge))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    return all(abs(count - 2) <= tolerance for count in edge_counts.values())


def _seam_deviation(vertices: LayerArray, faces: LayerArray) -> float:
    edge_map: dict[tuple[int, int], list[float]] = {}
    for tri in faces:
        for idx in range(3):
            a = int(tri[idx])
            b = int(tri[(idx + 1) % 3])
            key = tuple(sorted((a, b)))
            length = float(np.linalg.norm(vertices[a] - vertices[b]))
            edge_map.setdefault(key, []).append(length)
    deviations = [abs(values[0] - values[1]) for values in edge_map.values() if len(values) == 2]
    if not deviations:
        return 0.0
    return float(max(deviations))
