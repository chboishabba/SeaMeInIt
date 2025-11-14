"""Seam graph generation for soft undersuit panelisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from .undersuit_generator import LAYER_REFERENCE_MEASUREMENTS

__all__ = [
    "MeasurementLoop",
    "SeamPanel",
    "SeamGraph",
    "SeamGenerator",
]


Vector = np.ndarray


@dataclass(frozen=True, slots=True)
class MeasurementLoop:
    """Closed vertex loop corresponding to a measurement plane."""

    name: str
    vertex_indices: tuple[int, ...]
    axis_coordinate: float


@dataclass(frozen=True, slots=True)
class SeamPanel:
    """Single disk-like panel extracted from the body mesh."""

    name: str
    anchor_loops: tuple[str, str]
    side: str
    vertices: np.ndarray = field(repr=False)
    faces: np.ndarray = field(repr=False)
    global_indices: tuple[int, ...]
    seam_vertices: tuple[int, ...]
    loop_vertex_indices: tuple[tuple[str, tuple[int, ...]], ...]
    metadata: Mapping[str, float]

    def loop_vertices(self) -> Mapping[str, tuple[int, ...]]:
        """Return a mapping of measurement loop name to contributing vertices."""

        return MappingProxyType(dict(self.loop_vertex_indices))


@dataclass(frozen=True, slots=True)
class SeamGraph:
    """Collection of panels and seam metadata generated for a suit."""

    panels: tuple[SeamPanel, ...]
    measurement_loops: tuple[MeasurementLoop, ...]
    seam_metadata: Mapping[str, Mapping[str, float]]

    def to_payload(self) -> dict[str, object]:
        """Serialise panels into the payload expected by :mod:`PatternExporter`."""

        return {
            "panels": [
                {
                    "name": panel.name,
                    "vertices": panel.vertices.tolist(),
                    "faces": panel.faces.tolist(),
                }
                for panel in self.panels
            ]
        }


class SeamGenerator:
    """Analyse a watertight mesh to compute minimal seam graphs."""

    #: Default ordering of measurement loops from proximal (top) to distal (bottom).
    DEFAULT_LOOP_SEQUENCE: tuple[str, ...] = (
        "neck_circumference",
        "chest_circumference",
        "waist_circumference",
        "hip_circumference",
        "thigh_circumference",
    )

    #: Nominalised axis fractions for default loop placement.
    DEFAULT_LOOP_POSITIONS: Mapping[str, float] = MappingProxyType(
        {
            "neck_circumference": 0.87,
            "chest_circumference": 0.74,
            "waist_circumference": 0.58,
            "hip_circumference": 0.44,
            "thigh_circumference": 0.32,
        }
    )

    def __init__(
        self,
        *,
        base_allowance: float = 0.012,
        seam_bias: float = 0.004,
        seam_tolerance: float = 5e-3,
    ) -> None:
        self.base_allowance = float(base_allowance)
        self.seam_bias = float(seam_bias)
        self.seam_tolerance = float(seam_tolerance)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def derive_measurement_loops(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        axis_map: Mapping[str, Vector],
        *,
        measurements: Mapping[str, float] | None = None,
    ) -> tuple[MeasurementLoop, ...]:
        """Generate canonical measurement loops for the provided mesh."""

        axis = _normalise_vector(axis_map.get("longitudinal", np.array([0.0, 0.0, 1.0])))
        lateral = _normalise_vector(axis_map.get("lateral", np.array([1.0, 0.0, 0.0])))
        anterior = _normalise_vector(axis_map.get("anterior", np.cross(axis, lateral)))
        if np.linalg.norm(anterior) < 1e-8:
            # Fall back to any orthonormal basis in the plane orthogonal to axis
            anterior = _orthogonal_basis(axis)
        lateral = _normalise_vector(np.cross(axis, anterior))
        anterior = _normalise_vector(np.cross(lateral, axis))

        coords = vertices @ axis
        min_coord = float(coords.min())
        max_coord = float(coords.max())
        extent = max(max_coord - min_coord, 1e-4)
        edge_length = _average_edge_length(vertices, faces)
        tolerance = max(extent * 0.015, edge_length * 0.5, 1e-4)

        loops: list[MeasurementLoop] = []
        for name in self.DEFAULT_LOOP_SEQUENCE:
            base_fraction = self.DEFAULT_LOOP_POSITIONS.get(name, 0.5)
            adjustment = 0.0
            if measurements and name in measurements:
                reference = LAYER_REFERENCE_MEASUREMENTS.get(name)
                if reference:
                    ratio = float(measurements[name]) / float(reference)
                    adjustment = (ratio - 1.0) * 0.05
            target_coordinate = min_coord + extent * float(np.clip(base_fraction + adjustment, 0.05, 0.95))
            loop_indices = _extract_loop_vertices(
                vertices,
                axis,
                target_coordinate,
                tolerance,
                lateral,
                anterior,
            )
            if not loop_indices:
                continue
            loop = MeasurementLoop(
                name=name,
                vertex_indices=tuple(loop_indices),
                axis_coordinate=target_coordinate,
            )
            loops.append(loop)

        loops.sort(key=lambda loop: loop.axis_coordinate)
        return tuple(loops)

    def generate(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        measurement_loops: Mapping[str, Sequence[int]]
        | Sequence[MeasurementLoop]
        | None,
        axis_map: Mapping[str, Vector],
        *,
        measurements: Mapping[str, float] | None = None,
    ) -> SeamGraph:
        """Compute a minimal seam graph from the provided mesh."""

        vertices_arr = np.asarray(vertices, dtype=float)
        faces_arr = np.asarray(faces, dtype=int)
        if vertices_arr.ndim != 2 or vertices_arr.shape[1] != 3:
            msg = "vertices must be shaped (N, 3)"
            raise ValueError(msg)
        if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
            msg = "faces must be shaped (M, 3)"
            raise ValueError(msg)

        loops = self._normalise_loops(vertices_arr, measurement_loops, axis_map)
        if not loops:
            loops = self.derive_measurement_loops(vertices_arr, faces_arr, axis_map, measurements=measurements)
        if len(loops) < 2:
            raise ValueError("At least two measurement loops are required to construct panels.")

        axis = _normalise_vector(axis_map.get("longitudinal", np.array([0.0, 0.0, 1.0])))
        lateral = _normalise_vector(axis_map.get("lateral", np.array([1.0, 0.0, 0.0])))
        anterior = _normalise_vector(axis_map.get("anterior", np.cross(axis, lateral)))
        if np.linalg.norm(anterior) < 1e-8:
            anterior = _orthogonal_basis(axis)
        lateral = _normalise_vector(np.cross(axis, anterior))
        anterior = _normalise_vector(np.cross(lateral, axis))

        curvature = _estimate_mean_curvature(vertices_arr, faces_arr)
        coords = vertices_arr @ axis
        lateral_coords = vertices_arr @ lateral

        panels: list[SeamPanel] = []
        seam_lookup: dict[str, Mapping[str, float]] = {}
        size_scale = _measurement_scale(measurements)

        for lower, upper in _pairwise(loops):
            lower_coord = min(lower.axis_coordinate, upper.axis_coordinate)
            upper_coord = max(lower.axis_coordinate, upper.axis_coordinate)
            region_mask = (coords >= (lower_coord - self.seam_tolerance)) & (
                coords <= (upper_coord + self.seam_tolerance)
            )
            if not np.any(region_mask):
                continue
            for side_sign, side_name in ((1.0, "right"), (-1.0, "left")):
                panel = self._build_panel(
                    vertices_arr,
                    faces_arr,
                    curvature,
                    loops,
                    lower,
                    upper,
                    region_mask,
                    lateral_coords,
                    side_sign,
                    side_name,
                    size_scale,
                )
                if panel is None:
                    continue
                panels.append(panel)
                seam_lookup[panel.name] = panel.metadata

        panels.sort(key=lambda panel: (panel.anchor_loops, panel.side))
        return SeamGraph(
            panels=tuple(panels),
            measurement_loops=loops,
            seam_metadata=MappingProxyType(dict(seam_lookup)),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalise_loops(
        self,
        vertices: np.ndarray,
        measurement_loops: Mapping[str, Sequence[int]]
        | Sequence[MeasurementLoop]
        | None,
        axis_map: Mapping[str, Vector],
    ) -> tuple[MeasurementLoop, ...]:
        if measurement_loops is None:
            return ()
        axis = _normalise_vector(axis_map.get("longitudinal", np.array([0.0, 0.0, 1.0])))
        loops: list[MeasurementLoop] = []
        if isinstance(measurement_loops, Mapping):
            for name, indices in measurement_loops.items():
                clean = _validate_indices(indices)
                if not clean:
                    continue
                coordinate = float((vertices[clean] @ axis).mean())
                loops.append(
                    MeasurementLoop(
                        name=str(name),
                        vertex_indices=tuple(clean),
                        axis_coordinate=coordinate,
                    )
                )
        else:
            for loop in measurement_loops:
                if not isinstance(loop, MeasurementLoop):
                    raise TypeError("Sequence entries must be MeasurementLoop instances.")
                loops.append(loop)
        loops.sort(key=lambda loop: loop.axis_coordinate)
        return tuple(loops)

    def _build_panel(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        curvature: np.ndarray,
        loops: Sequence[MeasurementLoop],
        lower: MeasurementLoop,
        upper: MeasurementLoop,
        region_mask: np.ndarray,
        lateral_coords: np.ndarray,
        side_sign: float,
        side_name: str,
        size_scale: float,
    ) -> SeamPanel | None:
        tol = self.seam_tolerance
        if side_sign >= 0.0:
            side_mask = lateral_coords >= (-tol)
        else:
            side_mask = lateral_coords <= tol

        region_indices = np.nonzero(region_mask & side_mask)[0]
        if region_indices.size == 0:
            return None

        mask = np.zeros(vertices.shape[0], dtype=bool)
        mask[region_indices] = True

        face_mask = mask[faces].all(axis=1)
        if not np.any(face_mask):
            return None

        selected_faces = faces[face_mask]
        global_indices = np.unique(selected_faces)
        index_lookup = {int(idx): position for position, idx in enumerate(global_indices)}
        local_faces = np.vectorize(index_lookup.get)(selected_faces)

        seam_vertices = tuple(
            int(idx)
            for idx in np.nonzero(
                (np.abs(lateral_coords) <= tol) & region_mask & mask
            )[0]
        )

        panel_curvature = float(curvature[global_indices].mean()) if global_indices.size else 0.0
        seam_curvature = (
            float(curvature[list(seam_vertices)].mean())
            if seam_vertices
            else panel_curvature
        )
        tension_ratio = seam_curvature / (panel_curvature + 1e-6) if panel_curvature > 0.0 else 1.0
        allowance = self.base_allowance * size_scale * (1.0 + 0.4 * np.tanh(tension_ratio - 1.0))
        allowance += self.seam_bias * min(panel_curvature, 1.0)

        loop_vertex_indices: MutableMapping[str, tuple[int, ...]] = {}
        for loop in (lower, upper):
            indices = tuple(idx for idx in loop.vertex_indices if mask[idx])
            loop_vertex_indices[loop.name] = indices

        metadata: Mapping[str, float] = MappingProxyType(
            {
                "seam_allowance": float(max(allowance, self.base_allowance * 0.4)),
                "mean_curvature": seam_curvature,
                "panel_curvature": panel_curvature,
                "tension_ratio": tension_ratio,
            }
        )

        name = f"{lower.name}_to_{upper.name}_{side_name}"
        return SeamPanel(
            name=name,
            anchor_loops=(lower.name, upper.name),
            side=side_name,
            vertices=vertices[global_indices],
            faces=local_faces.astype(int),
            global_indices=tuple(int(idx) for idx in global_indices),
            seam_vertices=seam_vertices,
            loop_vertex_indices=tuple(loop_vertex_indices.items()),
            metadata=metadata,
        )


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------


def _validate_indices(indices: Sequence[int]) -> tuple[int, ...]:
    result = []
    for value in indices:
        ivalue = int(value)
        if ivalue not in result:
            result.append(ivalue)
    return tuple(result)


def _measurement_scale(measurements: Mapping[str, float] | None) -> float:
    if not measurements:
        return 1.0
    ratios: list[float] = []
    for name, reference in LAYER_REFERENCE_MEASUREMENTS.items():
        if name in measurements:
            ratios.append(float(measurements[name]) / float(reference))
    if not ratios:
        return 1.0
    return float(np.mean(ratios)) ** 0.5


def _pairwise(items: Sequence[MeasurementLoop]) -> Iterable[tuple[MeasurementLoop, MeasurementLoop]]:
    for index in range(len(items) - 1):
        yield items[index], items[index + 1]


def _normalise_vector(vector: Vector) -> Vector:
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if norm < 1e-12:
        return np.zeros_like(arr)
    return arr / norm


def _orthogonal_basis(axis: Vector) -> Vector:
    axis = _normalise_vector(axis)
    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if np.allclose(axis, reference):
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    basis = reference - np.dot(reference, axis) * axis
    norm = np.linalg.norm(basis)
    if norm < 1e-8:
        basis = np.array([0.0, 1.0, 0.0], dtype=float)
        basis = basis - np.dot(basis, axis) * axis
        norm = np.linalg.norm(basis)
    return basis / max(norm, 1e-8)


def _extract_loop_vertices(
    vertices: np.ndarray,
    axis: Vector,
    target_coordinate: float,
    tolerance: float,
    lateral: Vector,
    anterior: Vector,
) -> tuple[int, ...]:
    coords = vertices @ axis
    diff = np.abs(coords - target_coordinate)
    current_tol = tolerance
    selection = np.nonzero(diff <= current_tol)[0]
    iterations = 0
    while selection.size < 6 and iterations < 6:
        current_tol *= 1.6
        selection = np.nonzero(diff <= current_tol)[0]
        iterations += 1
    if selection.size == 0:
        return ()

    projected = vertices[selection]
    basis_u = _normalise_vector(lateral)
    basis_v = _normalise_vector(np.cross(axis, basis_u))
    if np.linalg.norm(basis_v) < 1e-8:
        basis_v = _orthogonal_basis(axis)
    planar = np.column_stack((projected @ basis_u, projected @ basis_v))
    angles = np.arctan2(planar[:, 1], planar[:, 0])
    ordered = selection[np.argsort(angles)]
    return tuple(int(idx) for idx in ordered)


def _estimate_mean_curvature(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    adjacency: list[set[int]] = [set() for _ in range(vertices.shape[0])]
    for tri in faces:
        a, b, c = map(int, tri)
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    curvature = np.zeros(vertices.shape[0], dtype=float)
    for index, neighbours in enumerate(adjacency):
        if not neighbours:
            continue
        neighbour_positions = vertices[list(neighbours)]
        centroid = neighbour_positions.mean(axis=0)
        curvature[index] = float(np.linalg.norm(vertices[index] - centroid))
    if curvature.max() > 0.0:
        curvature /= float(curvature.max())
    return curvature


def _average_edge_length(vertices: np.ndarray, faces: np.ndarray) -> float:
    if faces.size == 0:
        return 0.01
    edges = set()
    for tri in faces:
        indices = [int(idx) for idx in tri]
        for start, end in ((0, 1), (1, 2), (2, 0)):
            a = indices[start]
            b = indices[end]
            if a > b:
                a, b = b, a
            edges.add((a, b))
    if not edges:
        return 0.01
    lengths = [
        float(np.linalg.norm(vertices[a] - vertices[b]))
        for a, b in edges
    ]
    return float(np.mean(lengths))

