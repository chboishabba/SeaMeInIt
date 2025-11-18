"""Utilities for exporting undersuit panels as 2D pattern files."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # pragma: no cover - optional dependency shim
    import numpy as np
except ImportError:  # pragma: no cover - numpy optional
    np = None  # type: ignore[assignment]


@dataclass(slots=True)
class Panel3D:
    """Representation of a single garment panel in 3D space."""

    name: str
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Panel3D":
        vertices = [tuple(map(float, vertex)) for vertex in payload.get("vertices", [])]
        faces = [tuple(int(idx) for idx in face) for face in payload.get("faces", [])]
        name = str(payload.get("name", "panel"))
        metadata = dict(payload.get("metadata", {}) or {})
        return cls(name=name, vertices=vertices, faces=faces, metadata=metadata)


@dataclass(slots=True)
class GrainlineAnnotation:
    """Directional grainline arrow rendered on a flattened panel."""

    origin: tuple[float, float]
    direction: tuple[float, float]
    length: float | None = None

    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        dx, dy = self.direction
        magnitude = math.hypot(dx, dy)
        if magnitude <= 1e-9:
            return self.origin, self.origin
        length = float(self.length or magnitude)
        ux = dx / magnitude
        uy = dy / magnitude
        half = length * 0.5
        start = (self.origin[0] - ux * half, self.origin[1] - uy * half)
        end = (self.origin[0] + ux * half, self.origin[1] + uy * half)
        return start, end

    def arrowheads(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        start, end = self.endpoints()
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return [], []
        ux = dx / length
        uy = dy / length
        size = max(length * 0.12, 1e-3)
        normal = (-uy, ux)
        def _arrow(point: tuple[float, float], sign: float) -> list[tuple[float, float]]:
            base = (point[0] - ux * size * sign, point[1] - uy * size * sign)
            width = size * 0.45
            return [
                point,
                (base[0] + normal[0] * width, base[1] + normal[1] * width),
                (base[0] - normal[0] * width, base[1] - normal[1] * width),
            ]

        return _arrow(start, -1.0), _arrow(end, 1.0)


@dataclass(slots=True)
class NotchAnnotation:
    """Marker rendered perpendicular to the panel outline."""

    position: tuple[float, float]
    tangent: tuple[float, float]
    normal: tuple[float, float]
    depth: float = 0.01
    width: float = 0.01
    label: str | None = None

    def triangle(self) -> list[tuple[float, float]]:
        tx, ty = self.tangent
        nx, ny = self.normal
        tangent_len = math.hypot(tx, ty) or 1.0
        normal_len = math.hypot(nx, ny) or 1.0
        ux = tx / tangent_len
        uy = ty / tangent_len
        vx = nx / normal_len
        vy = ny / normal_len
        half_width = self.width * 0.5
        tip = (self.position[0] + vx * self.depth, self.position[1] + vy * self.depth)
        base_left = (
            self.position[0] - vx * self.depth * 0.2 + ux * half_width,
            self.position[1] - vy * self.depth * 0.2 + uy * half_width,
        )
        base_right = (
            self.position[0] - vx * self.depth * 0.2 - ux * half_width,
            self.position[1] - vy * self.depth * 0.2 - uy * half_width,
        )
        return [tip, base_left, base_right]


@dataclass(slots=True)
class FoldAnnotation:
    """Representation of a fold line indicator."""

    start: tuple[float, float]
    end: tuple[float, float]
    kind: str = "valley"

    def chevrons(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return []
        ux = dx / length
        uy = dy / length
        nx = -uy
        ny = ux
        spacing = max(length / 6.0, 1e-3)
        size = spacing * 0.6
        chevrons: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for offset in (-spacing, 0.0, spacing):
            mx = (self.start[0] + self.end[0]) * 0.5 + ux * offset
            my = (self.start[1] + self.end[1]) * 0.5 + uy * offset
            chevrons.append(((mx - nx * size, my - ny * size), (mx + nx * size, my + ny * size)))
        return chevrons


@dataclass(slots=True)
class LabelAnnotation:
    """Label text placed on the flattened panel."""

    text: str
    position: tuple[float, float]
    rotation: float = 0.0


@dataclass(slots=True)
class PanelAnnotations:
    """Aggregate annotation payload for a panel."""

    grainlines: list[GrainlineAnnotation] = field(default_factory=list)
    notches: list[NotchAnnotation] = field(default_factory=list)
    folds: list[FoldAnnotation] = field(default_factory=list)
    label: LabelAnnotation | None = None


@dataclass(slots=True)
class Panel2D:
    """Flattened 2D panel representation including seam and cut outlines."""

    name: str
    seam_outline: list[tuple[float, float]]
    seam_allowance: float
    cut_outline: list[tuple[float, float]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    grainlines: list[GrainlineAnnotation] = field(default_factory=list)
    notches: list[NotchAnnotation] = field(default_factory=list)
    folds: list[FoldAnnotation] = field(default_factory=list)
    label: LabelAnnotation | None = None


def build_panel_annotations(
    outline: Sequence[tuple[float, float]],
    *,
    seam_metadata: Mapping[str, Any] | None = None,
    panel_metadata: Mapping[str, Any] | None = None,
    panel_name: str | None = None,
) -> PanelAnnotations:
    """Create annotation primitives for a flattened panel."""

    seam_metadata = seam_metadata or {}
    panel_metadata = panel_metadata or {}

    grainlines = _grainlines_from_metadata(outline, panel_metadata)
    notches = _notches_from_metadata(outline, seam_metadata)
    folds = _folds_from_metadata(outline, seam_metadata)
    label = _label_from_metadata(outline, panel_metadata, seam_metadata, panel_name)

    return PanelAnnotations(
        grainlines=grainlines,
        notches=notches,
        folds=folds,
        label=label,
    )


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        area += x0 * y1 - x1 * y0
    return area * 0.5


def _outline_length(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    closed = list(points)
    if closed[0] != closed[-1]:
        closed.append(closed[0])
    for start, end in zip(closed, closed[1:]):
        total += math.dist(start, end)
    return total


def _outline_centroid(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    area = _polygon_area(points)
    if abs(area) <= 1e-9:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    factor = 1.0 / (6.0 * area)
    cx = 0.0
    cy = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    return cx * factor, cy * factor


def _outline_point_frame(
    points: Sequence[tuple[float, float]],
    fraction: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    if not points:
        return (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)
    closed = list(points)
    if closed[0] != closed[-1]:
        closed.append(closed[0])
    total = 0.0
    segments: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    for start, end in zip(closed, closed[1:]):
        length = math.dist(start, end)
        if length <= 1e-9:
            continue
        segments.append((start, end, length))
        total += length
    if total <= 1e-9:
        start, end = closed[0], closed[1]
        tangent = (end[0] - start[0], end[1] - start[1])
        normal = (-tangent[1], tangent[0])
        return start, tangent, normal
    orientation = 1.0 if _polygon_area(points) >= 0 else -1.0
    target = (fraction % 1.0) * total
    travelled = 0.0
    for start, end, length in segments:
        if travelled + length >= target - 1e-9:
            ratio = (target - travelled) / length if length > 1e-9 else 0.0
            x = start[0] + (end[0] - start[0]) * ratio
            y = start[1] + (end[1] - start[1]) * ratio
            tangent = (end[0] - start[0], end[1] - start[1])
            normal = (-tangent[1] * orientation, tangent[0] * orientation)
            return (x, y), tangent, normal
        travelled += length
    start, end, _ = segments[-1]
    tangent = (end[0] - start[0], end[1] - start[1])
    normal = (-tangent[1] * orientation, tangent[0] * orientation)
    return end, tangent, normal


def _outline_frame_from_point(
    points: Sequence[tuple[float, float]],
    point: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    if not points:
        return point, (1.0, 0.0), (0.0, 1.0)
    closed = list(points)
    if closed[0] != closed[-1]:
        closed.append(closed[0])
    best: tuple[float, tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    orientation = 1.0 if _polygon_area(points) >= 0 else -1.0
    for start, end in zip(closed, closed[1:]):
        sx, sy = start
        ex, ey = end
        vx = ex - sx
        vy = ey - sy
        seg_len_sq = vx * vx + vy * vy
        if seg_len_sq <= 1e-12:
            continue
        px, py = point
        t = ((px - sx) * vx + (py - sy) * vy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        proj = (sx + vx * t, sy + vy * t)
        dist_sq = (proj[0] - px) ** 2 + (proj[1] - py) ** 2
        if best is None or dist_sq < best[0]:
            tangent = (vx, vy)
            normal = (-vy * orientation, vx * orientation)
            best = (dist_sq, proj, tangent, normal)
    if best is None:
        return point, (1.0, 0.0), (0.0, 1.0)
    _, proj, tangent, normal = best
    return proj, tangent, normal


def _coerce_direction(values: Sequence[Any]) -> tuple[float, float]:
    coords = [float(v) for v in values[:2]]
    if not coords:
        return 0.0, 1.0
    if len(coords) == 1:
        return coords[0], 0.0
    return coords[0], coords[1]


def _grainlines_from_metadata(
    outline: Sequence[tuple[float, float]],
    panel_metadata: Mapping[str, Any],
) -> list[GrainlineAnnotation]:
    payload = panel_metadata.get("grainline") or panel_metadata.get("grainlines")
    entries: Sequence[Any]
    if isinstance(payload, Mapping):
        entries = [payload]
    elif isinstance(payload, Sequence):
        entries = list(payload)
    else:
        entries = []

    if not entries:
        return [_default_grainline(outline)]

    annotations: list[GrainlineAnnotation] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        direction_raw = entry.get("direction", entry.get("vector"))
        if isinstance(direction_raw, Sequence):
            direction = _coerce_direction(direction_raw)
        else:
            direction = (0.0, 1.0)
        origin_raw = entry.get("origin")
        if isinstance(origin_raw, Sequence) and len(origin_raw) >= 2:
            origin = (float(origin_raw[0]), float(origin_raw[1]))
        else:
            origin = _outline_centroid(outline)
        length = entry.get("length")
        annotations.append(
            GrainlineAnnotation(
                origin=origin,
                direction=direction,
                length=float(length) if length is not None else None,
            )
        )
    return annotations or [_default_grainline(outline)]


def _default_grainline(outline: Sequence[tuple[float, float]]) -> GrainlineAnnotation:
    cx, cy = _outline_centroid(outline)
    xs = [point[0] for point in outline] or [0.0]
    ys = [point[1] for point in outline] or [0.0]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    length = max(height, width, 1e-2)
    return GrainlineAnnotation(origin=(cx, cy), direction=(0.0, 1.0), length=length)


def _notches_from_metadata(
    outline: Sequence[tuple[float, float]],
    seam_metadata: Mapping[str, Any],
) -> list[NotchAnnotation]:
    entries: list[Any] = []
    raw_notches = seam_metadata.get("notches")
    if isinstance(raw_notches, Mapping):
        raw_notches = raw_notches.get("entries", [])
    if isinstance(raw_notches, Sequence):
        entries.extend(raw_notches)

    correspondences = seam_metadata.get("correspondences", [])
    if isinstance(correspondences, Sequence):
        for item in correspondences:
            if isinstance(item, Mapping) and (
                item.get("kind") == "notch" or item.get("notch") is True
            ):
                entries.append(item)

    if not entries:
        return []

    notches: list[NotchAnnotation] = []
    total_length = _outline_length(outline)
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        depth = float(entry.get("depth", 0.008))
        width = float(entry.get("width", depth * 1.5))
        label = entry.get("label")
        fraction: float | None
        if "fraction" in entry:
            fraction = float(entry.get("fraction", 0.0))
        elif "distance" in entry and total_length > 1e-9:
            fraction = float(entry["distance"]) / total_length
        else:
            fraction = None
        if fraction is not None:
            position, tangent, normal = _outline_point_frame(outline, fraction)
        else:
            raw_position = entry.get("position") or entry.get("point")
            if isinstance(raw_position, Sequence) and len(raw_position) >= 2:
                candidate = (float(raw_position[0]), float(raw_position[1]))
                position, tangent, normal = _outline_frame_from_point(outline, candidate)
            else:
                continue
        notches.append(
            NotchAnnotation(
                position=position,
                tangent=tangent,
                normal=normal,
                depth=depth,
                width=width,
                label=str(label) if label is not None else None,
            )
        )
    return notches


def _folds_from_metadata(
    outline: Sequence[tuple[float, float]],
    seam_metadata: Mapping[str, Any],
) -> list[FoldAnnotation]:
    raw_folds = seam_metadata.get("folds") or seam_metadata.get("fold_lines")
    if isinstance(raw_folds, Mapping):
        entries = raw_folds.get("entries", [])
    elif isinstance(raw_folds, Sequence):
        entries = list(raw_folds)
    else:
        entries = []

    folds: list[FoldAnnotation] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        start_point: tuple[float, float] | None = None
        end_point: tuple[float, float] | None = None
        if "start_fraction" in entry and "end_fraction" in entry:
            start_point, _, _ = _outline_point_frame(outline, float(entry["start_fraction"]))
            end_point, _, _ = _outline_point_frame(outline, float(entry["end_fraction"]))
        elif "positions" in entry:
            positions = entry.get("positions")
            if isinstance(positions, Sequence) and len(positions) >= 2:
                start_raw = positions[0]
                end_raw = positions[1]
                if isinstance(start_raw, Sequence) and len(start_raw) >= 2:
                    start_point = (float(start_raw[0]), float(start_raw[1]))
                if isinstance(end_raw, Sequence) and len(end_raw) >= 2:
                    end_point = (float(end_raw[0]), float(end_raw[1]))
        if start_point is None or end_point is None:
            continue
        folds.append(
            FoldAnnotation(
                start=start_point,
                end=end_point,
                kind=str(entry.get("kind", "valley")),
            )
        )
    return folds


def _label_from_metadata(
    outline: Sequence[tuple[float, float]],
    panel_metadata: Mapping[str, Any],
    seam_metadata: Mapping[str, Any],
    panel_name: str | None,
) -> LabelAnnotation | None:
    label_payload = (
        panel_metadata.get("label")
        or panel_metadata.get("panel_label")
        or seam_metadata.get("label")
    )
    centroid = _outline_centroid(outline)
    if isinstance(label_payload, str):
        text = label_payload.strip()
        if not text:
            return None
        return LabelAnnotation(text=text, position=centroid)
    if isinstance(label_payload, Mapping):
        text = str(label_payload.get("text") or label_payload.get("value") or "").strip()
        if not text and panel_name:
            text = panel_name
        position_raw = label_payload.get("position")
        if isinstance(position_raw, Sequence) and len(position_raw) >= 2:
            position = (float(position_raw[0]), float(position_raw[1]))
        else:
            position = centroid
        rotation = float(label_payload.get("rotation", 0.0) or 0.0)
        return LabelAnnotation(text=text, position=position, rotation=rotation)

    fallback = panel_metadata.get("label_text") or seam_metadata.get("label_text")
    if fallback is None and panel_name:
        fallback = panel_name
    if fallback is None:
        return None
    text = str(fallback).strip()
    if not text:
        return None
    return LabelAnnotation(text=text, position=centroid)

    def __post_init__(self) -> None:
        self.seam_allowance = float(self.seam_allowance)
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)
        self.seam_outline = [(float(x), float(y)) for x, y in self.seam_outline]
        if self.cut_outline is None:
            if not self.seam_outline:
                self.cut_outline = []
            elif self.seam_allowance <= 1e-9:
                self.cut_outline = list(self.seam_outline)
            else:
                offset = _offset_outline(self.seam_outline, float(self.seam_allowance))
                if offset:
                    self.cut_outline = offset
                else:
                    self.cut_outline = []
                    warnings = self.metadata.setdefault("warnings", [])
                    message = "Panel lacks sufficient geometry for seam allowance offset."
                    if message not in warnings:
                        warnings.append(message)
        else:
            self.cut_outline = [(float(x), float(y)) for x, y in self.cut_outline]


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    area = 0.0
    if not points:
        return area
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _line_intersection(
    point_a: tuple[float, float],
    direction_a: tuple[float, float],
    point_b: tuple[float, float],
    direction_b: tuple[float, float],
) -> tuple[float, float] | None:
    det = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    if math.isclose(det, 0.0, abs_tol=1e-12):
        return None
    diff_x = point_b[0] - point_a[0]
    diff_y = point_b[1] - point_a[1]
    t = (diff_x * direction_b[1] - diff_y * direction_b[0]) / det
    return (point_a[0] + direction_a[0] * t, point_a[1] + direction_a[1] * t)


def _offset_outline(
    points: Sequence[tuple[float, float]],
    distance: float,
) -> list[tuple[float, float]]:
    if distance <= 0.0:
        return [(float(x), float(y)) for x, y in points]
    if len(points) < 3:
        return []

    area = _polygon_area(points)
    if math.isclose(area, 0.0, abs_tol=1e-9):
        return []
    orientation = 1.0 if area > 0.0 else -1.0

    offset_points: list[tuple[float, float]] = []
    count = len(points)
    for idx in range(count):
        prev_point = points[(idx - 1) % count]
        current_point = points[idx]
        next_point = points[(idx + 1) % count]

        edge_prev = (current_point[0] - prev_point[0], current_point[1] - prev_point[1])
        edge_next = (next_point[0] - current_point[0], next_point[1] - current_point[1])

        length_prev = math.hypot(*edge_prev)
        length_next = math.hypot(*edge_next)
        if length_prev <= 1e-9 or length_next <= 1e-9:
            return []

        normal_prev = (
            orientation * edge_prev[1] / length_prev,
            -orientation * edge_prev[0] / length_prev,
        )
        normal_next = (
            orientation * edge_next[1] / length_next,
            -orientation * edge_next[0] / length_next,
        )

        point_prev = (
            current_point[0] + normal_prev[0] * distance,
            current_point[1] + normal_prev[1] * distance,
        )
        point_next = (
            current_point[0] + normal_next[0] * distance,
            current_point[1] + normal_next[1] * distance,
        )

        intersection = _line_intersection(point_prev, edge_prev, point_next, edge_next)
        if intersection is None:
            avg_normal = (normal_prev[0] + normal_next[0], normal_prev[1] + normal_next[1])
            length_avg = math.hypot(*avg_normal)
            if length_avg <= 1e-9:
                return []
            intersection = (
                current_point[0] + distance * avg_normal[0] / length_avg,
                current_point[1] + distance * avg_normal[1] / length_avg,
            )
        offset_points.append((float(intersection[0]), float(intersection[1])))

    return offset_points


class CADBackend:
    """Protocol for CAD flattening backends."""

    def flatten_panels(
        self,
        panels: Sequence[Panel3D],
        seams: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        scale: float,
        seam_allowance: float,
    ) -> list[Panel2D]:  # pragma: no cover - interface definition only
        raise NotImplementedError


class SimplePlaneProjectionBackend(CADBackend):
    """Fallback backend approximating flattening via PCA plane projection."""

    def flatten_panels(
        self,
        panels: Sequence[Panel3D],
        seams: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        scale: float,
        seam_allowance: float,
    ) -> list[Panel2D]:
        flattened: list[Panel2D] = []
        seam_lookup = seams or {}
        for panel in panels:
            seam_outline: list[tuple[float, float]]
            if not panel.vertices:
                seam_outline = []
            elif np is not None:
                points = np.asarray(panel.vertices, dtype=float)
                centroid = points.mean(axis=0)
                shifted = points - centroid
                _, _, vh = np.linalg.svd(shifted, full_matrices=False)
                basis = vh[:2, :]
                projected = shifted @ basis.T
                seam_outline = _ordered_outline(projected * scale)
            else:
                seam_outline = [
                    (float(vertex[0]) * scale, float(vertex[1]) * scale)
                    for vertex in panel.vertices
                ]

            allowance = float(seam_lookup.get(panel.name, {}).get("seam_allowance", seam_allowance))
            metadata = {
                "original_vertex_count": len(panel.vertices),
                "seam_allowance": allowance,
            }
            annotations = build_panel_annotations(
                seam_outline,
                seam_metadata=seam_lookup.get(panel.name, {}),
                panel_metadata=panel.metadata,
                panel_name=panel.name,
            )
            flattened.append(
                Panel2D(
                    name=panel.name,
                    seam_outline=seam_outline,
                    seam_allowance=allowance,
                    metadata=metadata,
                    grainlines=annotations.grainlines,
                    notches=annotations.notches,
                    folds=annotations.folds,
                    label=annotations.label,
                )
            )
        return flattened


def _fan_triangulation(count: int) -> list[tuple[int, int, int]]:
    if count < 3:
        return []
    return [(0, i, i + 1) for i in range(1, count - 1)]


class LSCMUnwrapBackend(CADBackend):
    """Flatten panels using libigl's Least Squares Conformal Maps solver."""

    def __init__(self, *, fallback: CADBackend | None = None) -> None:
        if np is None:  # pragma: no cover - numpy is an optional dependency up-stream
            raise ModuleNotFoundError(
                "LSCMUnwrapBackend requires NumPy. Install the 'numpy' package to enable it."
            )
        try:
            import igl  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised in tests via stubs
            raise ModuleNotFoundError(
                "LSCMUnwrapBackend requires the 'igl' package. Install python-igl to enable UV unwrapping."
            ) from exc
        self._igl = igl
        self._fallback = fallback or SimplePlaneProjectionBackend()

    def flatten_panels(
        self,
        panels: Sequence[Panel3D],
        seams: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        scale: float,
        seam_allowance: float,
    ) -> list[Panel2D]:
        flattened: list[Panel2D] = []
        seam_lookup = seams or {}
        for panel in panels:
            allowance = float(seam_lookup.get(panel.name, {}).get("seam_allowance", seam_allowance))
            metadata = {
                "original_vertex_count": len(panel.vertices),
                "seam_allowance": allowance,
                "backend": "lscm",
            }
            try:
                seam_outline = self._unwrap_panel(panel, scale=scale)
            except Exception:
                fallback = self._fallback.flatten_panels(
                    [panel],
                    seams,
                    scale=scale,
                    seam_allowance=seam_allowance,
                )
                flattened.extend(fallback)
                continue
            annotations = build_panel_annotations(
                outline,
                seam_metadata=seam_lookup.get(panel.name, {}),
                panel_metadata=panel.metadata,
                panel_name=panel.name,
            )
            flattened.append(
                Panel2D(
                    name=panel.name,
                    seam_outline=seam_outline,
                    seam_allowance=allowance,
                    metadata=metadata,
                    grainlines=annotations.grainlines,
                    notches=annotations.notches,
                    folds=annotations.folds,
                    label=annotations.label,
                )
            )
        return flattened

    def _unwrap_panel(self, panel: Panel3D, *, scale: float) -> list[tuple[float, float]]:
        if len(panel.vertices) < 3:
            return []
        vertices = np.asarray(panel.vertices, dtype=float)
        faces = panel.faces or _fan_triangulation(len(panel.vertices))
        if not faces:
            raise ValueError("Panel must provide at least one triangular face for LSCM unwrapping.")
        triangles = np.asarray(faces, dtype=int)

        boundary = self._igl.boundary_loop(triangles)
        if boundary.size < 2:
            raise ValueError("Panel must have a boundary loop for conformal unwrapping.")

        anchors = np.array(
            [int(boundary[0]), int(boundary[int(boundary.size // 2)])],
            dtype=int,
        )
        anchor_targets = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

        _, parameterisation = self._igl.lscm(vertices, triangles, anchors, anchor_targets)
        if parameterisation.shape[0] != vertices.shape[0]:
            raise ValueError("LSCM returned coordinates with unexpected dimensionality.")

        if boundary.size >= 3:
            outline_points = parameterisation[boundary]
        else:
            outline_points = parameterisation
        ordered = _ordered_outline(outline_points * float(scale))
        return _cleanup_panel_outline(ordered)


class PatternExporter:
    """Export undersuit panels to 2D pattern files in multiple formats."""

    def __init__(
        self,
        *,
        backend: CADBackend | str | None = None,
        scale: float = 1.0,
        seam_allowance: float = 0.01,
    ) -> None:
        self.backend = self._resolve_backend(backend)
        self.scale = float(scale)
        self.seam_allowance = float(seam_allowance)

    @staticmethod
    def _resolve_backend(candidate: CADBackend | str | None) -> CADBackend:
        if candidate is None:
            return SimplePlaneProjectionBackend()
        if isinstance(candidate, str):
            normalized = candidate.lower()
            if normalized == "simple":
                return SimplePlaneProjectionBackend()
            if normalized == "lscm":
                return LSCMUnwrapBackend()
            raise ValueError(f"Unknown pattern backend '{candidate}'.")
        return candidate

    def export(
        self,
        undersuit_mesh: Mapping[str, Any],
        seams: Mapping[str, Mapping[str, Any]] | None,
        *,
        output_dir: Path | str,
        formats: Iterable[str] = ("pdf", "svg", "dxf"),
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        panels = [Panel3D.from_mapping(panel) for panel in undersuit_mesh.get("panels", [])]
        flattened = self.backend.flatten_panels(
            panels,
            seams,
            scale=self.scale,
            seam_allowance=self.seam_allowance,
        )

        combined_metadata = {
            "scale": self.scale,
            "seam_allowance": self.seam_allowance,
            "panel_count": len(flattened),
        }
        panel_warnings: dict[str, list[str]] = {}
        for panel in flattened:
            warnings = panel.metadata.get("warnings")
            if warnings:
                panel_warnings[panel.name] = [str(message) for message in warnings]
        if panel_warnings:
            combined_metadata["panel_warnings"] = panel_warnings
        if metadata:
            combined_metadata.update(metadata)

        created: dict[str, Path] = {}
        for fmt in formats:
            normalized = fmt.lower()
            path = output_path / f"undersuit_pattern.{normalized}"
            if normalized == "svg":
                _write_svg(path, flattened, combined_metadata)
            elif normalized == "dxf":
                _write_dxf(path, flattened, combined_metadata)
            elif normalized == "pdf":
                _write_pdf(path, flattened, combined_metadata)
            else:  # pragma: no cover - defensive for unsupported formats
                msg = f"Unsupported pattern export format: {fmt}"
                raise ValueError(msg)
            created[normalized] = path
        return created


def _ordered_outline(points: "np.ndarray") -> list[tuple[float, float]]:
    if len(points) == 0:
        return []
    if len(points) < 3:
        return [(float(x), float(y)) for x, y in points]

    centroid = points.mean(axis=0)
    offsets = points - centroid
    norms = np.linalg.norm(offsets, axis=1)
    if np.all(norms < 1e-9):
        return [(float(x), float(y)) for x, y in points]

    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    order = np.argsort(angles)
    ordered = points[order]
    deduped: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in ordered:
        key = (int(round(float(x) * 1e6)), int(round(float(y) * 1e6)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((float(x), float(y)))
    return deduped


def _cleanup_panel_outline(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Normalize, smooth, and simplify an outline polyline."""

    normalized = _dedupe_consecutive([(float(x), float(y)) for x, y in points])
    if len(normalized) < 3:
        return normalized

    without_outliers = _remove_outlier_edges(normalized)
    if len(without_outliers) < 3:
        return without_outliers

    smoothed = _laplacian_smooth(without_outliers, iterations=2, weight=0.5)
    simplified = _simplify_polyline(smoothed)
    return simplified


def _dedupe_consecutive(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    last: tuple[float, float] | None = None
    for point in points:
        if last is not None and math.isclose(point[0], last[0], rel_tol=1e-9, abs_tol=1e-9) and math.isclose(
            point[1], last[1], rel_tol=1e-9, abs_tol=1e-9
        ):
            continue
        deduped.append(point)
        last = point
    if len(deduped) >= 2 and math.isclose(deduped[0][0], deduped[-1][0], rel_tol=1e-9, abs_tol=1e-9) and math.isclose(
        deduped[0][1], deduped[-1][1], rel_tol=1e-9, abs_tol=1e-9
    ):
        deduped.pop()
    return deduped


def _remove_outlier_edges(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned = list(points)
    if len(cleaned) < 4:
        return cleaned

    for _ in range(len(points)):
        count = len(cleaned)
        if count < 4:
            break
        lengths = [
            _euclidean_distance(cleaned[i], cleaned[(i + 1) % count])
            for i in range(count)
        ]
        positive = [length for length in lengths if length > 0]
        if not positive:
            break
        median_length = statistics.median(positive)
        max_length = max(lengths)
        if median_length <= 0:
            threshold = max_length
        else:
            threshold = median_length * 3.0
        if max_length <= threshold:
            break
        idx = lengths.index(max_length)
        remove_idx = (idx + 1) % count
        cleaned.pop(remove_idx)
    return cleaned


def _laplacian_smooth(
    points: Sequence[tuple[float, float]], *, iterations: int = 1, weight: float = 0.5
) -> list[tuple[float, float]]:
    if len(points) < 3:
        return list(points)
    smoothed = list(points)
    centroid = _polygon_centroid(smoothed)
    for _ in range(max(1, int(iterations))):
        next_points: list[tuple[float, float]] = []
        count = len(smoothed)
        for idx, current in enumerate(smoothed):
            prev_point = smoothed[(idx - 1) % count]
            next_point = smoothed[(idx + 1) % count]
            avg = ((prev_point[0] + next_point[0]) / 2.0, (prev_point[1] + next_point[1]) / 2.0)
            candidate = (
                current[0] * (1.0 - weight) + avg[0] * weight,
                current[1] * (1.0 - weight) + avg[1] * weight,
            )
            constrained = _constrain_to_interior(candidate, current, centroid)
            next_points.append(constrained)
        smoothed = next_points
        centroid = _polygon_centroid(smoothed)
    return smoothed


def _constrain_to_interior(
    candidate: tuple[float, float],
    original: tuple[float, float],
    centroid: tuple[float, float],
) -> tuple[float, float]:
    vec_candidate = (candidate[0] - centroid[0], candidate[1] - centroid[1])
    vec_original = (original[0] - centroid[0], original[1] - centroid[1])
    candidate_length = math.hypot(*vec_candidate)
    original_length = math.hypot(*vec_original)
    if original_length == 0:
        return candidate
    if candidate_length <= original_length or candidate_length == 0:
        return candidate
    scale = original_length / candidate_length
    return (centroid[0] + vec_candidate[0] * scale, centroid[1] + vec_candidate[1] * scale)


def _simplify_polyline(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 3:
        return list(points)
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    scale = max(max_x - min_x, max_y - min_y)
    tolerance = 1e-3 if scale == 0 else scale * 0.005
    simplified = _douglas_peucker(points + points[:1], tolerance)
    if simplified and simplified[-1] == simplified[0]:
        simplified.pop()
    return simplified


def _douglas_peucker(points: Sequence[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return list(points)
    first = points[0]
    last = points[-1]
    max_distance = -1.0
    index = -1
    for idx in range(1, len(points) - 1):
        distance = _point_to_segment_distance(points[idx], first, last)
        if distance > max_distance:
            max_distance = distance
            index = idx
    if max_distance > tolerance:
        left = _douglas_peucker(points[: index + 1], tolerance)
        right = _douglas_peucker(points[index:], tolerance)
        return left[:-1] + right
    return [first, last]


def _point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    if start == end:
        return _euclidean_distance(point, start)
    seg_vec = (end[0] - start[0], end[1] - start[1])
    point_vec = (point[0] - start[0], point[1] - start[1])
    seg_len_sq = seg_vec[0] ** 2 + seg_vec[1] ** 2
    proj = max(0.0, min(1.0, (point_vec[0] * seg_vec[0] + point_vec[1] * seg_vec[1]) / seg_len_sq))
    closest = (start[0] + seg_vec[0] * proj, start[1] + seg_vec[1] * proj)
    return _euclidean_distance(point, closest)


def _euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_pdf(path: Path, panels: Sequence[Panel2D], metadata: Mapping[str, Any]) -> None:
    min_x, min_y, max_x, max_y = _panel_bounds(panels)
    points_per_meter = 72.0 / 0.0254
    margin = 36.0
    drawing_width = (max_x - min_x) * points_per_meter
    drawing_height = (max_y - min_y) * points_per_meter
    width = max(200.0, drawing_width + margin * 2)
    height = max(200.0, drawing_height + margin * 2)

    def to_pdf(point: tuple[float, float]) -> tuple[float, float]:
        return (
            (point[0] - min_x) * points_per_meter + margin,
            (point[1] - min_y) * points_per_meter + margin,
        )

    info_lines = [
        "SMII Pattern Export",
        f"Scale: {metadata['scale']}",
        f"Seam allowance: {metadata['seam_allowance']}",
        f"Panels: {metadata['panel_count']}",
    ]

    summary_lines: list[str] = []
    for panel in panels:
        seam_coords = ", ".join(f"{x:.2f},{y:.2f}" for x, y in panel.seam_outline)
        cut_coords = ", ".join(f"{x:.2f},{y:.2f}" for x, y in (panel.cut_outline or []))
        if seam_coords:
            summary_lines.append(
                f"{panel.name} seam: {seam_coords} (allowance={panel.seam_allowance})"
            )
        else:
            summary_lines.append(
                f"{panel.name} seam: unavailable (allowance={panel.seam_allowance})"
            )
        if cut_coords:
            summary_lines.append(f"{panel.name} cut: {cut_coords}")
        else:
            summary_lines.append(f"{panel.name} cut: unavailable")

    drawing_ops: list[str] = []
    drawing_ops.extend(["0.6 w", "0 0 0 RG", "0 0 0 rg", "[] 0 d"])
    grain_color = (0.102, 0.451, 0.909)
    fold_color = (0.333, 0.333, 0.333)

    label_blocks: list[list[str]] = []

    for panel in panels:
        if panel.outline:
            transformed = [to_pdf(point) for point in panel.outline]
            x0, y0 = transformed[0]
            drawing_ops.append(f"{x0:.2f} {y0:.2f} m")
            for x, y in transformed[1:]:
                drawing_ops.append(f"{x:.2f} {y:.2f} l")
            drawing_ops.append("h")
            drawing_ops.append("S")

            for grain in panel.grainlines:
                start_pt, end_pt = grain.endpoints()
                start = to_pdf(start_pt)
                end = to_pdf(end_pt)
                drawing_ops.extend(
                    [
                        f"{grain_color[0]:.3f} {grain_color[1]:.3f} {grain_color[2]:.3f} RG",
                        "0.4 w",
                        "[] 0 d",
                        f"{start[0]:.2f} {start[1]:.2f} m",
                        f"{end[0]:.2f} {end[1]:.2f} l",
                        "S",
                    ]
                )
                drawing_ops.append(
                    f"{grain_color[0]:.3f} {grain_color[1]:.3f} {grain_color[2]:.3f} rg"
                )
                for arrow in grain.arrowheads():
                    if not arrow:
                        continue
                    a0 = to_pdf(arrow[0])
                    a1 = to_pdf(arrow[1])
                    a2 = to_pdf(arrow[2])
                    drawing_ops.extend(
                        [
                            f"{a0[0]:.2f} {a0[1]:.2f} m",
                            f"{a1[0]:.2f} {a1[1]:.2f} l",
                            f"{a2[0]:.2f} {a2[1]:.2f} l",
                            "h",
                            "f",
                        ]
                    )
                drawing_ops.extend(["0 0 0 rg", "0 0 0 RG", "0.6 w"])

            for notch in panel.notches:
                triangle = [to_pdf(point) for point in notch.triangle()]
                drawing_ops.extend(
                    [
                        "0 0 0 rg",
                        f"{triangle[0][0]:.2f} {triangle[0][1]:.2f} m",
                        f"{triangle[1][0]:.2f} {triangle[1][1]:.2f} l",
                        f"{triangle[2][0]:.2f} {triangle[2][1]:.2f} l",
                        "h",
                        "f",
                    ]
                )
                if notch.label:
                    label_x, label_y = to_pdf(
                        (
                            notch.position[0] + notch.normal[0] * 1.5,
                            notch.position[1] + notch.normal[1] * 1.5,
                        )
                    )
                    block = [
                        "0 0 0 rg",
                        "BT",
                        "/F1 6 Tf",
                        f"{label_x:.2f} {label_y:.2f} Td",
                        f"({_escape_pdf_text(str(notch.label))}) Tj",
                        "ET",
                    ]
                    label_blocks.append(block)

            for fold in panel.folds:
                start = to_pdf(fold.start)
                end = to_pdf(fold.end)
                drawing_ops.extend(
                    [
                        f"{fold_color[0]:.3f} {fold_color[1]:.3f} {fold_color[2]:.3f} RG",
                        "0.4 w",
                        "[2 2] 0 d",
                        f"{start[0]:.2f} {start[1]:.2f} m",
                        f"{end[0]:.2f} {end[1]:.2f} l",
                        "S",
                    ]
                )
                drawing_ops.append("[] 0 d")
                for chevron in fold.chevrons():
                    c0 = to_pdf(chevron[0])
                    c1 = to_pdf(chevron[1])
                    drawing_ops.extend(
                        [
                            f"{fold_color[0]:.3f} {fold_color[1]:.3f} {fold_color[2]:.3f} RG",
                            "0.3 w",
                            f"{c0[0]:.2f} {c0[1]:.2f} m",
                            f"{c1[0]:.2f} {c1[1]:.2f} l",
                            "S",
                        ]
                    )
                drawing_ops.extend(["0 0 0 RG", "0.6 w"])

        if panel.label:
            xs = [point[0] for point in panel.outline] or [panel.label.position[0]]
            ys = [point[1] for point in panel.outline] or [panel.label.position[1]]
            extent = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            px, py = to_pdf(panel.label.position)
            size = min(18.0, max(8.0, extent * 0.12))
            block: list[str] = ["0 0 0 rg", "BT", f"/F1 {size:.1f} Tf"]
            if abs(panel.label.rotation) > 1e-3:
                angle = math.radians(panel.label.rotation)
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                block.append(
                    f"{cos_a:.4f} {sin_a:.4f} {-sin_a:.4f} {cos_a:.4f} {px:.2f} {py:.2f} Tm"
                )
            else:
                block.append(f"{px:.2f} {py:.2f} Td")
            block.append(f"({_escape_pdf_text(panel.label.text)}) Tj")
            block.append("ET")
            label_blocks.append(block)

    header_ops = ["BT", "/F1 12 Tf"]
    top_text_y = height - margin
    for index, line in enumerate(info_lines):
        escaped = _escape_pdf_text(line)
        offset = 14.0 * index
        header_ops.append(f"1 0 0 1 {margin:.2f} {top_text_y - offset:.2f} Tm")
        header_ops.append(f"({escaped}) Tj")
    header_ops.append("ET")

    text_ops = ["0 0 0 rg", "BT", "/F1 12 Tf", f"{margin:.2f} {height - margin - 56:.2f} Td"]
    for index, line in enumerate(summary_lines):
        escaped = _escape_pdf_text(line)
        if index == 0:
            text_ops.append(f"({escaped}) Tj")
        else:
            text_ops.append(f"0 -14 Td ({escaped}) Tj")
    text_ops.append("ET")

    content_parts = drawing_ops + header_ops + text_ops
    for block in label_blocks:
        content_parts.extend(block)

    content_stream = "\n".join(content_parts).encode("utf-8")

    objects: list[bytes] = []
    catalog = b"<< /Type /Catalog /Pages 2 0 R >>"
    pages = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
    page = (
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.2f} {height:.2f}] "
        "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode("utf-8")
    )
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    contents = (
        f"<< /Length {len(content_stream)} >>\nstream\n".encode("utf-8")
        + content_stream
        + b"\nendstream"
    )
    objects.extend([catalog, pages, page, font, contents])

    buffer = bytearray()
    buffer.extend(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(buffer))
        buffer.extend(f"{idx} 0 obj\n".encode("utf-8"))
        buffer.extend(obj)
        if not obj.endswith(b"\n"):
            buffer.extend(b"\n")
        buffer.extend(b"endobj\n")
    xref_offset = len(buffer)
    count = len(objects) + 1
    buffer.extend(f"xref\n0 {count}\n".encode("utf-8"))
    buffer.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        buffer.extend(f"{offset:010d} 00000 n \n".encode("utf-8"))
    buffer.extend(
        f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode("utf-8")
    )
    path.write_bytes(buffer)


def _polygon_centroid(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    area = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        cross = x0 * y1 - x1 * y0
        area += cross
        centroid_x += (x0 + x1) * cross
        centroid_y += (y0 + y1) * cross
    area *= 0.5
    if math.isclose(area, 0.0):
        avg_x = sum(point[0] for point in points) / len(points)
        avg_y = sum(point[1] for point in points) / len(points)
        return avg_x, avg_y
    factor = 1.0 / (6.0 * area)
    return centroid_x * factor, centroid_y * factor


def _panel_bounds(panels: Sequence[Panel2D]) -> tuple[float, float, float, float]:
    all_points: list[tuple[float, float]] = []
    for panel in panels:
        if panel.cut_outline:
            all_points.extend(panel.cut_outline)
        if panel.seam_outline:
            all_points.extend(panel.seam_outline)
    min_x = min((point[0] for point in all_points), default=0.0)
    max_x = max((point[0] for point in all_points), default=1.0)
    min_y = min((point[1] for point in all_points), default=0.0)
    max_y = max((point[1] for point in all_points), default=1.0)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        max_y = min_y + 1.0
    return min_x, min_y, max_x, max_y


def _escape_svg_text(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_svg(path: Path, panels: Sequence[Panel2D], metadata: Mapping[str, Any]) -> None:
    min_x, min_y, max_x, max_y = _panel_bounds(panels)
    width = max_x - min_x
    height = max_y - min_y
    lines = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        f"<!-- Scale: {metadata['scale']} -->",
        f"<!-- Seam allowance: {metadata['seam_allowance']} -->",
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width:.2f}\" height=\"{height:.2f}\" viewBox=\"{min_x:.2f} {min_y:.2f} {width:.2f} {height:.2f}\">",
        "  <style>",
        "    .panel-outline { fill: none; stroke: #111; stroke-width: 0.6; }",
        "    .panel-grain { stroke: #1a73e8; stroke-width: 0.4; fill: none; }",
        "    .panel-grain-arrow { fill: #1a73e8; stroke: none; }",
        "    .panel-notch { fill: #111; stroke: none; }",
        "    .panel-notch-label { font: 3px sans-serif; fill: #111; }",
        "    .panel-fold { stroke: #555; stroke-width: 0.4; fill: none; stroke-dasharray: 2 2; }",
        "    .panel-fold-mountain { stroke-dasharray: 1 1; }",
        "    .panel-fold-chevron { stroke: #555; stroke-width: 0.3; }",
        "    .panel-label { font: 4px sans-serif; fill: #000; text-anchor: middle; }",
        "  </style>",
    ]
    if metadata.get("panel_warnings"):
        warnings_json = json.dumps(metadata["panel_warnings"], sort_keys=True)
        lines.insert(3, f"<!-- panel_warnings: {warnings_json} -->")
    for panel in panels:
        if panel.outline:
            points = " ".join(f"{x:.2f},{y:.2f}" for x, y in panel.outline)
            lines.append(
                f"  <polygon id=\"{panel.name}\" class=\"panel-outline\" points=\"{points}\" data-seam-allowance=\"{panel.seam_allowance}\" />"
            )
            for grain in panel.grainlines:
                start, end = grain.endpoints()
                lines.append(
                    f"  <line class=\"panel-grain\" x1=\"{start[0]:.2f}\" y1=\"{start[1]:.2f}\" x2=\"{end[0]:.2f}\" y2=\"{end[1]:.2f}\" />"
                )
                for arrow in grain.arrowheads():
                    if not arrow:
                        continue
                    arrow_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in arrow)
                    lines.append(
                        f"  <polygon class=\"panel-grain-arrow\" points=\"{arrow_points}\" />"
                    )
            for notch in panel.notches:
                triangle = notch.triangle()
                tri_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in triangle)
                lines.append(f"  <polygon class=\"panel-notch\" points=\"{tri_points}\" />")
                if notch.label:
                    lines.append(
                        f"  <text class=\"panel-notch-label\" x=\"{notch.position[0]:.2f}\" y=\"{notch.position[1]:.2f}\">{_escape_svg_text(str(notch.label))}</text>"
                    )
            for fold in panel.folds:
                kind = fold.kind.lower().replace(" ", "-")
                lines.append(
                    f"  <line class=\"panel-fold panel-fold-{kind}\" x1=\"{fold.start[0]:.2f}\" y1=\"{fold.start[1]:.2f}\" x2=\"{fold.end[0]:.2f}\" y2=\"{fold.end[1]:.2f}\" />"
                )
                for chevron in fold.chevrons():
                    (cx0, cy0), (cx1, cy1) = chevron
                    lines.append(
                        f"  <line class=\"panel-fold-chevron\" x1=\"{cx0:.2f}\" y1=\"{cy0:.2f}\" x2=\"{cx1:.2f}\" y2=\"{cy1:.2f}\" />"
                    )
        if panel.label:
            text = _escape_svg_text(panel.label.text)
            x, y = panel.label.position
            rotate_attr = (
                f" transform=\"rotate({panel.label.rotation:.2f} {x:.2f} {y:.2f})\""
                if abs(panel.label.rotation) > 1e-3
                else ""
            )
            lines.append(
                f"  <text class=\"panel-label\" x=\"{x:.2f}\" y=\"{y:.2f}\"{rotate_attr}>{text}</text>"
            )
        if not panel.seam_outline and not panel.cut_outline:
            continue
        lines.append(
            f"  <g id=\"{panel.name}\" data-seam-allowance=\"{panel.seam_allowance}\">"
        )
        if panel.cut_outline:
            cut_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in panel.cut_outline)
            lines.append(
                "    "
                f"<polygon class=\"cut-outline\" points=\"{cut_points}\" fill=\"none\" "
                "stroke=\"#d94f4f\" stroke-width=\"0.20\" stroke-linejoin=\"round\" />"
            )
        if panel.seam_outline:
            seam_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in panel.seam_outline)
            lines.append(
                "    "
                f"<polygon class=\"seam-outline\" points=\"{seam_points}\" fill=\"none\" "
                "stroke=\"#1a1a1a\" stroke-width=\"0.10\" stroke-dasharray=\"4 2\" stroke-linejoin=\"round\" />"
            )
        lines.append("  </g>")
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dxf(path: Path, panels: Sequence[Panel2D], metadata: Mapping[str, Any]) -> None:
    lines = [
        "0",
        "SECTION",
        "2",
        "HEADER",
        "9",
        "$SMII_SCALE",
        "1",
        str(metadata["scale"]),
        "9",
        "$SMII_SEAM_ALLOWANCE",
        "1",
        str(metadata["seam_allowance"]),
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "ENTITIES",
    ]
    for panel in panels:
        if not panel.outline:
            continue
        lines.extend([
            "0",
            "LWPOLYLINE",
            "8",
            panel.name,
            "90",
            str(len(panel.outline)),
        ])
        for x, y in panel.outline:
            lines.extend(["10", f"{x:.4f}", "20", f"{y:.4f}"])
        lines.extend(["43", f"{panel.seam_allowance:.4f}"])
        for grain in panel.grainlines:
            start, end = grain.endpoints()
            lines.extend([
                "0",
                "LINE",
                "8",
                f"{panel.name}_GRAIN",
                "10",
                f"{start[0]:.4f}",
                "20",
                f"{start[1]:.4f}",
                "11",
                f"{end[0]:.4f}",
                "21",
                f"{end[1]:.4f}",
            ])
            for arrow in grain.arrowheads():
                if not arrow:
                    continue
                lines.extend([
                    "0",
                    "LWPOLYLINE",
                    "8",
                    f"{panel.name}_GRAIN",
                    "90",
                    "3",
                    "70",
                    "1",
                ])
                for x, y in arrow:
                    lines.extend(["10", f"{x:.4f}", "20", f"{y:.4f}"])
        for notch in panel.notches:
            triangle = notch.triangle()
            lines.extend([
                "0",
                "LWPOLYLINE",
                "8",
                f"{panel.name}_NOTCH",
                "90",
                "3",
                "70",
                "1",
            ])
            for x, y in triangle:
                lines.extend(["10", f"{x:.4f}", "20", f"{y:.4f}"])
            if notch.label:
                lines.extend([
                    "0",
                    "TEXT",
                    "8",
                    f"{panel.name}_NOTCH",
                    "10",
                    f"{notch.position[0]:.4f}",
                    "20",
                    f"{notch.position[1]:.4f}",
                    "40",
                    "2.0",
                    "1",
                    str(notch.label),
                ])
        for fold in panel.folds:
            kind_layer = f"{panel.name}_FOLD_{fold.kind.upper()}"
            lines.extend([
                "0",
                "LINE",
                "8",
                kind_layer,
                "10",
                f"{fold.start[0]:.4f}",
                "20",
                f"{fold.start[1]:.4f}",
                "11",
                f"{fold.end[0]:.4f}",
                "21",
                f"{fold.end[1]:.4f}",
            ])
            for chevron in fold.chevrons():
                (cx0, cy0), (cx1, cy1) = chevron
                lines.extend([
                    "0",
                    "LINE",
                    "8",
                    kind_layer,
                    "10",
                    f"{cx0:.4f}",
                    "20",
                    f"{cy0:.4f}",
                    "11",
                    f"{cx1:.4f}",
                    "21",
                    f"{cy1:.4f}",
                ])
        if panel.label:
            xs = [point[0] for point in panel.outline] or [panel.label.position[0]]
            ys = [point[1] for point in panel.outline] or [panel.label.position[1]]
            extent = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            lines.extend([
                "0",
                "TEXT",
                "8",
                f"{panel.name}_LABEL",
                "10",
                f"{panel.label.position[0]:.4f}",
                "20",
                f"{panel.label.position[1]:.4f}",
                "40",
                f"{extent * 0.06:.4f}",
                "1",
                panel.label.text,
                "50",
                f"{panel.label.rotation:.2f}",
            ])
        if panel.seam_outline:
            lines.extend([
                "0",
                "LWPOLYLINE",
                "8",
                f"{panel.name}_SEAM",
                "90",
                str(len(panel.seam_outline)),
            ])
            for x, y in panel.seam_outline:
                lines.extend(["10", f"{x:.4f}", "20", f"{y:.4f}"])
            lines.extend(["43", f"{panel.seam_allowance:.4f}"])
        if panel.cut_outline:
            lines.extend([
                "0",
                "LWPOLYLINE",
                "8",
                f"{panel.name}_CUT",
                "90",
                str(len(panel.cut_outline)),
            ])
            for x, y in panel.cut_outline:
                lines.extend(["10", f"{x:.4f}", "20", f"{y:.4f}"])
            lines.extend(["43", f"{panel.seam_allowance:.4f}"])
    lines.extend(["0", "ENDSEC", "0", "EOF"])
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "Panel3D",
    "Panel2D",
    "PanelAnnotations",
    "GrainlineAnnotation",
    "NotchAnnotation",
    "FoldAnnotation",
    "LabelAnnotation",
    "build_panel_annotations",
    "CADBackend",
    "SimplePlaneProjectionBackend",
    "LSCMUnwrapBackend",
    "PatternExporter",
]


