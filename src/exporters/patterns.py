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

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Panel3D":
        vertices = [tuple(map(float, vertex)) for vertex in payload.get("vertices", [])]
        faces = [tuple(int(idx) for idx in face) for face in payload.get("faces", [])]
        name = str(payload.get("name", "panel"))
        return cls(name=name, vertices=vertices, faces=faces)


@dataclass(slots=True)
class Panel2D:
    """Flattened 2D panel representation."""

    name: str
    outline: list[tuple[float, float]]
    seam_allowance: float
    metadata: dict[str, Any] = field(default_factory=dict)


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
            outline: list[tuple[float, float]]
            if not panel.vertices:
                outline = []
            elif np is not None:
                points = np.asarray(panel.vertices, dtype=float)
                centroid = points.mean(axis=0)
                shifted = points - centroid
                _, _, vh = np.linalg.svd(shifted, full_matrices=False)
                basis = vh[:2, :]
                projected = shifted @ basis.T
                outline = _cleanup_panel_outline(_ordered_outline(projected * scale))
            else:
                outline = _cleanup_panel_outline(
                    (float(vertex[0]) * scale, float(vertex[1]) * scale)
                    for vertex in panel.vertices
                )

            allowance = float(seam_lookup.get(panel.name, {}).get("seam_allowance", seam_allowance))
            metadata = {
                "original_vertex_count": len(panel.vertices),
                "seam_allowance": allowance,
            }
            flattened.append(
                Panel2D(
                    name=panel.name,
                    outline=outline,
                    seam_allowance=allowance,
                    metadata=metadata,
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
                outline = self._unwrap_panel(panel, scale=scale)
            except Exception:
                fallback = self._fallback.flatten_panels(
                    [panel],
                    seams,
                    scale=scale,
                    seam_allowance=seam_allowance,
                )
                flattened.extend(fallback)
                continue
            flattened.append(
                Panel2D(
                    name=panel.name,
                    outline=outline,
                    seam_allowance=allowance,
                    metadata=metadata,
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


def _polygon_centroid(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    if len(points) == 0:
        return (0.0, 0.0)
    twice_area = 0.0
    c_x = 0.0
    c_y = 0.0
    count = len(points)
    for idx in range(count):
        x0, y0 = points[idx]
        x1, y1 = points[(idx + 1) % count]
        cross = x0 * y1 - x1 * y0
        twice_area += cross
        c_x += (x0 + x1) * cross
        c_y += (y0 + y1) * cross
    if abs(twice_area) < 1e-9:
        avg_x = sum(point[0] for point in points) / count
        avg_y = sum(point[1] for point in points) / count
        return (avg_x, avg_y)
    factor = 1.0 / (3.0 * twice_area)
    return (c_x * factor, c_y * factor)


def _panel_bounds(panels: Sequence[Panel2D]) -> tuple[float, float, float, float]:
    min_x = min((point[0] for panel in panels for point in panel.outline), default=0.0)
    max_x = max((point[0] for panel in panels for point in panel.outline), default=1.0)
    min_y = min((point[1] for panel in panels for point in panel.outline), default=0.0)
    max_y = max((point[1] for panel in panels for point in panel.outline), default=1.0)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        max_y = min_y + 1.0
    return min_x, min_y, max_x, max_y


def _write_svg(path: Path, panels: Sequence[Panel2D], metadata: Mapping[str, Any]) -> None:
    min_x, min_y, max_x, max_y = _panel_bounds(panels)
    width = max_x - min_x
    height = max_y - min_y
    lines = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        f"<!-- Scale: {metadata['scale']} -->",
        f"<!-- Seam allowance: {metadata['seam_allowance']} -->",
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width:.2f}\" height=\"{height:.2f}\" viewBox=\"{min_x:.2f} {min_y:.2f} {width:.2f} {height:.2f}\">",
    ]
    if metadata.get("panel_warnings"):
        warnings_json = json.dumps(metadata["panel_warnings"], sort_keys=True)
        lines.insert(3, f"<!-- panel_warnings: {warnings_json} -->")
    for panel in panels:
        if not panel.outline:
            continue
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in panel.outline)
        lines.append(
            f"  <polygon id=\"{panel.name}\" points=\"{points}\" data-seam-allowance=\"{panel.seam_allowance}\" />"
        )
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
    lines.extend(["0", "ENDSEC", "0", "EOF"])
    path.write_text("\n".join(lines), encoding="utf-8")


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_pdf(path: Path, panels: Sequence[Panel2D], metadata: Mapping[str, Any]) -> None:
    min_x, min_y, max_x, max_y = _panel_bounds(panels)
    margin = 20.0
    width = max(100.0, (max_x - min_x) + margin * 2)
    height = max(100.0, (max_y - min_y) + margin * 2)

    lines = [
        "SMII Pattern Export",
        f"Scale: {metadata['scale']}",
        f"Seam allowance: {metadata['seam_allowance']}",
        f"Panels: {metadata['panel_count']}",
    ]
    for panel in panels:
        coords = ", ".join(f"{x:.2f},{y:.2f}" for x, y in panel.outline)
        lines.append(f"{panel.name}: {coords} (allowance={panel.seam_allowance})")

    text_ops = ["BT", "/F1 12 Tf", f"20 {height - 40:.2f} Td"]
    for index, line in enumerate(lines):
        escaped = _escape_pdf_text(line)
        if index == 0:
            text_ops.append(f"({escaped}) Tj")
        else:
            text_ops.append(f"0 -14 Td ({escaped}) Tj")
    text_ops.append("ET")
    content_stream = "\n".join(text_ops).encode("utf-8")

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


__all__ = [
    "Panel3D",
    "Panel2D",
    "CADBackend",
    "SimplePlaneProjectionBackend",
    "LSCMUnwrapBackend",
    "PatternExporter",
]
