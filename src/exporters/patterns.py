"""Utilities for exporting undersuit panels as 2D pattern files."""

from __future__ import annotations

import json
import math
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
                outline = _ordered_outline(projected * scale)
            else:
                outline = [
                    (float(vertex[0]) * scale, float(vertex[1]) * scale)
                    for vertex in panel.vertices
                ]

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
        return _ordered_outline(outline_points * float(scale))


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
