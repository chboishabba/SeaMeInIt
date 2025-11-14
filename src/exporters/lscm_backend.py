"""Least-Squares Conformal Mapping backend for undersuit panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

try:  # pragma: no cover - optional dependency shim
    import numpy as np
except ImportError:  # pragma: no cover - numpy optional
    np = None  # type: ignore[assignment]

from .patterns import CADBackend, Panel2D, Panel3D

FloatArray = "np.ndarray"


def _triangle_complex(vertices: FloatArray, i: int, j: int, k: int) -> tuple[complex, complex]:
    """Return complex edge vectors for triangle ``(i, j, k)``."""

    p_i = vertices[i]
    p_j = vertices[j]
    p_k = vertices[k]
    e1 = p_j - p_i
    e2 = p_k - p_i

    normal = np.cross(e1, e2)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        # Degenerate triangle: pick an arbitrary orthonormal basis.
        tangent = np.array([1.0, 0.0, 0.0])
        bitangent = np.array([0.0, 1.0, 0.0])
    else:
        tangent = e1 / float(np.linalg.norm(e1) or 1.0)
        bitangent = np.cross(normal, tangent)
        bitangent /= float(np.linalg.norm(bitangent) or 1.0)

    z1 = complex(float(np.dot(e1, tangent)), float(np.dot(e1, bitangent)))
    z2 = complex(float(np.dot(e2, tangent)), float(np.dot(e2, bitangent)))
    return z1, z2


def _choose_anchor_vertices(vertices: FloatArray) -> tuple[int, int]:
    """Select two vertices that are furthest apart to anchor the UV solve."""

    if len(vertices) < 2:
        return 0, 0

    max_distance = -1.0
    anchor_pair = (0, 1 if len(vertices) > 1 else 0)
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            distance = float(np.linalg.norm(vertices[i] - vertices[j]))
            if distance > max_distance:
                max_distance = distance
                anchor_pair = (i, j)
    return anchor_pair


def _mesh_area(vertices: FloatArray, faces: FloatArray) -> float:
    """Compute total surface area for a triangulated mesh."""

    area = 0.0
    for tri in faces:
        i, j, k = (int(tri[0]), int(tri[1]), int(tri[2]))
        v0 = vertices[i]
        v1 = vertices[j]
        v2 = vertices[k]
        area += 0.5 * float(np.linalg.norm(np.cross(v1 - v0, v2 - v0)))
    return area


def _triangle_area_2d(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))


def _mesh_area_2d(uv: FloatArray, faces: FloatArray) -> float:
    area = 0.0
    for tri in faces:
        i, j, k = (int(tri[0]), int(tri[1]), int(tri[2]))
        area += _triangle_area_2d(tuple(uv[i]), tuple(uv[j]), tuple(uv[k]))
    return area


def _polyline_length(points: Sequence[tuple[float, float]], *, closed: bool = False) -> float:
    if not points:
        return 0.0
    total = 0.0
    for start, end in zip(points, points[1:]):
        total += math.dist(start, end)
    if closed and len(points) > 2:
        total += math.dist(points[-1], points[0])
    return total


def _gather_loop_indices(loop: Mapping[str, Any], vertex_count: int) -> list[int]:
    indices = loop.get("indices")
    if not isinstance(indices, Sequence):
        return list(range(vertex_count))
    gathered: list[int] = []
    for value in indices:
        try:
            idx = int(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if 0 <= idx < vertex_count:
            gathered.append(idx)
    return gathered or list(range(vertex_count))


class LSCMConformalBackend(CADBackend):
    """Flatten panels using Least-Squares Conformal Mapping."""

    def __init__(self, *, max_iterations: int = 3, loop_tolerance: float = 0.02) -> None:
        if np is None:  # pragma: no cover - optional dependency guard
            raise ImportError("numpy is required for the LSCM backend")
        self.max_iterations = max(0, int(max_iterations))
        self.loop_tolerance = float(loop_tolerance)

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
            allowance = float(
                seam_lookup.get(panel.name, {}).get("seam_allowance", seam_allowance)
            )

            vertices = np.asarray(panel.vertices, dtype=float)
            faces = np.asarray(panel.faces, dtype=int)

            if len(vertices) < 3 or len(faces) == 0:
                outline = [(0.0, 0.0) for _ in panel.vertices]
                metadata: dict[str, Any] = {
                    "flattening": {"method": "lscm", "iterations": 0},
                    "warnings": ["Panel does not contain enough geometry for flattening."],
                    "requires_subdivision": True,
                }
                flattened.append(
                    Panel2D(
                        name=panel.name,
                        outline=outline,
                        seam_allowance=allowance,
                        metadata=metadata,
                    )
                )
                continue

            uv = self._solve_lscm(vertices, faces)
            uv *= float(scale)

            original_area = _mesh_area(vertices, faces)
            uv_area = _mesh_area_2d(uv, faces)
            stretch_ratio = uv_area / original_area if original_area > 1e-12 else 1.0

            seam_config = seam_lookup.get(panel.name, {})
            loop_targets = self._resolve_loop_targets(seam_config, len(vertices))
            (
                adjusted_uv,
                loop_metrics,
                warnings,
                requires_subdivision,
                iterations,
            ) = self._enforce_loop_targets(uv, loop_targets)

            perimeter_uv = _polyline_length([tuple(point) for point in adjusted_uv], closed=True)

            metadata = {
                "flattening": {
                    "method": "lscm",
                    "loop_iterations": iterations,
                },
                "loop_metrics": loop_metrics,
                "warnings": warnings,
                "requires_subdivision": requires_subdivision,
                "original_area": original_area,
                "uv_area": uv_area,
                "stretch_ratio": stretch_ratio,
                "perimeter_uv": perimeter_uv,
            }

            flattened.append(
                Panel2D(
                    name=panel.name,
                    outline=[(float(x), float(y)) for x, y in adjusted_uv],
                    seam_allowance=allowance,
                    metadata=metadata,
                )
            )

        return flattened

    def _solve_lscm(self, vertices: FloatArray, faces: FloatArray) -> FloatArray:
        """Solve the LSCM system for the provided vertices and faces."""

        vertex_count = len(vertices)
        anchors = _choose_anchor_vertices(vertices)
        i0, i1 = anchors

        rows: list[np.ndarray] = []
        rhs: list[float] = []

        for tri in faces:
            i, j, k = (int(tri[0]), int(tri[1]), int(tri[2]))
            z1, z2 = _triangle_complex(vertices, i, j, k)
            coeffs = {
                i: -z1 + z2,
                j: -z2,
                k: z1,
            }

            row_real = np.zeros(2 * vertex_count, dtype=float)
            row_imag = np.zeros(2 * vertex_count, dtype=float)

            for index, coeff in coeffs.items():
                a = coeff.real
                b = coeff.imag
                row_real[2 * index] += a
                row_real[2 * index + 1] += -b
                row_imag[2 * index] += b
                row_imag[2 * index + 1] += a

            rows.append(row_real)
            rows.append(row_imag)
            rhs.extend([0.0, 0.0])

        # Anchor constraints to fix the solution in place.
        for index, target in ((i0, (0.0, 0.0)), (i1, (1.0, 0.0))):
            row_u = np.zeros(2 * vertex_count, dtype=float)
            row_v = np.zeros(2 * vertex_count, dtype=float)
            row_u[2 * index] = 1.0
            row_v[2 * index + 1] = 1.0
            rows.append(row_u)
            rows.append(row_v)
            rhs.extend([target[0], target[1]])

        matrix = np.vstack(rows)
        solution, *_ = np.linalg.lstsq(matrix, np.array(rhs, dtype=float), rcond=None)

        uv = solution.reshape(vertex_count, 2)

        # Rescale to respect anchor edge length.
        edge_length = float(np.linalg.norm(vertices[i1] - vertices[i0])) or 1.0
        uv *= edge_length
        return uv

    def _resolve_loop_targets(
        self, seam_config: Mapping[str, Any], vertex_count: int
    ) -> dict[str, dict[str, Any]]:
        raw_targets = seam_config.get("loop_targets")
        targets: dict[str, dict[str, Any]] = {}
        if isinstance(raw_targets, Mapping):
            for name, loop in raw_targets.items():
                if not isinstance(loop, Mapping):
                    continue
                indices = _gather_loop_indices(loop, vertex_count)
                closed = bool(loop.get("closed", name != "length"))
                target_value = loop.get("target")
                tolerance = float(loop.get("tolerance", self.loop_tolerance))
                targets[name] = {
                    "indices": indices,
                    "closed": closed,
                    "target": float(target_value) if target_value is not None else None,
                    "tolerance": tolerance,
                }
        return targets

    def _enforce_loop_targets(
        self,
        uv: FloatArray,
        loop_targets: Mapping[str, Mapping[str, Any]],
    ) -> tuple[FloatArray, dict[str, dict[str, Any]], list[str], bool, int]:
        if not loop_targets:
            metrics, _, _ = self._compute_loop_metrics(uv, loop_targets)
            return uv, metrics, [], False, 0

        adjusted = np.array(uv, dtype=float)
        iterations = 0
        for iteration in range(self.max_iterations):
            metrics, all_within, _ = self._compute_loop_metrics(adjusted, loop_targets)
            if all_within:
                break

            circumference_ratio = self._loop_ratio("circumference", metrics, loop_targets)
            length_ratio = self._loop_ratio("length", metrics, loop_targets)
            if circumference_ratio is None and length_ratio is None:
                break

            scale_u = 1.0
            scale_v = 1.0
            if circumference_ratio is not None:
                scale = math.sqrt(max(circumference_ratio, 1e-6))
                scale_u *= scale
                scale_v *= scale
            if length_ratio is not None:
                scale_v *= max(length_ratio, 1e-6)

            adjusted[:, 0] *= scale_u
            adjusted[:, 1] *= scale_v
            iterations = iteration + 1

        metrics, all_within, warnings = self._compute_loop_metrics(adjusted, loop_targets)
        requires_subdivision = not all_within
        return adjusted, metrics, warnings, requires_subdivision, iterations

    def _loop_ratio(
        self,
        name: str,
        metrics: Mapping[str, Mapping[str, Any]],
        loop_targets: Mapping[str, Mapping[str, Any]],
    ) -> float | None:
        metric = metrics.get(name)
        target_info = loop_targets.get(name)
        if not metric or not target_info:
            return None
        target = target_info.get("target")
        length = metric.get("uv_length")
        if target in (None, 0.0) or length in (None, 0.0):
            return None
        return float(target) / float(length)

    def _compute_loop_metrics(
        self,
        uv: FloatArray,
        loop_targets: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], bool, list[str]]:
        metrics: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []
        within = True

        for name, loop in loop_targets.items():
            indices = loop.get("indices", [])
            closed = bool(loop.get("closed", name != "length"))
            points = [tuple(uv[idx]) for idx in indices]
            uv_length = _polyline_length(points, closed=closed)
            target = loop.get("target")
            tolerance = float(loop.get("tolerance", self.loop_tolerance))
            deviation = None
            meets_tolerance = True
            if target is not None:
                deviation = abs(uv_length - float(target))
                allowed = tolerance * max(abs(float(target)), 1e-8)
                meets_tolerance = deviation <= allowed
                if not meets_tolerance:
                    within = False
                    warnings.append(
                        f"Loop '{name}' deviates by {deviation:.4f} (allowed ±{allowed:.4f})."
                    )

            metrics[name] = {
                "indices": indices,
                "uv_length": uv_length,
                "target": target,
                "tolerance": tolerance,
                "deviation": deviation,
                "within_tolerance": meets_tolerance,
            }

        return metrics, within, warnings


__all__ = ["LSCMConformalBackend"]

