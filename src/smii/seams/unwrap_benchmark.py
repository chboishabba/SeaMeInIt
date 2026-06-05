"""Metric surfaces for comparing rectangle unwrap strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smii.seams.unwrap_backends import BOOTSTRAP_BACKEND, LSCM_BACKEND, unwrap_panel_vertices

GRAPH_ULTRAMETRIC_RECTANGLE = "graph_ultrametric_rectangle"
ORTHOGRAPHIC_SQUARE = "orthographic_square"
SPHERE_RECTANGLE_CANDIDATES = (
    GRAPH_ULTRAMETRIC_RECTANGLE,
    LSCM_BACKEND,
    BOOTSTRAP_BACKEND,
    ORTHOGRAPHIC_SQUARE,
)


@dataclass(frozen=True, slots=True)
class SphereMesh:
    """Synthetic sphere mesh with canonical longitude/latitude parameters."""

    vertices: np.ndarray
    faces: np.ndarray
    lon: np.ndarray
    lat: np.ndarray


@dataclass(frozen=True, slots=True)
class UnwrapMetricVector:
    """Comparable residuals for one unwrap candidate."""

    edge_length_residual: float
    area_residual: float
    angle_residual: float
    foldover_ratio: float
    agreement_depth: int
    agreement_distance: int
    aggregate_score: float


@dataclass(frozen=True, slots=True)
class UnwrapCandidateResult:
    """Benchmark result for one candidate rectangle unwrap."""

    strategy: str
    uv: np.ndarray
    metrics: UnwrapMetricVector


@dataclass(frozen=True, slots=True)
class SphereRectangleBenchmark:
    """Ranked comparison for sphere-to-rectangle unwrap candidates."""

    candidates: tuple[UnwrapCandidateResult, ...]

    @property
    def winner(self) -> UnwrapCandidateResult:
        return self.candidates[0]


def build_uv_sphere_mesh(longitude_steps: int = 32, latitude_steps: int = 16) -> SphereMesh:
    """Build a deterministic unit sphere mesh with duplicated seam/pole vertices."""

    if longitude_steps < 4:
        raise ValueError("longitude_steps must be at least 4.")
    if latitude_steps < 4:
        raise ValueError("latitude_steps must be at least 4.")
    lon_values = np.linspace(-np.pi, np.pi, longitude_steps + 1)
    lat_values = np.linspace(-np.pi / 2.0, np.pi / 2.0, latitude_steps + 1)
    vertices: list[tuple[float, float, float]] = []
    lon: list[float] = []
    lat: list[float] = []
    for phi in lat_values:
        cos_phi = float(np.cos(phi))
        for theta in lon_values:
            vertices.append(
                (
                    cos_phi * float(np.cos(theta)),
                    cos_phi * float(np.sin(theta)),
                    float(np.sin(phi)),
                )
            )
            lon.append(float(theta))
            lat.append(float(phi))
    row = longitude_steps + 1
    faces: list[tuple[int, int, int]] = []
    for j in range(latitude_steps):
        for i in range(longitude_steps):
            a = j * row + i
            b = a + 1
            c = (j + 1) * row + i
            d = c + 1
            faces.append((a, c, d))
            faces.append((a, d, b))
    return SphereMesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=int),
        lon=np.asarray(lon, dtype=float),
        lat=np.asarray(lat, dtype=float),
    )


def benchmark_sphere_rectangle_unwraps(
    *,
    longitude_steps: int = 32,
    latitude_steps: int = 16,
    strategies: tuple[str, ...] = SPHERE_RECTANGLE_CANDIDATES,
) -> SphereRectangleBenchmark:
    """Return ranked sphere-to-rectangle unwrap candidates.

    This is a comparison surface, not a claim that a sphere can be flattened
    isometrically into a rectangle. The graph/ultrametric candidate uses a
    declared cut graph at the longitude seam and scores residual agreement over
    edge, area, angle, and foldover metrics.
    """

    mesh = build_uv_sphere_mesh(longitude_steps=longitude_steps, latitude_steps=latitude_steps)
    results: list[UnwrapCandidateResult] = []
    for strategy in strategies:
        uv = _candidate_uv(mesh, strategy)
        results.append(
            UnwrapCandidateResult(
                strategy=strategy,
                uv=uv,
                metrics=_evaluate_uv(mesh.vertices, mesh.faces, uv),
            )
        )
    return SphereRectangleBenchmark(
        candidates=tuple(
            sorted(
                results,
                key=lambda result: (
                    result.metrics.aggregate_score,
                    result.metrics.agreement_distance,
                    result.metrics.foldover_ratio,
                    result.strategy,
                ),
            )
        )
    )


def _candidate_uv(mesh: SphereMesh, strategy: str) -> np.ndarray:
    if strategy == GRAPH_ULTRAMETRIC_RECTANGLE:
        u = mesh.lon / (2.0 * np.pi)
        v = mesh.lat / np.pi
        return np.column_stack([u, v])
    if strategy == ORTHOGRAPHIC_SQUARE:
        return mesh.vertices[:, :2].copy()
    if strategy in {BOOTSTRAP_BACKEND, LSCM_BACKEND}:
        return unwrap_panel_vertices(
            mesh.vertices,
            panel_vertices=tuple(range(len(mesh.vertices))),
            panel_faces=tuple(tuple(int(value) for value in face) for face in mesh.faces),
            method=strategy,
        )
    raise ValueError(f"Unknown unwrap strategy '{strategy}'.")


def _evaluate_uv(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> UnwrapMetricVector:
    valid_faces = [
        tuple(int(value) for value in face)
        for face in np.asarray(faces, dtype=int)
        if _triangle_area_3d(vertices[face]) > 1e-12
    ]
    edge_residual = _edge_length_residual(vertices, valid_faces, uv)
    scale = _best_fit_scale(vertices, valid_faces, uv)
    area_residual = _area_residual(vertices, valid_faces, uv, scale=scale)
    angle_residual = _angle_residual(vertices, valid_faces, uv)
    foldover_ratio = _foldover_ratio(valid_faces, uv)
    residuals = (
        edge_residual,
        area_residual,
        angle_residual,
        foldover_ratio,
    )
    agreement_depth = _agreement_depth(residuals)
    agreement_distance = len(residuals) - agreement_depth
    aggregate_score = (
        0.35 * edge_residual
        + 0.25 * area_residual
        + 0.25 * angle_residual
        + 0.15 * foldover_ratio
        + 0.05 * agreement_distance
    )
    return UnwrapMetricVector(
        edge_length_residual=float(edge_residual),
        area_residual=float(area_residual),
        angle_residual=float(angle_residual),
        foldover_ratio=float(foldover_ratio),
        agreement_depth=int(agreement_depth),
        agreement_distance=int(agreement_distance),
        aggregate_score=float(aggregate_score),
    )


def _unique_edges(faces: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    return sorted(edges)


def _best_fit_scale(
    vertices: np.ndarray, faces: list[tuple[int, int, int]], uv: np.ndarray
) -> float:
    numerator = 0.0
    denominator = 0.0
    for a, b in _unique_edges(faces):
        length_3d = float(np.linalg.norm(vertices[a] - vertices[b]))
        length_2d = float(np.linalg.norm(uv[a] - uv[b]))
        numerator += length_2d * length_3d
        denominator += length_2d * length_2d
    return numerator / denominator if denominator > 1e-12 else 1.0


def _edge_length_residual(
    vertices: np.ndarray, faces: list[tuple[int, int, int]], uv: np.ndarray
) -> float:
    scale = _best_fit_scale(vertices, faces, uv)
    residuals: list[float] = []
    for a, b in _unique_edges(faces):
        length_3d = float(np.linalg.norm(vertices[a] - vertices[b]))
        length_2d = scale * float(np.linalg.norm(uv[a] - uv[b]))
        if length_3d > 1e-12:
            residuals.append(abs(length_2d - length_3d) / length_3d)
    return float(np.mean(residuals)) if residuals else 1.0


def _area_residual(
    vertices: np.ndarray,
    faces: list[tuple[int, int, int]],
    uv: np.ndarray,
    *,
    scale: float,
) -> float:
    residuals: list[float] = []
    for face in faces:
        area_3d = _triangle_area_3d(vertices[list(face)])
        area_2d = (scale * scale) * abs(_signed_area_2d(uv[list(face)]))
        if area_3d > 1e-12:
            residuals.append(abs(area_2d - area_3d) / area_3d)
    return float(np.mean(residuals)) if residuals else 1.0


def _angle_residual(
    vertices: np.ndarray, faces: list[tuple[int, int, int]], uv: np.ndarray
) -> float:
    residuals: list[float] = []
    for face in faces:
        angles_3d = _triangle_angles(vertices[list(face)])
        angles_2d = _triangle_angles_2d(uv[list(face)])
        residuals.extend(abs(a - b) / np.pi for a, b in zip(angles_3d, angles_2d))
    return float(np.mean(residuals)) if residuals else 1.0


def _foldover_ratio(faces: list[tuple[int, int, int]], uv: np.ndarray) -> float:
    signs: list[int] = []
    for face in faces:
        area = _signed_area_2d(uv[list(face)])
        if abs(area) <= 1e-12:
            signs.append(0)
        else:
            signs.append(1 if area > 0.0 else -1)
    nonzero = [sign for sign in signs if sign != 0]
    if not signs:
        return 1.0
    if not nonzero:
        return 1.0
    positive = nonzero.count(1)
    negative = nonzero.count(-1)
    majority = 1 if positive >= negative else -1
    folded = sum(1 for sign in signs if sign == 0 or sign != majority)
    return float(folded / len(signs))


def _agreement_depth(residuals: tuple[float, float, float, float]) -> int:
    thresholds = (
        (0.24, 0.42),
        (0.38, 0.70),
        (0.16, 0.28),
        (0.01, 0.10),
    )
    depth = 0
    for residual, (promoted, diagnostic) in zip(residuals, thresholds):
        if residual <= promoted:
            depth += 1
            continue
        if residual <= diagnostic:
            return depth
        return depth
    return depth


def _triangle_area_3d(points: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])))


def _signed_area_2d(points: np.ndarray) -> float:
    return 0.5 * float(
        (points[1, 0] - points[0, 0]) * (points[2, 1] - points[0, 1])
        - (points[2, 0] - points[0, 0]) * (points[1, 1] - points[0, 1])
    )


def _triangle_angles(points: np.ndarray) -> tuple[float, float, float]:
    vectors = (
        (points[1] - points[0], points[2] - points[0]),
        (points[0] - points[1], points[2] - points[1]),
        (points[0] - points[2], points[1] - points[2]),
    )
    return tuple(_angle_between(a, b) for a, b in vectors)


def _triangle_angles_2d(points: np.ndarray) -> tuple[float, float, float]:
    lifted = np.column_stack([points, np.zeros(3, dtype=float)])
    return _triangle_angles(lifted)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


__all__ = [
    "GRAPH_ULTRAMETRIC_RECTANGLE",
    "ORTHOGRAPHIC_SQUARE",
    "SPHERE_RECTANGLE_CANDIDATES",
    "SphereMesh",
    "SphereRectangleBenchmark",
    "UnwrapCandidateResult",
    "UnwrapMetricVector",
    "benchmark_sphere_rectangle_unwraps",
    "build_uv_sphere_mesh",
]
