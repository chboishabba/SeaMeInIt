"""Tests for geodesic measurement loop extraction utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.suit.measurement_loops import solve_measurement_loops
from src.suit.undersuit_generator import UnderSuitGenerator


def _cylinder_mesh(
    radius: float,
    height: float,
    *,
    radial_segments: int = 32,
    height_segments: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * math.pi, radial_segments, endpoint=False)
    z_levels = np.linspace(-height / 2.0, height / 2.0, height_segments + 1)

    vertices: list[tuple[float, float, float]] = []
    for z in z_levels:
        for angle in angles:
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            vertices.append((x, y, z))

    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, -height / 2.0))
    top_center = len(vertices)
    vertices.append((0.0, 0.0, height / 2.0))

    def vid(level: int, seg: int) -> int:
        return level * radial_segments + (seg % radial_segments)

    faces: list[tuple[int, int, int]] = []
    for level in range(height_segments):
        for seg in range(radial_segments):
            a = vid(level, seg)
            b = vid(level, seg + 1)
            c = vid(level + 1, seg)
            d = vid(level + 1, seg + 1)
            faces.append((a, b, d))
            faces.append((a, d, c))

    for seg in range(radial_segments):
        a = vid(0, seg)
        b = vid(0, seg + 1)
        faces.append((bottom_center, b, a))

    for seg in range(radial_segments):
        a = vid(height_segments, seg)
        b = vid(height_segments, seg + 1)
        faces.append((top_center, a, b))

    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def test_measurement_loop_tracks_target_circumference() -> None:
    radius = 0.5
    height = 2.0
    vertices, faces = _cylinder_mesh(radius, height)
    target = 2.0 * math.pi * radius

    loops = solve_measurement_loops(
        vertices,
        faces,
        {"waist_circumference": target},
    )

    loop = loops["waist_circumference"]
    assert loop.vertices[0] == loop.vertices[-1]
    assert loop.relative_error <= 0.05
    assert loop.perimeter == pytest.approx(target, rel=0.05)


def test_generator_metadata_contains_measurement_loops() -> None:
    reference_circumference = 80.0
    radius = reference_circumference / (2.0 * math.pi)
    height = 3.0 * radius
    vertices, faces = _cylinder_mesh(radius, height)

    generator = UnderSuitGenerator()
    result = generator.generate(
        {"vertices": vertices, "faces": faces},
        measurements={"waist_circumference": reference_circumference},
    )

    metadata = result.metadata
    loops = metadata.get("measurement_loops", {})
    loop = loops["waist_circumference"]
    assert loop["vertices"][0] == loop["vertices"][-1]
    assert loop["perimeter"] == pytest.approx(reference_circumference, rel=0.05)

