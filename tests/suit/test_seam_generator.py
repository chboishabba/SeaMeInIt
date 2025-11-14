from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pytest

from suit.seam_generator import MeasurementLoop, SeamGenerator


@pytest.fixture()
def seam_generator() -> SeamGenerator:
    return SeamGenerator(base_allowance=0.01, seam_bias=0.002, seam_tolerance=5e-3)


def _build_cylindrical_mesh(
    *,
    levels: Mapping[str, float],
    segments: int,
    radius_profile: Mapping[str, float],
    axis: str = "z",
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, ...]]]:
    level_names = list(levels.keys())
    level_positions = np.array(list(levels.values()), dtype=float)
    order = np.argsort(level_positions)
    sorted_levels = [level_names[idx] for idx in order]
    sorted_positions = level_positions[order]

    vertices: list[list[float]] = []
    loops: dict[str, tuple[int, ...]] = {}
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    other_axes = [0, 1, 2]
    other_axes.remove(axis_index)

    for level_index, level_name in enumerate(sorted_levels):
        radius = radius_profile[level_name]
        coord = sorted_positions[level_index]
        level_vertices: list[int] = []
        for segment in range(segments):
            theta = (2.0 * math.pi * segment) / segments
            point = [0.0, 0.0, 0.0]
            point[axis_index] = coord
            point[other_axes[0]] = radius * math.cos(theta)
            point[other_axes[1]] = radius * math.sin(theta)
            level_vertices.append(len(vertices))
            vertices.append(point)
        loops[level_name] = tuple(level_vertices)

    faces: list[list[int]] = []
    for lower_idx in range(len(sorted_levels) - 1):
        lower_loop = loops[sorted_levels[lower_idx]]
        upper_loop = loops[sorted_levels[lower_idx + 1]]
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a = lower_loop[segment]
            b = lower_loop[next_segment]
            c = upper_loop[segment]
            d = upper_loop[next_segment]
            faces.append([a, b, c])
            faces.append([b, d, c])

    vertex_array = np.asarray(vertices, dtype=float)
    face_array = np.asarray(faces, dtype=int)
    return vertex_array, face_array, loops


@pytest.fixture()
def human_mesh() -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, ...]], dict[str, np.ndarray]]:
    levels = {
        "neck_circumference": 0.8,
        "chest_circumference": 0.6,
        "waist_circumference": 0.4,
        "hip_circumference": 0.2,
    }
    radii = {
        "neck_circumference": 0.15,
        "chest_circumference": 0.25,
        "waist_circumference": 0.2,
        "hip_circumference": 0.23,
    }
    vertices, faces, loops = _build_cylindrical_mesh(
        levels=levels,
        segments=12,
        radius_profile=radii,
        axis="z",
    )
    axis_map = {
        "longitudinal": np.array([0.0, 0.0, 1.0], dtype=float),
        "lateral": np.array([1.0, 0.0, 0.0], dtype=float),
        "anterior": np.array([0.0, 1.0, 0.0], dtype=float),
    }
    return vertices, faces, loops, axis_map


@pytest.fixture()
def canine_mesh() -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, ...]], dict[str, np.ndarray]]:
    levels = {
        "neck_circumference": -0.5,
        "chest_circumference": 0.0,
        "girth_circumference": 0.45,
        "waist_circumference": 0.9,
    }
    radii = {
        "neck_circumference": 0.18,
        "chest_circumference": 0.24,
        "girth_circumference": 0.21,
        "waist_circumference": 0.19,
    }
    vertices, faces, loops = _build_cylindrical_mesh(
        levels=levels,
        segments=10,
        radius_profile=radii,
        axis="x",
    )
    axis_map = {
        "longitudinal": np.array([1.0, 0.0, 0.0], dtype=float),
        "lateral": np.array([0.0, 1.0, 0.0], dtype=float),
        "anterior": np.array([0.0, 0.0, 1.0], dtype=float),
    }
    return vertices, faces, loops, axis_map


def _is_topological_disk(vertices: np.ndarray, faces: np.ndarray) -> bool:
    if faces.size == 0:
        return False
    edges: dict[tuple[int, int], int] = {}
    for face in faces:
        tri = [int(idx) for idx in face]
        for start, end in ((0, 1), (1, 2), (2, 0)):
            a, b = tri[start], tri[end]
            edge = (min(a, b), max(a, b))
            edges[edge] = edges.get(edge, 0) + 1
    boundary_edges = sum(1 for count in edges.values() if count == 1)
    if boundary_edges == 0:
        return False
    euler = vertices.shape[0] - len(edges) + faces.shape[0]
    return euler == 1 and boundary_edges > 0


def test_human_panels_form_disks(human_mesh: tuple[np.ndarray, np.ndarray, dict[str, tuple[int, ...]], dict[str, np.ndarray]], seam_generator: SeamGenerator) -> None:
    vertices, faces, loops, axis_map = human_mesh
    loop_objects = []
    for name, indices in loops.items():
        coords = vertices[list(indices)] @ axis_map["longitudinal"]
        loop_objects.append(MeasurementLoop(name, tuple(indices), float(np.mean(coords))))
    seam_graph = seam_generator.generate(vertices, faces, loop_objects, axis_map)

    assert len(seam_graph.panels) == 2 * (len(loop_objects) - 1)
    metadata = seam_graph.seam_metadata
    for panel in seam_graph.panels:
        assert panel.anchor_loops[0] in loops
        assert panel.anchor_loops[1] in loops
        assert panel.side in {"left", "right"}
        assert panel.name in metadata
        seam_meta = metadata[panel.name]
        assert seam_meta["seam_allowance"] > 0.0
        loop_vertices = panel.loop_vertices()
        for anchor in panel.anchor_loops:
            assert loop_vertices[anchor], f"panel {panel.name} missing anchor loop {anchor}"
        assert _is_topological_disk(panel.vertices, panel.faces)
        seam_coords = vertices[list(panel.seam_vertices)] @ axis_map["lateral"]
        if panel.seam_vertices:
            assert np.all(np.abs(seam_coords) <= seam_generator.seam_tolerance + 1e-6)


def test_canine_axis_alignment(canine_mesh: tuple[np.ndarray, np.ndarray, dict[str, tuple[int, ...]], dict[str, np.ndarray]], seam_generator: SeamGenerator) -> None:
    vertices, faces, loops, axis_map = canine_mesh
    seam_graph = seam_generator.generate(vertices, faces, loops, axis_map)

    loop_order = [loop.name for loop in seam_graph.measurement_loops]
    assert loop_order[0] == "neck_circumference"
    assert loop_order[-1] == "waist_circumference"

    for panel in seam_graph.panels:
        seam_vertices = np.array(panel.seam_vertices, dtype=int)
        if seam_vertices.size == 0:
            continue
        coords = vertices[seam_vertices] @ axis_map["lateral"]
        assert np.all(np.abs(coords) <= seam_generator.seam_tolerance + 1e-6)
        allowance = seam_graph.seam_metadata[panel.name]["seam_allowance"]
        assert allowance >= seam_generator.base_allowance * 0.4

