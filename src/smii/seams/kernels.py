"""Kernel construction for seam optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import numpy as np

from smii.rom.constraints import ConstraintRegistry
from smii.rom.seam_costs import SeamCostField
from smii.seams.fabric_kernels import FabricProfile, fabric_penalty, rotate_grain

try:  # Optional heavy import
    from suit.seam_generator import SeamGraph
except Exception:  # pragma: no cover
    SeamGraph = None  # type: ignore[assignment]

__all__ = [
    "EdgeKernel",
    "KernelWeights",
    "edge_energy",
    "build_edge_kernels",
]


@dataclass(frozen=True, slots=True)
class EdgeKernel:
    """Edge-local kernel terms derived from ROM and geometry."""

    rom_mean: float
    rom_max: float
    rom_grad: float
    curvature: float
    fabric_misalignment: float


@dataclass(frozen=True, slots=True)
class KernelWeights:
    """Weights applied to kernel terms."""

    rom_mean: float = 1.0
    rom_max: float = 1.0
    rom_grad: float = 1.0
    curvature: float = 1.0
    fabric: float = 1.0


def edge_energy(kernel: EdgeKernel, weights: KernelWeights) -> float:
    """Compute scalar edge energy as a dot product of kernel and weights."""

    return float(
        weights.rom_mean * kernel.rom_mean
        + weights.rom_max * kernel.rom_max
        + weights.rom_grad * kernel.rom_grad
        + weights.curvature * kernel.curvature
        + weights.fabric * kernel.fabric_misalignment
    )


def _unit(vector: np.ndarray | None) -> np.ndarray:
    if vector is None:
        return np.zeros(3, dtype=float)
    arr = np.asarray(vector, dtype=float).reshape(-1)
    norm = np.linalg.norm(arr)
    if norm <= 1e-12:
        return np.zeros_like(arr)
    return arr / norm


def _edge_length(edge: tuple[int, int], vertices: np.ndarray) -> float:
    a, b = edge
    if a >= len(vertices) or b >= len(vertices):
        return 0.0
    return float(np.linalg.norm(vertices[a] - vertices[b]))


def _clamp_rotation(rotation_deg: float, profile: FabricProfile | None) -> float:
    if profile is None or profile.constraints.max_grain_rotation_deg is None:
        return rotation_deg
    return float(np.clip(rotation_deg, -profile.constraints.max_grain_rotation_deg, profile.constraints.max_grain_rotation_deg))


def _seam_vertices(seam_graph: SeamGraph) -> set[int]:
    vertices: set[int] = set()
    for panel in seam_graph.panels:
        vertices.update(int(idx) for idx in panel.seam_vertices)
    return vertices


def build_edge_kernels(
    cost_field: SeamCostField,
    seam_graph: "SeamGraph",
    *,
    vertices: np.ndarray,
    curvature: np.ndarray | None = None,
    fabric_grain: np.ndarray | None = None,
    fabric_profile: FabricProfile | None = None,
    grain_rotation_deg: float = 0.0,
    constraints: ConstraintRegistry | None = None,
) -> Mapping[tuple[int, int], EdgeKernel]:
    """Construct kernel vectors for seam edges.

    Edges are filtered to seam vertices and forbidden vertices are skipped.
    """

    if SeamGraph is None:
        raise ImportError("suit.seam_generator.SeamGraph is required to build kernels.")

    seam_vertices = _seam_vertices(seam_graph)
    grain = rotate_grain(_unit(fabric_grain), _clamp_rotation(grain_rotation_deg, fabric_profile))
    curvature_arr = np.asarray(curvature, dtype=float) if curvature is not None else None

    kernels: MutableMapping[tuple[int, int], EdgeKernel] = {}
    for edge in cost_field.edges:
        a, b = int(edge[0]), int(edge[1])
        if a not in seam_vertices or b not in seam_vertices:
            continue
        if constraints and (constraints.is_vertex_forbidden(a) or constraints.is_vertex_forbidden(b)):
            continue
        rom_values = np.asarray(cost_field.vertex_costs, dtype=float)
        rom_pair = np.asarray([rom_values[a], rom_values[b]], dtype=float)
        rom_mean = float(np.nanmean(rom_pair))
        rom_max = float(np.nanmax(rom_pair))
        rom_grad = float(np.abs(rom_pair[0] - rom_pair[1]))
        curv_value = 0.0
        if curvature_arr is not None and len(curvature_arr) > max(a, b):
            curv_value = float(np.nanmean(curvature_arr[[a, b]]))

        edge_vec = np.asarray(vertices[b] - vertices[a], dtype=float)
        edge_unit = _unit(edge_vec)
        if np.allclose(edge_unit, 0.0) or np.allclose(grain, 0.0):
            misalignment = 0.0
        elif fabric_profile is not None:
            misalignment = fabric_penalty(edge_vec, grain, fabric_profile)
        else:
            misalignment = float(1.0 - abs(float(np.dot(edge_unit, grain))))

        kernel = EdgeKernel(
            rom_mean=np.nan_to_num(rom_mean, nan=0.0, posinf=0.0, neginf=0.0),
            rom_max=np.nan_to_num(rom_max, nan=0.0, posinf=0.0, neginf=0.0),
            rom_grad=np.nan_to_num(rom_grad, nan=0.0, posinf=0.0, neginf=0.0),
            curvature=np.nan_to_num(curv_value, nan=0.0, posinf=0.0, neginf=0.0),
            fabric_misalignment=np.nan_to_num(misalignment, nan=0.0, posinf=0.0, neginf=0.0),
        )
        kernels[(min(a, b), max(a, b))] = kernel

    return kernels
