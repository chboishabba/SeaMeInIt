"""Shared panel schema for undersuit workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SurfacePatch:
    """Reference to a connected subset of the source mesh."""

    mesh_id: str | None = None
    face_indices: tuple[int, ...] = field(default_factory=tuple)
    vertex_indices: tuple[int, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PanelBoundary3D:
    """Panel boundary defined on the 3D surface."""

    curve: tuple[tuple[float, float, float], ...] = field(default_factory=tuple)
    smoothness: str | None = None
    landmarks: dict[str, tuple[float, float, float]] = field(default_factory=dict)


@dataclass(slots=True)
class PanelBoundary2D:
    """Panel boundary defined in flattened 2D space."""

    curve: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    representation: str | None = None
    export_ready: bool = False


@dataclass(slots=True)
class PanelSeams:
    """Seam partner metadata for a panel boundary."""

    partners: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    correspondence: str | None = None
    allowance_policy: str | None = None


@dataclass(slots=True)
class PanelGrain:
    """Preferred material stretch direction in 3D/2D."""

    direction_3d: tuple[float, float, float] | None = None
    direction_2d: tuple[float, float] | None = None


@dataclass(slots=True)
class PanelBudgets:
    """Distortion and sewability budgets used for validation gates."""

    distortion_max: float | None = None
    curvature_min_radius: float | None = None
    turning_max_per_length: float | None = None
    min_feature_size: float | None = None


@dataclass(slots=True)
class PanelStatus:
    """Panel validation state."""

    sewable: bool = False
    reason: str | None = None


@dataclass(slots=True)
class Panel:
    """Top-level panel schema shared across segmentation and export."""

    panel_id: str
    surface_patch: SurfacePatch | None = None
    boundary_3d: PanelBoundary3D | None = None
    boundary_2d: PanelBoundary2D | None = None
    seams: PanelSeams | None = None
    grain: PanelGrain | None = None
    budgets: PanelBudgets | None = None
    status: PanelStatus | None = None


__all__ = [
    "Panel",
    "PanelBoundary2D",
    "PanelBoundary3D",
    "PanelBudgets",
    "PanelGrain",
    "PanelSeams",
    "PanelStatus",
    "SurfacePatch",
]
