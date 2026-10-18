"""Typed data models for full-suit garment pattern generation and seam topology."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class BoundaryKind(StrEnum):
    SEAM = "seam"
    OPENING = "opening"
    HEM = "hem"
    FOLD = "fold"
    CUT = "cut"


class MarkKind(StrEnum):
    CUT = "cut"
    SEW = "sew"
    FOLD = "fold"
    DRILL = "drill"
    NOTCH = "notch"
    EASE = "ease"


@dataclass(frozen=True)
class PatternMark:
    kind: MarkKind
    points_mm: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        points = np.asarray(self.points_mm, dtype=float)
        expected = 1 if self.kind == MarkKind.DRILL else 2
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < expected:
            raise ValueError(f"{self.kind.value} mark requires {expected} 2D point(s)")
        if not np.isfinite(points).all():
            raise ValueError("Pattern mark coordinates must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "points_mm": [list(p) for p in self.points_mm]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PatternMark:
        return cls(
            kind=MarkKind(data["kind"]),
            points_mm=tuple(tuple(float(x) for x in p) for p in data["points_mm"]),
        )


@dataclass(frozen=True)
class PatternFeature:
    feature_id: str
    operator_id: str
    kind: str
    marks: tuple[PatternMark, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "operator_id": self.operator_id,
            "kind": self.kind,
            "marks": [mark.to_dict() for mark in self.marks],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PatternFeature:
        return cls(
            feature_id=str(data["feature_id"]),
            operator_id=str(data["operator_id"]),
            kind=str(data["kind"]),
            marks=tuple(PatternMark.from_dict(m) for m in data["marks"]),
        )


@dataclass(frozen=True)
class GarmentRegion:
    """Anatomical or functional garment region partitioning the source mesh."""

    region_id: str
    source_face_ids: IntArray
    material_id: str
    orientation_hint: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "source_face_ids": self.source_face_ids.tolist(),
            "material_id": self.material_id,
            "orientation_hint": list(self.orientation_hint),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GarmentRegion:
        return cls(
            region_id=str(data["region_id"]),
            source_face_ids=np.asarray(data["source_face_ids"], dtype=np.int64),
            material_id=str(data["material_id"]),
            orientation_hint=tuple(float(x) for x in data["orientation_hint"]),
        )


@dataclass(frozen=True)
class SeamSide:
    """One side of a sewable seam belonging to a specific pattern chart."""

    chart_id: str
    source_edge_ids: tuple[int, ...]
    boundary_vertex_ids: tuple[int, ...]
    reversed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_id": self.chart_id,
            "source_edge_ids": list(self.source_edge_ids),
            "boundary_vertex_ids": list(self.boundary_vertex_ids),
            "reversed": self.reversed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SeamSide:
        return cls(
            chart_id=str(data["chart_id"]),
            source_edge_ids=tuple(int(x) for x in data["source_edge_ids"]),
            boundary_vertex_ids=tuple(int(x) for x in data["boundary_vertex_ids"]),
            reversed=bool(data["reversed"]),
        )


@dataclass(frozen=True)
class SeamConnection:
    """Sewable connection between two pattern chart boundaries."""

    seam_id: str
    left: SeamSide
    right: SeamSide
    allowance_mm: float
    seam_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seam_id": self.seam_id,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "allowance_mm": self.allowance_mm,
            "seam_type": self.seam_type,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SeamConnection:
        return cls(
            seam_id=str(data["seam_id"]),
            left=SeamSide.from_dict(data["left"]),
            right=SeamSide.from_dict(data["right"]),
            allowance_mm=float(data["allowance_mm"]),
            seam_type=str(data["seam_type"]),
        )


@dataclass(frozen=True)
class PatternChart:
    """A flattened 2D pattern piece with physical scale (mm) and source face correspondence."""

    chart_id: str
    region_id: str
    source_face_ids: IntArray
    uv_mm: FloatArray
    local_faces: IntArray
    boundary_loops: tuple[IntArray, ...]
    source_vertex_ids: IntArray
    features: tuple[PatternFeature, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_id": self.chart_id,
            "region_id": self.region_id,
            "source_face_ids": self.source_face_ids.tolist(),
            "uv_mm": self.uv_mm.tolist(),
            "local_faces": self.local_faces.tolist(),
            "boundary_loops": [loop.tolist() for loop in self.boundary_loops],
            "source_vertex_ids": self.source_vertex_ids.tolist(),
            "features": [feature.to_dict() for feature in self.features],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PatternChart:
        return cls(
            chart_id=str(data["chart_id"]),
            region_id=str(data["region_id"]),
            source_face_ids=np.asarray(data["source_face_ids"], dtype=np.int64),
            uv_mm=np.asarray(data["uv_mm"], dtype=np.float64),
            local_faces=np.asarray(data["local_faces"], dtype=np.int64),
            boundary_loops=tuple(
                np.asarray(loop, dtype=np.int64) for loop in data["boundary_loops"]
            ),
            source_vertex_ids=np.asarray(data["source_vertex_ids"], dtype=np.int64),
            features=tuple(PatternFeature.from_dict(x) for x in data.get("features", [])),
        )


@dataclass(frozen=True)
class FullSuitPattern:
    """Complete full-suit garment pattern containing regions, charts, seams, and boundaries."""

    source_mesh_sha256: str
    regions: tuple[GarmentRegion, ...]
    charts: tuple[PatternChart, ...]
    seams: tuple[SeamConnection, ...]
    boundary_kinds: Mapping[str, BoundaryKind]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_mesh_sha256": self.source_mesh_sha256,
            "regions": [r.to_dict() for r in self.regions],
            "charts": [c.to_dict() for c in self.charts],
            "seams": [s.to_dict() for s in self.seams],
            "boundary_kinds": {k: str(v) for k, v in self.boundary_kinds.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FullSuitPattern:
        return cls(
            source_mesh_sha256=str(data["source_mesh_sha256"]),
            regions=tuple(GarmentRegion.from_dict(r) for r in data["regions"]),
            charts=tuple(PatternChart.from_dict(c) for c in data["charts"]),
            seams=tuple(SeamConnection.from_dict(s) for s in data["seams"]),
            boundary_kinds={k: BoundaryKind(v) for k, v in data["boundary_kinds"].items()},
        )


__all__ = [
    "BoundaryKind",
    "FloatArray",
    "FullSuitPattern",
    "GarmentRegion",
    "IntArray",
    "MarkKind",
    "PatternFeature",
    "PatternMark",
    "PatternChart",
    "SeamConnection",
    "SeamSide",
]
