"""Undersuit generation helpers with thermal weight integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Tuple

from suit.thermal_zones import DEFAULT_THERMAL_ZONE_SPEC, ThermalZoneSpec

from ..tools.thermal_brush import load_weights


@dataclass(frozen=True)
class UndersuitMesh:
    """Light-weight mesh container used for undersuit prototypes."""

    vertices: Tuple[Tuple[float, float, float], ...]
    faces: Tuple[Tuple[int, int, int], ...] = ()

    def to_payload(self) -> Dict[str, object]:
        """Return a serialisable representation of the mesh."""

        return {
            "vertices": [list(vertex) for vertex in self.vertices],
            "faces": [list(face) for face in self.faces],
        }


@dataclass(slots=True)
class UndersuitGenerationResult:
    """Container describing the outputs of the undersuit pipeline."""

    mesh: UndersuitMesh
    thermal_map: Dict[str, object]

    def to_payload(self) -> Dict[str, object]:
        """Return a serialisable payload containing mesh and thermal data."""

        return {
            "mesh": self.mesh.to_payload(),
            "thermal": self.thermal_map,
        }


@dataclass(slots=True)
class UndersuitPipeline:
    """Pipeline orchestrator for undersuit meshes with thermal control."""

    spec: ThermalZoneSpec = field(default_factory=lambda: DEFAULT_THERMAL_ZONE_SPEC)

    def generate(
        self,
        mesh: UndersuitMesh,
        *,
        brush_path: Path | str | None = None,
        weight_overrides: Mapping[str, float] | None = None,
    ) -> UndersuitGenerationResult:
        """Generate a payload bundling the mesh with thermal weights."""

        weights: Dict[str, float] = self.spec.normalise_weights()

        if brush_path is not None:
            brush_weights = load_weights(brush_path)
            weights.update(self.spec.normalise_weights(brush_weights))

        if weight_overrides:
            resolved_overrides = self.spec.normalise_weights(dict(weight_overrides))
            for zone_id in weight_overrides:
                if zone_id in weights:
                    weights[zone_id] = resolved_overrides[zone_id]

        thermal_map = self.spec.build_allocation(mesh.vertices, weights)
        return UndersuitGenerationResult(mesh=mesh, thermal_map=thermal_map)


__all__ = [
    "UndersuitGenerationResult",
    "UndersuitMesh",
    "UndersuitPipeline",
]
