"""Undersuit generation helpers with thermal weight integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

from modules.heating import (
    HeatingController,
    HeatingMeshPlan,
    HeatingMeshPlanner,
    LayerWattageConfig,
    PanelGeometry,
    ZoneBudget,
)
from suit.thermal_zones import DEFAULT_THERMAL_ZONE_SPEC, ThermalZoneSpec

from ..tools.thermal_brush import load_weights


@dataclass(frozen=True)
class UndersuitMesh:
    """Light-weight mesh container used for undersuit prototypes."""

    vertices: Tuple[Tuple[float, float, float], ...]
    faces: Tuple[Tuple[int, int, int], ...] = ()
    panels: Tuple[Mapping[str, object], ...] = ()

    def to_payload(self) -> Dict[str, object]:
        """Return a serialisable representation of the mesh."""

        return {
            "vertices": [list(vertex) for vertex in self.vertices],
            "faces": [list(face) for face in self.faces],
        }


@dataclass(slots=True)
class HeatingOutput:
    """Heating artefacts emitted alongside thermal zoning."""

    plan: HeatingMeshPlan
    zone_budgets: Mapping[str, ZoneBudget]
    controller: HeatingController

    def to_payload(self) -> Dict[str, object]:
        return {
            "plan": self.plan.to_payload(),
            "bom": self.plan.build_bom(),
            "zones": [budget.to_payload() for budget in self.zone_budgets.values()],
            "total_wattage": self.plan.total_wattage(),
            "controller": self.controller.describe(),
        }


@dataclass(slots=True)
class UndersuitGenerationResult:
    """Container describing the outputs of the undersuit pipeline."""

    mesh: UndersuitMesh
    thermal_map: Dict[str, object]
    heating: HeatingOutput | None = None

    def to_payload(self) -> Dict[str, object]:
        """Return a serialisable payload containing mesh and thermal data."""

        payload: Dict[str, object] = {
            "mesh": self.mesh.to_payload(),
            "thermal": self.thermal_map,
        }
        if self.heating is not None:
            payload["heating"] = self.heating.to_payload()
        return payload


@dataclass(slots=True)
class UndersuitPipeline:
    """Pipeline orchestrator for undersuit meshes with thermal control."""

    spec: ThermalZoneSpec = field(default_factory=lambda: DEFAULT_THERMAL_ZONE_SPEC)
    heating_planner: HeatingMeshPlanner | None = None
    heating_controller: HeatingController | None = None

    def __post_init__(self) -> None:
        if self.heating_planner is None:
            self.heating_planner = HeatingMeshPlanner.with_defaults()
        if self.heating_controller is None:
            zone_ids = [zone.identifier for zone in self.spec.zones]
            self.heating_controller = HeatingController.with_default_limits(zone_ids)

    def generate(
        self,
        mesh: UndersuitMesh,
        *,
        brush_path: Path | str | None = None,
        weight_overrides: Mapping[str, float] | None = None,
        panel_layout: Sequence[PanelGeometry | Mapping[str, object]] | None = None,
        layer_overrides: Sequence[LayerWattageConfig] | None = None,
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
        heating_output = None

        if self.heating_planner is not None and self.heating_controller is not None:
            panels = panel_layout or mesh.panels
            if panels:
                plan = self.heating_planner.plan(panels, layers=layer_overrides)
                zone_budgets = self.heating_controller.derive_zone_budgets(plan)
                heating_output = HeatingOutput(
                    plan=plan,
                    zone_budgets=zone_budgets,
                    controller=self.heating_controller,
                )

        return UndersuitGenerationResult(mesh=mesh, thermal_map=thermal_map, heating=heating_output)


__all__ = [
    "HeatingOutput",
    "UndersuitGenerationResult",
    "UndersuitMesh",
    "UndersuitPipeline",
]
