"""Plan heater element placement across undersuit panels."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence


@dataclass(frozen=True)
class HeaterTemplate:
    """Reusable flexible heater definition."""

    identifier: str
    wattage: float
    coverage_area: float
    description: str = ""

    def __post_init__(self) -> None:
        if self.wattage <= 0.0:
            raise ValueError("Heater templates must define a positive wattage.")
        if self.coverage_area <= 0.0:
            raise ValueError("Heater templates must define a positive coverage area.")

    @property
    def power_density(self) -> float:
        """Return the delivered wattage per square metre."""

        return self.wattage / self.coverage_area


@dataclass(frozen=True)
class LayerWattageConfig:
    """Configuration for layering heater templates on a panel."""

    layer_id: str
    template_id: str
    coverage_ratio: float
    watt_density: float

    def __post_init__(self) -> None:
        if not self.layer_id:
            raise ValueError("Layer identifier must not be empty.")
        if not self.template_id:
            raise ValueError("Layer must reference a heater template.")
        if self.coverage_ratio < 0.0:
            raise ValueError("Coverage ratio cannot be negative.")
        if self.watt_density < 0.0:
            raise ValueError("Watt density cannot be negative.")


@dataclass(frozen=True)
class PanelGeometry:
    """Simplified panel record used for heating layout."""

    panel_id: str
    area: float
    zone: str | None = None

    def __post_init__(self) -> None:
        if not self.panel_id:
            raise ValueError("Panel geometry requires an identifier.")
        if self.area < 0.0:
            raise ValueError("Panel area cannot be negative.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "PanelGeometry":
        panel_id = str(payload.get("panel") or payload.get("name") or payload.get("id") or "panel")
        area = float(payload.get("area", 0.0))
        zone = payload.get("zone")
        zone_id = str(zone) if zone is not None else None
        return cls(panel_id=panel_id, area=area, zone=zone_id)


@dataclass(frozen=True)
class HeaterPlacement:
    """Result of planning heaters for a panel layer."""

    panel_id: str
    layer_id: str
    template_id: str
    element_count: int
    target_coverage_area: float
    coverage_area: float
    total_wattage: float

    def average_power_density(self) -> float:
        """Return the delivered watt density across the covered area."""

        if self.coverage_area <= 0.0:
            return 0.0
        return self.total_wattage / self.coverage_area

    def to_payload(self) -> Dict[str, float | int | str]:
        return {
            "panel": self.panel_id,
            "layer": self.layer_id,
            "template": self.template_id,
            "element_count": self.element_count,
            "target_coverage_area": self.target_coverage_area,
            "coverage_area": self.coverage_area,
            "total_wattage": self.total_wattage,
            "average_watt_density": self.average_power_density(),
        }


@dataclass(frozen=True)
class HeatingMeshPlan:
    """Collection of planned heater placements across panels."""

    panel_layers: Mapping[str, tuple[HeaterPlacement, ...]]
    panel_areas: Mapping[str, float]
    panel_zones: Mapping[str, str | None]
    templates: Mapping[str, HeaterTemplate]
    layers: Mapping[str, LayerWattageConfig]

    def total_wattage(self) -> float:
        return sum(
            placement.total_wattage
            for placements in self.panel_layers.values()
            for placement in placements
        )

    def panel_wattage(self, panel_id: str) -> float:
        return sum(placement.total_wattage for placement in self.panel_layers.get(panel_id, ()))

    def zone_wattage(self, zone_id: str) -> float:
        return sum(
            placement.total_wattage
            for panel, placements in self.panel_layers.items()
            if self.panel_zones.get(panel) == zone_id
            for placement in placements
        )

    def zone_area(self, zone_id: str) -> float:
        return sum(
            self.panel_areas.get(panel, 0.0)
            for panel, zone in self.panel_zones.items()
            if zone == zone_id
        )

    def build_bom(self) -> Dict[str, object]:
        template_totals: MutableMapping[str, MutableMapping[str, float | int]] = {}
        panel_entries: list[Mapping[str, object]] = []
        for panel_id, placements in self.panel_layers.items():
            panel_total = 0.0
            panel_payload: list[Mapping[str, object]] = []
            for placement in placements:
                panel_payload.append(placement.to_payload())
                panel_total += placement.total_wattage
                template_totals.setdefault(
                    placement.template_id,
                    {
                        "template": placement.template_id,
                        "element_count": 0,
                        "total_wattage": 0.0,
                        "coverage_area": 0.0,
                    },
                )
                template_totals[placement.template_id]["element_count"] = int(
                    template_totals[placement.template_id]["element_count"]
                ) + placement.element_count
                template_totals[placement.template_id]["total_wattage"] = float(
                    template_totals[placement.template_id]["total_wattage"]
                ) + placement.total_wattage
                template_totals[placement.template_id]["coverage_area"] = float(
                    template_totals[placement.template_id]["coverage_area"]
                ) + placement.coverage_area

            panel_entries.append(
                {
                    "panel": panel_id,
                    "zone": self.panel_zones.get(panel_id),
                    "total_wattage": panel_total,
                    "placements": panel_payload,
                }
            )

        for template_id, data in template_totals.items():
            template = self.templates.get(template_id)
            if template is not None:
                data["wattage_per_element"] = template.wattage
                data["coverage_area_per_element"] = template.coverage_area

        return {
            "templates": {key: dict(value) for key, value in template_totals.items()},
            "panels": panel_entries,
            "total_wattage": self.total_wattage(),
        }

    def to_payload(self) -> Dict[str, object]:
        return {
            "panels": {
                panel_id: [placement.to_payload() for placement in placements]
                for panel_id, placements in self.panel_layers.items()
            },
            "panel_zones": dict(self.panel_zones),
            "panel_areas": dict(self.panel_areas),
            "layers": {
                layer_id: {
                    "template": config.template_id,
                    "coverage_ratio": config.coverage_ratio,
                    "watt_density": config.watt_density,
                }
                for layer_id, config in self.layers.items()
            },
            "templates": {
                template_id: {
                    "wattage": template.wattage,
                    "coverage_area": template.coverage_area,
                    "description": template.description,
                }
                for template_id, template in self.templates.items()
            },
            "total_wattage": self.total_wattage(),
        }


@dataclass(slots=True)
class HeatingMeshPlanner:
    """Map heater templates across undersuit panels."""

    templates: Mapping[str, HeaterTemplate]
    default_layers: Sequence[LayerWattageConfig] = field(default_factory=tuple)

    @classmethod
    def with_defaults(cls) -> "HeatingMeshPlanner":
        return cls(DEFAULT_HEATER_TEMPLATES, DEFAULT_HEATER_LAYER_CONFIGS)

    def plan(
        self,
        panels: Sequence[PanelGeometry | Mapping[str, object]],
        *,
        layers: Sequence[LayerWattageConfig] | None = None,
    ) -> HeatingMeshPlan:
        if not panels:
            empty: Mapping[str, tuple[HeaterPlacement, ...]] = {}
            return HeatingMeshPlan(empty, {}, {}, self.templates, {})

        layer_sequence = tuple(layers or self.default_layers)
        layer_lookup = {layer.layer_id: layer for layer in layer_sequence}
        if not layer_lookup:
            empty: Mapping[str, tuple[HeaterPlacement, ...]] = {}
            panel_areas = {panel.panel_id: panel.area for panel in self._iter_panels(panels)}
            panel_zones = {panel.panel_id: panel.zone for panel in self._iter_panels(panels)}
            return HeatingMeshPlan(empty, panel_areas, panel_zones, self.templates, {})

        placements: MutableMapping[str, list[HeaterPlacement]] = {}
        panel_areas: Dict[str, float] = {}
        panel_zones: Dict[str, str | None] = {}

        for panel in self._iter_panels(panels):
            panel_areas[panel.panel_id] = panel.area
            panel_zones[panel.panel_id] = panel.zone
            panel_layers: list[HeaterPlacement] = []
            if panel.area <= 0.0:
                placements[panel.panel_id] = []
                continue

            for layer in layer_sequence:
                template = self.templates.get(layer.template_id)
                if template is None:
                    raise KeyError(f"Unknown heater template: {layer.template_id}")
                if layer.coverage_ratio <= 0.0 or layer.watt_density <= 0.0:
                    continue

                target_area = min(panel.area * layer.coverage_ratio, panel.area)
                if target_area <= 0.0:
                    continue

                required_wattage = target_area * layer.watt_density
                coverage_count = math.ceil(target_area / template.coverage_area)
                watt_count = math.ceil(required_wattage / template.wattage)
                element_count = max(coverage_count, watt_count)

                if element_count <= 0:
                    continue

                coverage_area = min(element_count * template.coverage_area, panel.area)
                total_wattage = element_count * template.wattage
                panel_layers.append(
                    HeaterPlacement(
                        panel_id=panel.panel_id,
                        layer_id=layer.layer_id,
                        template_id=layer.template_id,
                        element_count=element_count,
                        target_coverage_area=target_area,
                        coverage_area=coverage_area,
                        total_wattage=total_wattage,
                    )
                )

            placements[panel.panel_id] = panel_layers

        resolved_layers = {layer.layer_id: layer for layer in layer_sequence}
        resolved_plan = {
            panel_id: tuple(layer_list) for panel_id, layer_list in placements.items()
        }
        return HeatingMeshPlan(resolved_plan, panel_areas, panel_zones, self.templates, resolved_layers)

    @staticmethod
    def _iter_panels(
        panels: Sequence[PanelGeometry | Mapping[str, object]]
    ) -> Iterable[PanelGeometry]:
        for entry in panels:
            if isinstance(entry, PanelGeometry):
                yield entry
            else:
                yield PanelGeometry.from_mapping(entry)


DEFAULT_HEATER_TEMPLATES: Mapping[str, HeaterTemplate] = {
    "carbon_patch_large": HeaterTemplate(
        identifier="carbon_patch_large",
        wattage=24.0,
        coverage_area=0.036,
        description="Large carbon-fibre flexible heater (24W @ 12V)",
    ),
    "carbon_patch_small": HeaterTemplate(
        identifier="carbon_patch_small",
        wattage=12.0,
        coverage_area=0.018,
        description="Half-length carbon-fibre heater (12W @ 12V)",
    ),
}


DEFAULT_HEATER_LAYER_CONFIGS: tuple[LayerWattageConfig, ...] = (
    LayerWattageConfig(
        layer_id="primary",
        template_id="carbon_patch_large",
        coverage_ratio=0.45,
        watt_density=165.0,
    ),
    LayerWattageConfig(
        layer_id="booster",
        template_id="carbon_patch_small",
        coverage_ratio=0.2,
        watt_density=110.0,
    ),
)


__all__ = [
    "DEFAULT_HEATER_LAYER_CONFIGS",
    "DEFAULT_HEATER_TEMPLATES",
    "HeaterPlacement",
    "HeaterTemplate",
    "HeatingMeshPlan",
    "HeatingMeshPlanner",
    "LayerWattageConfig",
    "PanelGeometry",
]
