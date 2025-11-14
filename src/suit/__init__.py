"""Soft suit modelling package."""

from .material_model import (
    DirectionalElasticModulus,
    MaterialBlend,
    MaterialCatalog,
    MaterialLayer,
    MaterialStack,
    PressureComfortRatings,
    ThermalResistance,
)
from .body_axes import BodyAxes, fit_body_axes
from .thermal_zones import (
    DEFAULT_THERMAL_ZONE_SPEC,
    ThermalZone,
    ThermalZoneAssignment,
    ThermalZoneSpec,
)
from .undersuit_generator import (
    LAYER_REFERENCE_MEASUREMENTS,
    MeshLayer,
    UnderSuitGenerator,
    UnderSuitOptions,
    UnderSuitResult,
)

__all__ = [
    "DirectionalElasticModulus",
    "MaterialBlend",
    "MaterialCatalog",
    "MaterialLayer",
    "MaterialStack",
    "PressureComfortRatings",
    "ThermalResistance",
    "BodyAxes",
    "ThermalZone",
    "ThermalZoneAssignment",
    "ThermalZoneSpec",
    "DEFAULT_THERMAL_ZONE_SPEC",
    "LAYER_REFERENCE_MEASUREMENTS",
    "MeshLayer",
    "UnderSuitGenerator",
    "UnderSuitOptions",
    "UnderSuitResult",
    "fit_body_axes",
]
