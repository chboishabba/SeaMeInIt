"""Thermal zoning and control specifications for undersuits."""

from .thermal_zones import (
    DEFAULT_THERMAL_ZONE_SPEC,
    ThermalZone,
    ThermalZoneAssignment,
    ThermalZoneSpec,
)

__all__ = [
    "DEFAULT_THERMAL_ZONE_SPEC",
    "ThermalZone",
    "ThermalZoneAssignment",
    "ThermalZoneSpec",
"""Utilities for generating undersuit meshes from body models."""

from .undersuit_generator import (
    LAYER_REFERENCE_MEASUREMENTS,
    MeshLayer,
    UnderSuitGenerator,
    UnderSuitOptions,
    UnderSuitResult,
)

__all__ = [
    "LAYER_REFERENCE_MEASUREMENTS",
    "MeshLayer",
    "UnderSuitGenerator",
    "UnderSuitOptions",
    "UnderSuitResult",
]
