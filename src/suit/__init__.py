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
from .seam_generator import (
    MeasurementLoop,
    SeamGenerator,
    SeamGraph,
    SeamPanel,
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
    "MeasurementLoop",
    "SeamGenerator",
    "SeamGraph",
    "SeamPanel",
    "LAYER_REFERENCE_MEASUREMENTS",
    "MeshLayer",
    "UnderSuitGenerator",
    "UnderSuitOptions",
    "UnderSuitResult",
    "fit_body_axes",
]

_ATTRIBUTE_MODULES: dict[str, str] = {
    "DirectionalElasticModulus": ".material_model",
    "MaterialBlend": ".material_model",
    "MaterialCatalog": ".material_model",
    "MaterialLayer": ".material_model",
    "MaterialStack": ".material_model",
    "PressureComfortRatings": ".material_model",
    "ThermalResistance": ".material_model",
    "ThermalZone": ".thermal_zones",
    "ThermalZoneAssignment": ".thermal_zones",
    "ThermalZoneSpec": ".thermal_zones",
    "DEFAULT_THERMAL_ZONE_SPEC": ".thermal_zones",
    "LAYER_REFERENCE_MEASUREMENTS": ".undersuit_generator",
    "MeshLayer": ".undersuit_generator",
    "UnderSuitGenerator": ".undersuit_generator",
    "UnderSuitOptions": ".undersuit_generator",
    "UnderSuitResult": ".undersuit_generator",
}


def __getattr__(name: str):
    try:
        module_name = _ATTRIBUTE_MODULES[name]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise AttributeError(f"module 'src.suit' has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
