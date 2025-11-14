"""Material catalog models used by the undersuit generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from schemas.validators import (
    SUIT_MATERIAL_SCHEMA_NAME,
    load_material_catalog as _load_material_catalog_payload,
    validate_material_catalog,
)

__all__ = [
    "DirectionalElasticModulus",
    "MaterialBlend",
    "MaterialCatalog",
    "MaterialLayer",
    "MaterialStack",
    "PressureComfortRatings",
    "ThermalResistance",
]


def _lerp(a: float, b: float, ratio: float) -> float:
    return (1.0 - ratio) * a + ratio * b


def _coerce_ratio(ratio: float) -> float:
    value = float(ratio)
    if not 0.0 <= value <= 1.0:
        raise ValueError("Interpolation ratio must be between 0 and 1 inclusive.")
    return value


@dataclass(frozen=True, slots=True)
class MaterialLayer:
    """Represents a single fabric layer within a stack."""

    material: str
    thickness_mm: float
    description: str | None = None
    density_kg_m3: float | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MaterialLayer":
        return cls(
            material=str(payload["material"]),
            thickness_mm=float(payload["thickness_mm"]),
            description=str(payload["description"])
            if payload.get("description") is not None
            else None,
            density_kg_m3=float(payload["density_kg_m3"])
            if payload.get("density_kg_m3") is not None
            else None,
        )

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "material": self.material,
            "thickness_mm": self.thickness_mm,
        }
        if self.description is not None:
            mapping["description"] = self.description
        if self.density_kg_m3 is not None:
            mapping["density_kg_m3"] = self.density_kg_m3
        return mapping


@dataclass(frozen=True, slots=True)
class DirectionalElasticModulus:
    """Directional elastic properties, typically measured in MPa."""

    warp: float
    weft: float
    bias: float | None = None
    unit: str = "MPa"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DirectionalElasticModulus":
        return cls(
            warp=float(payload["warp"]),
            weft=float(payload["weft"]),
            bias=float(payload["bias"]) if payload.get("bias") is not None else None,
            unit=str(payload.get("unit", "MPa")),
        )

    def interpolate(
        self, other: "DirectionalElasticModulus", ratio: float
    ) -> "DirectionalElasticModulus":
        ratio = _coerce_ratio(ratio)
        bias = None
        if self.bias is not None or other.bias is not None:
            bias = _lerp(self.bias or 0.0, other.bias or 0.0, ratio)
        unit = self.unit if self.unit else other.unit
        return DirectionalElasticModulus(
            warp=_lerp(self.warp, other.warp, ratio),
            weft=_lerp(self.weft, other.weft, ratio),
            bias=bias,
            unit=unit,
        )

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "warp": self.warp,
            "weft": self.weft,
            "unit": self.unit,
        }
        if self.bias is not None:
            mapping["bias"] = self.bias
        return mapping


@dataclass(frozen=True, slots=True)
class ThermalResistance:
    """Thermal resistance values for a material stack."""

    clo: float
    r_value: float | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ThermalResistance":
        return cls(
            clo=float(payload["clo"]),
            r_value=float(payload["r_value"]) if payload.get("r_value") is not None else None,
        )

    def interpolate(self, other: "ThermalResistance", ratio: float) -> "ThermalResistance":
        ratio = _coerce_ratio(ratio)
        r_value = None
        if self.r_value is not None or other.r_value is not None:
            r_value = _lerp(self.r_value or 0.0, other.r_value or 0.0, ratio)
        return ThermalResistance(clo=_lerp(self.clo, other.clo, ratio), r_value=r_value)

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {"clo": self.clo}
        if self.r_value is not None:
            mapping["r_value"] = self.r_value
        return mapping


@dataclass(frozen=True, slots=True)
class PressureComfortRatings:
    """Interface pressure comfort envelope for a material stack."""

    preferred_kpa: float
    max_kpa: float
    notes: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PressureComfortRatings":
        return cls(
            preferred_kpa=float(payload["preferred_kpa"]),
            max_kpa=float(payload["max_kpa"]),
            notes=str(payload["notes"]) if payload.get("notes") is not None else None,
        )

    def interpolate(
        self, other: "PressureComfortRatings", ratio: float
    ) -> "PressureComfortRatings":
        ratio = _coerce_ratio(ratio)
        notes = self.notes if self.notes else other.notes
        return PressureComfortRatings(
            preferred_kpa=_lerp(self.preferred_kpa, other.preferred_kpa, ratio),
            max_kpa=_lerp(self.max_kpa, other.max_kpa, ratio),
            notes=notes,
        )

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "preferred_kpa": self.preferred_kpa,
            "max_kpa": self.max_kpa,
        }
        if self.notes is not None:
            mapping["notes"] = self.notes
        return mapping


@dataclass(frozen=True, slots=True)
class MaterialStack:
    """A predefined layering strategy for a body panel."""

    identifier: str
    name: str
    layers: tuple[MaterialLayer, ...]
    elastic_modulus: DirectionalElasticModulus
    thermal_resistance: ThermalResistance
    pressure_comfort: PressureComfortRatings
    notes: str | None = None
    applications: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MaterialStack":
        layers = tuple(MaterialLayer.from_mapping(entry) for entry in payload["layers"])
        return cls(
            identifier=str(payload["id"]),
            name=str(payload["name"]),
            layers=layers,
            elastic_modulus=DirectionalElasticModulus.from_mapping(payload["elastic_modulus"]),
            thermal_resistance=ThermalResistance.from_mapping(payload["thermal_resistance"]),
            pressure_comfort=PressureComfortRatings.from_mapping(payload["pressure_comfort"]),
            notes=str(payload["notes"]) if payload.get("notes") is not None else None,
            applications=tuple(str(value) for value in payload.get("applications", [])),
        )

    @property
    def total_thickness_mm(self) -> float:
        return sum(layer.thickness_mm for layer in self.layers)

    def interpolate(
        self, other: "MaterialStack", ratio: float, *, name: str | None = None
    ) -> "MaterialBlend":
        ratio = _coerce_ratio(ratio)
        blend_name = name or f"{self.name} / {other.name}"
        return MaterialBlend(
            name=blend_name,
            primary_id=self.identifier,
            secondary_id=other.identifier,
            ratio=ratio,
            elastic_modulus=self.elastic_modulus.interpolate(other.elastic_modulus, ratio),
            thermal_resistance=self.thermal_resistance.interpolate(other.thermal_resistance, ratio),
            pressure_comfort=self.pressure_comfort.interpolate(other.pressure_comfort, ratio),
            total_thickness_mm=_lerp(self.total_thickness_mm, other.total_thickness_mm, ratio),
        )

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "kind": "stack",
            "id": self.identifier,
            "name": self.name,
            "layers": [layer.to_mapping() for layer in self.layers],
            "elastic_modulus": self.elastic_modulus.to_mapping(),
            "thermal_resistance": self.thermal_resistance.to_mapping(),
            "pressure_comfort": self.pressure_comfort.to_mapping(),
            "total_thickness_mm": self.total_thickness_mm,
        }
        if self.notes is not None:
            metadata["notes"] = self.notes
        if self.applications:
            metadata["applications"] = list(self.applications)
        return metadata


@dataclass(frozen=True, slots=True)
class MaterialBlend:
    """Represents an interpolated blend between two material stacks."""

    name: str
    primary_id: str
    secondary_id: str
    ratio: float
    elastic_modulus: DirectionalElasticModulus
    thermal_resistance: ThermalResistance
    pressure_comfort: PressureComfortRatings
    total_thickness_mm: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "kind": "blend",
            "name": self.name,
            "composition": {
                "primary": self.primary_id,
                "secondary": self.secondary_id,
                "ratio": self.ratio,
            },
            "elastic_modulus": self.elastic_modulus.to_mapping(),
            "thermal_resistance": self.thermal_resistance.to_mapping(),
            "pressure_comfort": self.pressure_comfort.to_mapping(),
            "total_thickness_mm": self.total_thickness_mm,
        }


@dataclass(slots=True)
class MaterialCatalog:
    """Container for undersuit material stacks and associated operations."""

    version: str
    description: str
    stacks: dict[str, MaterialStack]

    def __post_init__(self) -> None:
        self.stacks = dict(self.stacks)
        if not self.stacks:
            raise ValueError("Material catalog must define at least one stack.")

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        validate: bool = True,
        schema_name: str = SUIT_MATERIAL_SCHEMA_NAME,
    ) -> "MaterialCatalog":
        if validate:
            validate_material_catalog(payload, schema_name=schema_name)
        version = str(payload.get("version", ""))
        description = str(payload.get("description", ""))
        stacks_payload = payload.get("stacks", [])
        if not isinstance(stacks_payload, Iterable):
            raise TypeError("Catalog 'stacks' entry must be iterable.")
        stacks: dict[str, MaterialStack] = {}
        for entry in stacks_payload:
            if not isinstance(entry, Mapping):
                raise TypeError("Each stack entry must be a mapping.")
            stack = MaterialStack.from_mapping(entry)
            stacks[stack.identifier] = stack
        return cls(version=version, description=description, stacks=stacks)

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        schema_name: str = SUIT_MATERIAL_SCHEMA_NAME,
    ) -> "MaterialCatalog":
        payload = _load_material_catalog_payload(path)
        return cls.from_payload(payload, validate=False, schema_name=schema_name)

    def get_stack(self, stack_id: str) -> MaterialStack:
        try:
            return self.stacks[stack_id]
        except KeyError as exc:
            raise KeyError(f"Unknown material stack '{stack_id}'") from exc

    def blend(
        self, primary_id: str, secondary_id: str, ratio: float, *, name: str | None = None
    ) -> MaterialBlend:
        primary = self.get_stack(primary_id)
        secondary = self.get_stack(secondary_id)
        return primary.interpolate(secondary, ratio, name=name)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "description": self.description,
            "stacks": {stack_id: stack.to_metadata() for stack_id, stack in self.stacks.items()},
        }

    def panel_materials(self, assignments: Mapping[str, str]) -> dict[str, dict[str, Any]]:
        """Return metadata describing the stacks assigned to each panel."""

        return {
            panel: self.get_stack(stack_id).to_metadata() for panel, stack_id in assignments.items()
        }

    @property
    def stack_ids(self) -> Sequence[str]:
        return tuple(self.stacks.keys())
