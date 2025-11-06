"""Thermal zoning specification for undersuit meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

AxisBounds = Mapping[str, Tuple[float, float]]
_NORMAL_AXES = ("x", "y", "z")
_AXIS_INDEX = {axis: idx for idx, axis in enumerate(_NORMAL_AXES)}
_EPSILON = 1e-9


@dataclass(frozen=True)
class ThermalZone:
    """A contiguous undersuit region that can be thermally controlled."""

    identifier: str
    label: str
    description: str
    default_heat_load: float
    bounds: AxisBounds = field(default_factory=dict)
    fallback: bool = False

    def contains(self, point: Mapping[str, float]) -> bool:
        """Return ``True`` when the normalised coordinate lies within this zone."""

        for axis, (lower, upper) in self.bounds.items():
            value = point.get(axis)
            if value is None:
                return False
            if value < lower - _EPSILON or value > upper + _EPSILON:
                return False
        return True


@dataclass(slots=True)
class ThermalZoneAssignment:
    """Vertex coverage produced by applying a :class:`ThermalZoneSpec`."""

    spec: "ThermalZoneSpec"
    zone_vertices: Mapping[str, Tuple[int, ...]]

    def vertex_indices(self, zone_id: str) -> Tuple[int, ...]:
        """Return vertex indices belonging to the requested zone."""

        return self.zone_vertices.get(zone_id, ())

    def total_vertices(self) -> int:
        """Return the total number of assigned vertices."""

        return sum(len(indices) for indices in self.zone_vertices.values())

    def build_payload(self, weights: Mapping[str, float]) -> Dict[str, object]:
        """Generate a serialisable payload describing the thermal map."""

        zone_entries: List[Dict[str, object]] = []
        total_effective_load = 0.0
        for zone in self.spec.zones:
            vertices = list(self.vertex_indices(zone.identifier))
            weight = float(weights.get(zone.identifier, 1.0))
            weight = max(weight, 0.0)
            effective_load = zone.default_heat_load * weight
            total_effective_load += effective_load
            zone_entries.append(
                {
                    "id": zone.identifier,
                    "label": zone.label,
                    "description": zone.description,
                    "default_heat_load": zone.default_heat_load,
                    "priority_weight": weight,
                    "cooling_target": effective_load,
                    "vertex_indices": vertices,
                }
            )

        return {
            "spec": {
                "name": self.spec.name,
                "zone_count": len(self.spec.zones),
            },
            "zones": zone_entries,
            "total_effective_load": total_effective_load,
        }


@dataclass(frozen=True)
class ThermalZoneSpec:
    """Specification describing how to partition meshes into thermal zones."""

    name: str
    zones: Tuple[ThermalZone, ...]

    def assign_vertices(self, vertices: Sequence[Sequence[float]]) -> ThermalZoneAssignment:
        """Assign each vertex to the most appropriate thermal zone."""

        if not vertices:
            zone_vertices: Dict[str, Tuple[int, ...]] = {
                zone.identifier: () for zone in self.zones
            }
            return ThermalZoneAssignment(spec=self, zone_vertices=zone_vertices)

        minima = [min(vertex[idx] for vertex in vertices) for idx in range(3)]
        maxima = [max(vertex[idx] for vertex in vertices) for idx in range(3)]

        fallback_zone = next((zone for zone in self.zones if zone.fallback), None)
        zone_assignments: MutableMapping[str, List[int]] = {
            zone.identifier: [] for zone in self.zones
        }

        for index, vertex in enumerate(vertices):
            normalised = _normalise_vertex(vertex, minima, maxima)
            assigned = False
            for zone in self.zones:
                if zone.contains(normalised):
                    zone_assignments[zone.identifier].append(index)
                    assigned = True
                    break
            if not assigned and fallback_zone is not None:
                zone_assignments[fallback_zone.identifier].append(index)

        zone_vertices = {
            zone_id: tuple(indices) for zone_id, indices in zone_assignments.items()
        }
        return ThermalZoneAssignment(spec=self, zone_vertices=zone_vertices)

    def normalise_weights(self, weights: Mapping[str, float] | None = None) -> Dict[str, float]:
        """Return a copy of weights aligned with this specification."""

        resolved: Dict[str, float] = {}
        weights = weights or {}
        for zone in self.zones:
            weight = float(weights.get(zone.identifier, 1.0))
            if weight < 0.0:
                weight = 0.0
            resolved[zone.identifier] = weight

        if all(weight == 0.0 for weight in resolved.values()):
            resolved = {zone_id: 1.0 for zone_id in resolved}

        return resolved

    def build_allocation(
        self,
        vertices: Sequence[Sequence[float]],
        weights: Mapping[str, float] | None = None,
    ) -> Dict[str, object]:
        """Partition ``vertices`` and return a thermal allocation payload."""

        assignment = self.assign_vertices(vertices)
        resolved_weights = self.normalise_weights(weights)
        return assignment.build_payload(resolved_weights)


def _normalise_vertex(
    vertex: Sequence[float],
    minima: Sequence[float],
    maxima: Sequence[float],
) -> Dict[str, float]:
    """Normalise a vertex to the ``[0, 1]`` range along each axis."""

    normalised: Dict[str, float] = {}
    for axis in _NORMAL_AXES:
        idx = _AXIS_INDEX[axis]
        span = maxima[idx] - minima[idx]
        if span <= _EPSILON:
            normalised[axis] = 0.5
        else:
            normalised[axis] = (vertex[idx] - minima[idx]) / span
    return normalised


def _default_zones() -> Tuple[ThermalZone, ...]:
    """Return the canonical set of thermal zones for the undersuit."""

    return (
        ThermalZone(
            identifier="head_neck",
            label="Head & Neck",
            description="Cranial and cervical coverage up to the helmet seal.",
            default_heat_load=65.0,
            bounds={"z": (0.82, 1.0)},
        ),
        ThermalZone(
            identifier="upper_torso_front",
            label="Upper Torso (Front)",
            description="Pectoral and clavicular panels facing forward.",
            default_heat_load=120.0,
            bounds={"z": (0.55, 0.82), "y": (0.55, 1.0)},
        ),
        ThermalZone(
            identifier="upper_torso_back",
            label="Upper Torso (Back)",
            description="Scapular and upper back plating.",
            default_heat_load=115.0,
            bounds={"z": (0.55, 0.82), "y": (0.0, 0.45)},
        ),
        ThermalZone(
            identifier="core",
            label="Core",
            description="Abdominal section responsible for primary metabolic heat.",
            default_heat_load=140.0,
            bounds={"z": (0.42, 0.6)},
        ),
        ThermalZone(
            identifier="pelvis",
            label="Pelvis & Glutes",
            description="Hip interface panels wrapping the pelvis.",
            default_heat_load=95.0,
            bounds={"z": (0.32, 0.48)},
        ),
        ThermalZone(
            identifier="left_arm",
            label="Left Arm",
            description="Entire left arm from deltoid to wrist.",
            default_heat_load=80.0,
            bounds={"z": (0.48, 0.78), "x": (0.0, 0.44)},
        ),
        ThermalZone(
            identifier="right_arm",
            label="Right Arm",
            description="Entire right arm from deltoid to wrist.",
            default_heat_load=80.0,
            bounds={"z": (0.48, 0.78), "x": (0.56, 1.0)},
        ),
        ThermalZone(
            identifier="left_leg",
            label="Left Leg",
            description="Left leg from thigh through calf.",
            default_heat_load=110.0,
            bounds={"z": (0.0, 0.32), "x": (0.0, 0.48)},
        ),
        ThermalZone(
            identifier="right_leg",
            label="Right Leg",
            description="Right leg from thigh through calf.",
            default_heat_load=110.0,
            bounds={"z": (0.0, 0.32), "x": (0.52, 1.0)},
        ),
        ThermalZone(
            identifier="fallback",
            label="Fallback Coverage",
            description="Residual surfaces not captured by canonical zones.",
            default_heat_load=60.0,
            fallback=True,
        ),
    )


DEFAULT_THERMAL_ZONE_SPEC = ThermalZoneSpec(
    name="smii_default_undersuit",
    zones=_default_zones(),
)

__all__ = [
    "ThermalZone",
    "ThermalZoneAssignment",
    "ThermalZoneSpec",
    "DEFAULT_THERMAL_ZONE_SPEC",
]
