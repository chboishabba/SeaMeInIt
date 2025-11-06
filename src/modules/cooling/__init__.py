"""Cooling module planners for undersuit thermal regulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from suit.thermal_zones import ThermalZoneSpec

__all__ = [
    "PCMPack",
    "TubeCircuit",
    "Pump",
    "RoutingNode",
    "RoutingEdge",
    "RoutingGraph",
    "CoolingPlan",
    "plan_cooling_layout",
]

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_PCM_POWER_DENSITY = 160.0  # Watts dissipated per kilogram of paraffin PCM
_MINIMUM_FLOW_L_MIN = 1.5


@dataclass(frozen=True, slots=True)
class PCMPack:
    """Latent heat storage pack paired to a thermal zone."""

    identifier: str
    zone_id: str
    capacity_watts: float
    melt_point_c: float
    mass_kg: float

    def to_manifest(self) -> Mapping[str, float | str]:
        return {
            "id": self.identifier,
            "zone": self.zone_id,
            "capacity_watts": self.capacity_watts,
            "melt_point_c": self.melt_point_c,
            "mass_kg": self.mass_kg,
        }


@dataclass(frozen=True, slots=True)
class TubeCircuit:
    """Closed loop cooling circuit routed through a thermal zone."""

    identifier: str
    zone_id: str
    medium: str
    flow_rate_l_min: float
    length_m: float
    inner_diameter_mm: float

    def to_manifest(self) -> Mapping[str, float | str]:
        return {
            "id": self.identifier,
            "zone": self.zone_id,
            "medium": self.medium,
            "flow_rate_l_min": self.flow_rate_l_min,
            "length_m": self.length_m,
            "inner_diameter_mm": self.inner_diameter_mm,
        }


@dataclass(frozen=True, slots=True)
class Pump:
    """Pump sized to drive the undersuit cooling loop."""

    identifier: str
    max_flow_l_min: float
    head_pressure_kpa: float
    nominal_power_w: float

    def to_manifest(self) -> Mapping[str, float | str]:
        return {
            "id": self.identifier,
            "max_flow_l_min": self.max_flow_l_min,
            "head_pressure_kpa": self.head_pressure_kpa,
            "nominal_power_w": self.nominal_power_w,
        }


@dataclass(frozen=True, slots=True)
class RoutingNode:
    """Graph node used to describe supply/return manifolds and zones."""

    identifier: str
    kind: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_manifest(self) -> Mapping[str, object]:
        return {
            "id": self.identifier,
            "kind": self.kind,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutingEdge:
    """Directed connection between two routing nodes."""

    source: str
    target: str
    length_m: float
    inner_diameter_mm: float
    circuit_id: str

    def to_manifest(self) -> Mapping[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "length_m": self.length_m,
            "inner_diameter_mm": self.inner_diameter_mm,
            "circuit": self.circuit_id,
        }


@dataclass(slots=True)
class RoutingGraph:
    """Directed multi-graph describing the cooling loop topology."""

    nodes: tuple[RoutingNode, ...]
    edges: tuple[RoutingEdge, ...]

    def to_manifest(self) -> Mapping[str, object]:
        return {
            "nodes": [node.to_manifest() for node in self.nodes],
            "edges": [edge.to_manifest() for edge in self.edges],
        }


@dataclass(slots=True)
class CoolingPlan:
    """Aggregated cooling layout for an undersuit."""

    spec: ThermalZoneSpec
    medium: str
    pcm_packs: tuple[PCMPack, ...]
    circuits: tuple[TubeCircuit, ...]
    pump: Pump
    routing: RoutingGraph

    def zone_pcm_capacity(self) -> Mapping[str, float]:
        totals: MutableMapping[str, float] = {zone.identifier: 0.0 for zone in self.spec.zones}
        for pack in self.pcm_packs:
            totals[pack.zone_id] = totals.get(pack.zone_id, 0.0) + pack.capacity_watts
        return dict(totals)

    def zone_flow_rate(self) -> Mapping[str, float]:
        totals: MutableMapping[str, float] = {zone.identifier: 0.0 for zone in self.spec.zones}
        for circuit in self.circuits:
            totals[circuit.zone_id] = totals.get(circuit.zone_id, 0.0) + circuit.flow_rate_l_min
        return dict(totals)

    def to_manifest(self) -> Mapping[str, object]:
        return {
            "spec": {
                "name": self.spec.name,
                "zones": [zone.identifier for zone in self.spec.zones],
            },
            "medium": self.medium,
            "pump": self.pump.to_manifest(),
            "pcm_packs": [pack.to_manifest() for pack in self.pcm_packs],
            "circuits": [circuit.to_manifest() for circuit in self.circuits],
            "routing": self.routing.to_manifest(),
            "zone_pcm_capacity": self.zone_pcm_capacity(),
            "zone_flow_rate": self.zone_flow_rate(),
        }


def plan_cooling_layout(
    spec: ThermalZoneSpec,
    vertices: Sequence[Sequence[float]] | np.ndarray | None,
    *,
    medium: str = "liquid",
    total_pcm_capacity: float | None = None,
    total_flow_rate_l_min: float | None = None,
) -> CoolingPlan:
    """Plan PCM packs and fluid routing for the provided thermal zones."""

    medium = medium.lower()
    if medium not in {"liquid", "pcm"}:
        raise ValueError("Cooling medium must be either 'liquid' or 'pcm'.")

    zone_loads = {zone.identifier: float(zone.default_heat_load) for zone in spec.zones}
    total_zone_load = sum(zone_loads.values())
    if total_zone_load <= 0.0:
        total_zone_load = float(len(zone_loads))
        zone_loads = {zone_id: 1.0 for zone_id in zone_loads}

    if total_pcm_capacity is None:
        total_pcm_capacity = total_zone_load * 1.2

    if total_flow_rate_l_min is None:
        if medium == "liquid":
            total_flow_rate_l_min = max(total_zone_load / 80.0, _MINIMUM_FLOW_L_MIN)
        else:
            total_flow_rate_l_min = 0.0

    body_span = _infer_body_span(vertices)

    pcm_packs = _build_pcm_packs(spec, zone_loads, total_zone_load, total_pcm_capacity)
    circuits = _build_circuits(
        spec,
        zone_loads,
        total_zone_load,
        total_flow_rate_l_min,
        medium,
        body_span,
    )
    pump = _select_pump(total_flow_rate_l_min, medium)
    routing = _build_routing_graph(spec, circuits)

    return CoolingPlan(
        spec=spec,
        medium=medium,
        pcm_packs=tuple(pcm_packs),
        circuits=tuple(circuits),
        pump=pump,
        routing=routing,
    )


def _build_pcm_packs(
    spec: ThermalZoneSpec,
    zone_loads: Mapping[str, float],
    total_load: float,
    total_pcm_capacity: float,
) -> Iterable[PCMPack]:
    packs = []
    for index, zone in enumerate(spec.zones, start=1):
        zone_load = zone_loads[zone.identifier]
        fraction = zone_load / total_load if total_load > 0 else 1.0 / len(spec.zones)
        capacity = total_pcm_capacity * fraction
        mass = capacity / _PCM_POWER_DENSITY
        packs.append(
            PCMPack(
                identifier=f"pcm_{index:02d}",
                zone_id=zone.identifier,
                capacity_watts=round(capacity, 3),
                melt_point_c=26.5,
                mass_kg=round(mass, 4),
            )
        )
    return packs


def _build_circuits(
    spec: ThermalZoneSpec,
    zone_loads: Mapping[str, float],
    total_load: float,
    total_flow: float,
    medium: str,
    body_span: np.ndarray,
) -> Iterable[TubeCircuit]:
    circuits = []
    for index, zone in enumerate(spec.zones, start=1):
        fraction = (
            zone_loads[zone.identifier] / total_load if total_load > 0 else 1.0 / len(spec.zones)
        )
        flow = total_flow * fraction
        length = _estimate_zone_path_length(zone.bounds, body_span)
        diameter = 6.0 if medium == "liquid" else 10.0
        circuits.append(
            TubeCircuit(
                identifier=f"loop_{index:02d}",
                zone_id=zone.identifier,
                medium=medium,
                flow_rate_l_min=round(flow, 3),
                length_m=round(length, 3),
                inner_diameter_mm=diameter,
            )
        )
    return circuits


def _select_pump(total_flow_rate_l_min: float, medium: str) -> Pump:
    if medium == "liquid":
        max_flow = (
            total_flow_rate_l_min * 1.15 if total_flow_rate_l_min > 0 else _MINIMUM_FLOW_L_MIN
        )
        head_pressure = 55.0
        power = 48.0
    else:
        max_flow = 0.0
        head_pressure = 0.0
        power = 0.0
    return Pump(
        identifier="loop_pump",
        max_flow_l_min=round(max_flow, 3),
        head_pressure_kpa=head_pressure,
        nominal_power_w=power,
    )


def _build_routing_graph(spec: ThermalZoneSpec, circuits: Iterable[TubeCircuit]) -> RoutingGraph:
    nodes = [
        RoutingNode("supply_manifold", "manifold", {"role": "supply"}),
        RoutingNode("return_manifold", "manifold", {"role": "return"}),
    ]
    nodes.extend(
        RoutingNode(f"zone_{zone.identifier}", "zone", {"zone": zone.identifier})
        for zone in spec.zones
    )

    edges: list[RoutingEdge] = []
    for circuit in circuits:
        zone_node = f"zone_{circuit.zone_id}"
        half_length = max(circuit.length_m / 2.0, 0.05)
        edges.append(
            RoutingEdge(
                source="supply_manifold",
                target=zone_node,
                length_m=round(half_length, 3),
                inner_diameter_mm=circuit.inner_diameter_mm,
                circuit_id=circuit.identifier,
            )
        )
        edges.append(
            RoutingEdge(
                source=zone_node,
                target="return_manifold",
                length_m=round(half_length, 3),
                inner_diameter_mm=circuit.inner_diameter_mm,
                circuit_id=circuit.identifier,
            )
        )

    return RoutingGraph(nodes=tuple(nodes), edges=tuple(edges))


def _infer_body_span(vertices: Sequence[Sequence[float]] | np.ndarray | None) -> np.ndarray:
    if vertices is None:
        return np.array([1.6, 0.5, 0.4])
    array = np.asarray(vertices, dtype=float)
    if array.size == 0:
        return np.array([1.6, 0.5, 0.4])
    minima = array.min(axis=0)
    maxima = array.max(axis=0)
    return np.maximum(maxima - minima, np.array([0.3, 0.3, 0.3]))


def _estimate_zone_path_length(
    bounds: Mapping[str, tuple[float, float]] | None, span: np.ndarray
) -> float:
    if not bounds:
        return float(span.mean())
    length = 0.0
    for axis, (lower, upper) in bounds.items():
        idx = _AXIS_INDEX.get(axis)
        if idx is None:
            continue
        length += span[idx] * max(upper - lower, 0.05)
    if length <= 0.0:
        length = float(span.mean())
    return length
