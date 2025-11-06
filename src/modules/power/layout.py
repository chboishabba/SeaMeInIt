"""Power distribution layout planner for suit modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from ...schemas.validators import (
    POWER_MODULE_SCHEMA_NAME,
    SchemaValidationError,
    validate_power_catalog,
)

__all__ = [
    "BatteryLoadSummary",
    "ConnectorAllocation",
    "ModuleAllocation",
    "ModuleRequest",
    "PowerLayoutPlanner",
    "PowerLayoutResult",
]


@dataclass(frozen=True, slots=True)
class VoltageDomain:
    """Describes an electrical voltage domain available in the suit."""

    identifier: str
    nominal_voltage: float
    min_voltage: float
    max_voltage: float
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class BatteryPack:
    """Represents an energy storage module feeding one or more connectors."""

    identifier: str
    voltage_domain: str
    capacity_wh: float
    max_continuous_w: float
    connectors: tuple[str, ...]
    redundancy_group: str | None = None


@dataclass(frozen=True, slots=True)
class PowerConnector:
    """Represents a physical power connector on the suit."""

    identifier: str
    voltage_domain: str
    battery_pack_id: str
    max_power_w: float
    compatible_modules: tuple[str, ...]
    name: str | None = None
    max_current_a: float | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModuleProfile:
    """Catalog entry describing the nominal behaviour of a powered module."""

    identifier: str
    module_type: str
    voltage_domain: str
    nominal_power_w: float
    peak_power_w: float | None = None
    redundancy_paths: int = 1


@dataclass(frozen=True, slots=True)
class CompatibilityRule:
    """Defines suit-level compatibility guidance for a module type."""

    module_type: str
    allowed_connectors: tuple[str, ...]
    minimum_paths: int
    requires_redundancy: bool


@dataclass(frozen=True, slots=True)
class ModuleRequest:
    """Request for powering a module instance within the suit."""

    module_id: str
    module_type: str
    profile_id: str | None = None
    power_w: float | None = None
    voltage_domain: str | None = None
    redundancy_paths: int | None = None
    critical: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ModuleRequest":
        module_id = payload.get("module_id") or payload.get("id")
        if module_id is None:
            raise KeyError("Module request must define a 'module_id'.")
        module_type = payload.get("module_type")
        if module_type is None:
            raise KeyError("Module request must define a 'module_type'.")
        profile_id = payload.get("profile_id")
        power_w = payload.get("power_w")
        voltage_domain = payload.get("voltage_domain")
        redundancy_paths = payload.get("redundancy_paths") or payload.get("redundancy")
        critical = bool(payload.get("critical", False))
        return cls(
            module_id=str(module_id),
            module_type=str(module_type),
            profile_id=str(profile_id) if profile_id is not None else None,
            power_w=float(power_w) if power_w is not None else None,
            voltage_domain=str(voltage_domain) if voltage_domain is not None else None,
            redundancy_paths=int(redundancy_paths) if redundancy_paths is not None else None,
            critical=critical,
        )


@dataclass(frozen=True, slots=True)
class ConnectorAllocation:
    """Represents a routed connection between a module and a connector."""

    connector_id: str
    battery_pack_id: str
    voltage_domain: str
    allocated_power_w: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "battery_pack_id": self.battery_pack_id,
            "voltage_domain": self.voltage_domain,
            "allocated_power_w": self.allocated_power_w,
        }


@dataclass(frozen=True, slots=True)
class ModuleAllocation:
    """Allocation result for a module with its redundant paths."""

    module_id: str
    module_type: str
    voltage_domain: str
    power_w: float
    connections: tuple[ConnectorAllocation, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "module_type": self.module_type,
            "voltage_domain": self.voltage_domain,
            "power_w": self.power_w,
            "connections": [connection.to_mapping() for connection in self.connections],
        }


@dataclass(frozen=True, slots=True)
class BatteryLoadSummary:
    """Summarizes the aggregate loading on a battery pack."""

    battery_pack_id: str
    voltage_domain: str
    max_output_w: float
    allocated_power_w: float

    def remaining_power_w(self) -> float:
        return self.max_output_w - self.allocated_power_w

    def to_mapping(self) -> dict[str, Any]:
        return {
            "battery_pack_id": self.battery_pack_id,
            "voltage_domain": self.voltage_domain,
            "max_output_w": self.max_output_w,
            "allocated_power_w": self.allocated_power_w,
            "remaining_power_w": self.remaining_power_w(),
        }


@dataclass(frozen=True, slots=True)
class PowerLayoutResult:
    """Aggregate result of a power routing planning session."""

    assignments: tuple[ModuleAllocation, ...]
    battery_summaries: tuple[BatteryLoadSummary, ...]
    harness_diagram: tuple[Mapping[str, Any], ...]
    interface_manifest: Mapping[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "assignments": [allocation.to_mapping() for allocation in self.assignments],
            "battery_summary": [summary.to_mapping() for summary in self.battery_summaries],
            "harness_diagram": list(self.harness_diagram),
            "interface_manifest": dict(self.interface_manifest),
        }


class PowerLayoutPlanner:
    """Plan redundant power routes for heating and cooling modules."""

    def __init__(
        self,
        catalog: Mapping[str, Any],
        *,
        schema_name: str = POWER_MODULE_SCHEMA_NAME,
    ) -> None:
        try:
            validate_power_catalog(catalog, schema_name=schema_name)
        except SchemaValidationError as exc:  # pragma: no cover - defensive
            raise ValueError("Invalid power catalog supplied.") from exc

        self._catalog: dict[str, Any] = dict(catalog)
        self._domains: dict[str, VoltageDomain] = {}
        self._battery_packs: dict[str, BatteryPack] = {}
        self._connectors: dict[str, PowerConnector] = {}
        self._module_profiles: dict[str, ModuleProfile] = {}
        self._compatibility: dict[str, CompatibilityRule] = {}

        self._parse_domains()
        self._parse_battery_packs()
        self._parse_connectors()
        self._parse_module_profiles()
        self._parse_rules()

    def _parse_domains(self) -> None:
        for entry in self._catalog.get("voltage_domains", []):
            domain = VoltageDomain(
                identifier=str(entry["id"]),
                nominal_voltage=float(entry["nominal_voltage"]),
                min_voltage=float(entry["min_voltage"]),
                max_voltage=float(entry["max_voltage"]),
                notes=str(entry.get("notes")) if entry.get("notes") is not None else None,
            )
            self._domains[domain.identifier] = domain

    def _parse_battery_packs(self) -> None:
        for entry in self._catalog.get("battery_packs", []):
            connectors = tuple(str(identifier) for identifier in entry.get("connectors", []))
            pack = BatteryPack(
                identifier=str(entry["id"]),
                voltage_domain=str(entry["voltage_domain"]),
                capacity_wh=float(entry["capacity_wh"]),
                max_continuous_w=float(entry["max_continuous_w"]),
                connectors=connectors,
                redundancy_group=str(entry.get("redundancy_group"))
                if entry.get("redundancy_group")
                else None,
            )
            self._battery_packs[pack.identifier] = pack

    def _parse_connectors(self) -> None:
        for entry in self._catalog.get("connectors", []):
            connector = PowerConnector(
                identifier=str(entry["id"]),
                voltage_domain=str(entry["voltage_domain"]),
                battery_pack_id=str(entry["battery_pack"]),
                max_power_w=float(entry["max_power_w"]),
                compatible_modules=tuple(
                    str(value) for value in entry.get("compatible_modules", [])
                ),
                name=str(entry.get("name")) if entry.get("name") else None,
                max_current_a=float(entry.get("max_current_a"))
                if entry.get("max_current_a") is not None
                else None,
                tags=tuple(str(tag) for tag in entry.get("tags", [])),
            )
            if connector.voltage_domain not in self._domains:
                raise KeyError(
                    f"Connector '{connector.identifier}' references unknown voltage domain '{connector.voltage_domain}'."
                )
            if connector.battery_pack_id not in self._battery_packs:
                raise KeyError(
                    f"Connector '{connector.identifier}' references unknown battery pack '{connector.battery_pack_id}'."
                )
            pack = self._battery_packs[connector.battery_pack_id]
            if pack.voltage_domain != connector.voltage_domain:
                raise ValueError(
                    f"Connector '{connector.identifier}' voltage domain '{connector.voltage_domain}' does not match "
                    f"battery pack '{pack.identifier}' domain '{pack.voltage_domain}'."
                )
            self._connectors[connector.identifier] = connector

    def _parse_module_profiles(self) -> None:
        for entry in self._catalog.get("module_profiles", []):
            profile = ModuleProfile(
                identifier=str(entry["id"]),
                module_type=str(entry["module_type"]),
                voltage_domain=str(entry["voltage_domain"]),
                nominal_power_w=float(entry["nominal_power_w"]),
                peak_power_w=float(entry.get("peak_power_w"))
                if entry.get("peak_power_w") is not None
                else None,
                redundancy_paths=int(entry.get("redundancy_paths", 1)),
            )
            self._module_profiles[profile.identifier] = profile

    def _parse_rules(self) -> None:
        for entry in self._catalog.get("compatibility_rules", []):
            rule = CompatibilityRule(
                module_type=str(entry["module_type"]),
                allowed_connectors=tuple(
                    str(connector) for connector in entry.get("allowed_connectors", [])
                ),
                minimum_paths=int(entry.get("minimum_paths", 1)),
                requires_redundancy=bool(entry.get("requires_redundancy", False)),
            )
            self._compatibility[rule.module_type] = rule

    @property
    def connectors(self) -> Mapping[str, PowerConnector]:
        return self._connectors

    @property
    def battery_packs(self) -> Mapping[str, BatteryPack]:
        return self._battery_packs

    @property
    def module_profiles(self) -> Mapping[str, ModuleProfile]:
        return self._module_profiles

    def allocate(
        self, module_requests: Sequence[ModuleRequest | Mapping[str, Any]]
    ) -> PowerLayoutResult:
        """Allocate connectors and battery packs to serve *module_requests*."""

        requests = [
            request if isinstance(request, ModuleRequest) else ModuleRequest.from_mapping(request)
            for request in module_requests
        ]
        if not requests:
            return PowerLayoutResult(
                assignments=(),
                battery_summaries=tuple(
                    BatteryLoadSummary(
                        battery_pack_id=pack.identifier,
                        voltage_domain=pack.voltage_domain,
                        max_output_w=pack.max_continuous_w,
                        allocated_power_w=0.0,
                    )
                    for pack in self._battery_packs.values()
                ),
                harness_diagram=(),
                interface_manifest={"modules": [], "batteries": []},
            )

        connector_remaining: MutableMapping[str, float] = {
            connector_id: connector.max_power_w
            for connector_id, connector in self._connectors.items()
        }
        battery_remaining: MutableMapping[str, float] = {
            pack_id: pack.max_continuous_w for pack_id, pack in self._battery_packs.items()
        }
        connector_usage: MutableMapping[str, float] = {
            connector_id: 0.0 for connector_id in self._connectors
        }
        battery_usage: MutableMapping[str, float] = {
            pack_id: 0.0 for pack_id in self._battery_packs
        }

        assignments: list[ModuleAllocation] = []

        for request in requests:
            profile = self._resolve_profile(request)
            voltage_domain = request.voltage_domain or (profile.voltage_domain if profile else None)
            if voltage_domain is None:
                raise ValueError(
                    f"Module '{request.module_id}' did not declare a voltage domain and no profile was resolved."
                )
            if voltage_domain not in self._domains:
                raise ValueError(
                    f"Module '{request.module_id}' references unknown voltage domain '{voltage_domain}'."
                )

            rule = self._compatibility.get(request.module_type)
            eligible_connectors = self._eligible_connectors(
                request.module_type, voltage_domain, rule
            )
            if not eligible_connectors:
                raise ValueError(
                    f"No connectors are compatible with module '{request.module_id}' ({request.module_type})."
                )

            module_power = self._required_power(request, profile)
            paths_required = self._paths_required(request, profile, rule)
            per_path_power = module_power / float(paths_required)

            used_connectors: set[str] = set()
            used_packs: set[str] = set()
            connector_allocations: list[ConnectorAllocation] = []

            for _ in range(paths_required):
                connector_id = self._select_connector(
                    eligible_connectors,
                    used_connectors,
                    used_packs,
                    per_path_power,
                    connector_remaining,
                    battery_remaining,
                    require_unique_pack=paths_required > 1,
                )
                connector = self._connectors[connector_id]
                pack_id = connector.battery_pack_id

                connector_remaining[connector_id] -= per_path_power
                battery_remaining[pack_id] -= per_path_power
                connector_usage[connector_id] += per_path_power
                battery_usage[pack_id] += per_path_power

                used_connectors.add(connector_id)
                used_packs.add(pack_id)
                connector_allocations.append(
                    ConnectorAllocation(
                        connector_id=connector_id,
                        battery_pack_id=pack_id,
                        voltage_domain=connector.voltage_domain,
                        allocated_power_w=per_path_power,
                    )
                )

            assignments.append(
                ModuleAllocation(
                    module_id=request.module_id,
                    module_type=request.module_type,
                    voltage_domain=voltage_domain,
                    power_w=module_power,
                    connections=tuple(connector_allocations),
                )
            )

        battery_summaries = tuple(
            BatteryLoadSummary(
                battery_pack_id=pack_id,
                voltage_domain=self._battery_packs[pack_id].voltage_domain,
                max_output_w=self._battery_packs[pack_id].max_continuous_w,
                allocated_power_w=battery_usage[pack_id],
            )
            for pack_id in self._battery_packs
        )

        harness_diagram = tuple(
            {
                "module_id": allocation.module_id,
                "module_type": allocation.module_type,
                "voltage_domain": allocation.voltage_domain,
                "paths": [
                    {
                        "connector_id": connection.connector_id,
                        "battery_pack_id": connection.battery_pack_id,
                        "voltage_domain": connection.voltage_domain,
                        "power_w": connection.allocated_power_w,
                    }
                    for connection in allocation.connections
                ],
            }
            for allocation in assignments
        )

        interface_manifest = {
            "modules": [
                {
                    "module_id": allocation.module_id,
                    "module_type": allocation.module_type,
                    "voltage_domain": allocation.voltage_domain,
                    "power_budget_w": allocation.power_w,
                    "connections": [
                        {
                            **connection.to_mapping(),
                            "capacity_margin_w": connector_remaining[connection.connector_id],
                        }
                        for connection in allocation.connections
                    ],
                }
                for allocation in assignments
            ],
            "batteries": [summary.to_mapping() for summary in battery_summaries],
        }

        return PowerLayoutResult(
            assignments=tuple(assignments),
            battery_summaries=battery_summaries,
            harness_diagram=harness_diagram,
            interface_manifest=interface_manifest,
        )

    def _resolve_profile(self, request: ModuleRequest) -> ModuleProfile | None:
        if request.profile_id:
            profile = self._module_profiles.get(request.profile_id)
            if profile is None:
                raise KeyError(
                    f"Module '{request.module_id}' references unknown profile '{request.profile_id}'."
                )
            return profile
        candidates = [
            profile
            for profile in self._module_profiles.values()
            if profile.module_type == request.module_type
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _eligible_connectors(
        self,
        module_type: str,
        voltage_domain: str,
        rule: CompatibilityRule | None,
    ) -> list[str]:
        eligible = [
            connector_id
            for connector_id, connector in self._connectors.items()
            if connector.voltage_domain == voltage_domain
            and module_type in connector.compatible_modules
        ]
        if rule is not None:
            allowed = set(rule.allowed_connectors)
            eligible = [connector_id for connector_id in eligible if connector_id in allowed]
        return eligible

    def _required_power(self, request: ModuleRequest, profile: ModuleProfile | None) -> float:
        if request.power_w is not None:
            return float(request.power_w)
        if profile is None:
            raise ValueError(
                f"Module '{request.module_id}' must declare 'power_w' when no profile is provided."
            )
        return float(profile.peak_power_w or profile.nominal_power_w)

    def _paths_required(
        self,
        request: ModuleRequest,
        profile: ModuleProfile | None,
        rule: CompatibilityRule | None,
    ) -> int:
        values = [1]
        if request.redundancy_paths:
            values.append(int(request.redundancy_paths))
        if profile and profile.redundancy_paths:
            values.append(int(profile.redundancy_paths))
        if rule and rule.minimum_paths:
            values.append(int(rule.minimum_paths))
        if request.critical or (rule and rule.requires_redundancy):
            values.append(2)
        return max(values)

    def _select_connector(
        self,
        eligible_connectors: Iterable[str],
        used_connectors: set[str],
        used_packs: set[str],
        per_path_power: float,
        connector_remaining: MutableMapping[str, float],
        battery_remaining: MutableMapping[str, float],
        *,
        require_unique_pack: bool,
    ) -> str:
        def choose(prefer_new_pack: bool) -> str | None:
            best_id: str | None = None
            best_remaining = -1.0
            for connector_id in eligible_connectors:
                if connector_id in used_connectors:
                    continue
                connector = self._connectors[connector_id]
                pack_id = connector.battery_pack_id
                if prefer_new_pack and pack_id in used_packs:
                    continue
                if connector_remaining[connector_id] < per_path_power:
                    continue
                if battery_remaining[pack_id] < per_path_power:
                    continue
                remaining = min(connector_remaining[connector_id], battery_remaining[pack_id])
                if remaining > best_remaining:
                    best_remaining = remaining
                    best_id = connector_id
            return best_id

        connector_id = choose(prefer_new_pack=True)
        if connector_id is None:
            connector_id = choose(prefer_new_pack=False)
            if (
                connector_id is not None
                and require_unique_pack
                and self._connectors[connector_id].battery_pack_id in used_packs
            ):
                raise ValueError(
                    "Unable to satisfy redundancy requirement with distinct battery packs."
                )
        if connector_id is None:
            raise ValueError("No available connector capacity remains for redundant routing.")
        return connector_id
