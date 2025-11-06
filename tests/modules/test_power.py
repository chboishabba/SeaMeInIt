from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

import pytest

from src.modules.power import ModuleRequest, PowerLayoutPlanner
from src.pipelines.hard_shell import HardShellPipeline
from src.schemas.validators import SchemaValidationError, validate_power_catalog
from src.suit_hard.attachments import PanelSegment


@pytest.fixture()
def power_catalog() -> dict[str, object]:
    return {
        "version": "1.0.0",
        "voltage_domains": [
            {
                "id": "hv",
                "nominal_voltage": 96.0,
                "min_voltage": 88.0,
                "max_voltage": 104.0,
            },
            {
                "id": "lv",
                "nominal_voltage": 24.0,
                "min_voltage": 22.0,
                "max_voltage": 26.0,
            },
        ],
        "battery_packs": [
            {
                "id": "pack-alpha",
                "voltage_domain": "hv",
                "capacity_wh": 920.0,
                "max_continuous_w": 600.0,
                "connectors": ["hv-main"],
            },
            {
                "id": "pack-beta",
                "voltage_domain": "hv",
                "capacity_wh": 920.0,
                "max_continuous_w": 600.0,
                "connectors": ["hv-aux"],
            },
            {
                "id": "pack-gamma",
                "voltage_domain": "lv",
                "capacity_wh": 480.0,
                "max_continuous_w": 250.0,
                "connectors": ["lv-service"],
            },
        ],
        "connectors": [
            {
                "id": "hv-main",
                "name": "HV Main Bus",
                "voltage_domain": "hv",
                "battery_pack": "pack-alpha",
                "max_power_w": 550.0,
                "compatible_modules": ["cooling", "heating"],
                "tags": ["primary"],
            },
            {
                "id": "hv-aux",
                "name": "HV Auxiliary",
                "voltage_domain": "hv",
                "battery_pack": "pack-beta",
                "max_power_w": 450.0,
                "compatible_modules": ["cooling"],
                "tags": ["redundant"],
            },
            {
                "id": "lv-service",
                "name": "LV Service",
                "voltage_domain": "lv",
                "battery_pack": "pack-gamma",
                "max_power_w": 200.0,
                "compatible_modules": ["heating"],
                "tags": ["service"],
            },
        ],
        "module_profiles": [
            {
                "id": "cooling_primary",
                "module_type": "cooling",
                "voltage_domain": "hv",
                "nominal_power_w": 360.0,
                "peak_power_w": 420.0,
                "redundancy_paths": 2,
            },
            {
                "id": "heating_limb",
                "module_type": "heating",
                "voltage_domain": "lv",
                "nominal_power_w": 120.0,
                "redundancy_paths": 1,
            },
        ],
        "compatibility_rules": [
            {
                "module_type": "cooling",
                "allowed_connectors": ["hv-main", "hv-aux"],
                "minimum_paths": 2,
                "requires_redundancy": True,
            },
            {
                "module_type": "heating",
                "allowed_connectors": ["lv-service"],
            },
        ],
    }


@pytest.fixture()
def power_modules() -> Sequence[Mapping[str, object]]:
    return [
        {
            "module_id": "cooling_loop",
            "module_type": "cooling",
            "profile_id": "cooling_primary",
            "critical": True,
        },
        {
            "module_id": "heater_core",
            "module_type": "heating",
            "profile_id": "heating_limb",
        },
    ]


@pytest.fixture()
def attachment_catalog() -> Mapping[str, object]:
    return {
        "version": "1.0.0",
        "description": "Minimal attachment catalog for power testing",
        "connectors": [
            {
                "id": "rail_m4",
                "name": "M4 utility rail",
                "connector_type": "rail",
                "fastener_pattern": {
                    "primitive": "bolt",
                    "count": 2,
                    "diameter": 0.004,
                    "spacing": 0.025,
                },
                "placement": {
                    "normal_offset": 0.008,
                    "default_orientation": "longitudinal",
                },
                "compatibility_tags": ["power", "data", "utility"],
            }
        ],
        "placement_rules": [
            {
                "panel": "torso_front",
                "allowed_connector_types": ["rail"],
                "required_compatibility": ["power"],
                "max_per_panel": 2,
                "spacing": 0.18,
                "edge_clearance": 0.05,
                "normal_offset": 0.006,
                "harness": "front_bus",
                "routing_tags": ["power", "data"],
                "nominal_cable_length": 0.6,
            }
        ],
        "compatibility_tags": [
            {"name": "power", "description": "High current bus"},
            {"name": "data", "description": "Digital signaling"},
            {"name": "utility", "description": "General purpose"},
        ],
    }


@pytest.fixture()
def panel_segments() -> Sequence[PanelSegment]:
    return [
        PanelSegment(
            "torso_front",
            outline=[
                (0.0, 0.0, 0.0),
                (0.4, 0.0, 0.0),
                (0.4, 0.5, 0.0),
                (0.0, 0.5, 0.0),
            ],
            normal=(0.0, 0.0, 1.0),
            tags={"primary"},
        )
    ]


def test_power_catalog_validation(power_catalog: Mapping[str, object]) -> None:
    validate_power_catalog(power_catalog)

    invalid_catalog = deepcopy(power_catalog)
    broken_connector = deepcopy(invalid_catalog["connectors"][0])  # type: ignore[index]
    broken_connector.pop("battery_pack", None)
    invalid_catalog["connectors"] = [broken_connector]

    with pytest.raises(SchemaValidationError):
        validate_power_catalog(invalid_catalog)


def test_power_layout_allocates_redundant_paths(
    power_catalog: Mapping[str, object], power_modules: Sequence[Mapping[str, object]]
) -> None:
    planner = PowerLayoutPlanner(power_catalog)
    result = planner.allocate(power_modules)

    assert len(result.assignments) == 2

    cooling_assignment = next(
        entry for entry in result.assignments if entry.module_id == "cooling_loop"
    )
    connector_ids = {connection.connector_id for connection in cooling_assignment.connections}
    assert connector_ids == {"hv-main", "hv-aux"}
    assert all(
        connection.allocated_power_w == pytest.approx(210.0, rel=1e-3)
        for connection in cooling_assignment.connections
    )

    battery_summary = {summary.battery_pack_id: summary for summary in result.battery_summaries}
    assert pytest.approx(battery_summary["pack-alpha"].allocated_power_w, rel=1e-3) == 210.0
    assert pytest.approx(battery_summary["pack-beta"].allocated_power_w, rel=1e-3) == 210.0
    assert pytest.approx(battery_summary["pack-gamma"].allocated_power_w, rel=1e-3) == 120.0

    harness_entry = next(
        entry for entry in result.harness_diagram if entry["module_id"] == "cooling_loop"
    )
    assert len(harness_entry["paths"]) == 2

    interface_modules = {
        module["module_id"]: module for module in result.interface_manifest["modules"]
    }
    assert interface_modules["heater_core"]["connections"][0]["connector_id"] == "lv-service"


def test_pipeline_emits_power_payload(
    attachment_catalog: Mapping[str, object],
    panel_segments: Sequence[PanelSegment],
    power_catalog: Mapping[str, object],
    power_modules: Sequence[Mapping[str, object]],
) -> None:
    pipeline = HardShellPipeline(catalog=attachment_catalog, power_catalog=power_catalog)
    payload = pipeline.generate(panel_segments, power_modules=power_modules)

    assert "power" in payload
    power_payload = payload["power"]
    harness_ids = {entry["module_id"] for entry in power_payload["harness_diagram"]}
    assert {"cooling_loop", "heater_core"} <= harness_ids

    manifest_modules = {
        entry["module_id"]: entry for entry in power_payload["interface_manifest"]["modules"]
    }
    assert manifest_modules["cooling_loop"]["connections"][0]["connector_id"] in {
        "hv-main",
        "hv-aux",
    }


def test_redundancy_failure_when_packs_not_distinct(
    power_catalog: Mapping[str, object], power_modules: Sequence[Mapping[str, object]]
) -> None:
    limited_catalog = deepcopy(power_catalog)
    for connector in limited_catalog["connectors"]:  # type: ignore[index]
        if connector["id"] == "hv-aux":
            connector["battery_pack"] = "pack-alpha"
    for pack in limited_catalog["battery_packs"]:  # type: ignore[index]
        if pack["id"] == "pack-alpha":
            connectors = set(pack.get("connectors", []))
            connectors.add("hv-main")
            connectors.add("hv-aux")
            pack["connectors"] = sorted(connectors)

    planner = PowerLayoutPlanner(limited_catalog)

    with pytest.raises(ValueError):
        planner.allocate([ModuleRequest.from_mapping(power_modules[0])])
