from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

np = pytest.importorskip("numpy")

from modules.cooling import plan_cooling_layout
from suit.thermal_zones import ThermalZone, ThermalZoneSpec

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import numpy as np


@pytest.fixture(scope="module")
def sample_spec() -> ThermalZoneSpec:
    return ThermalZoneSpec(
        name="demo",
        zones=(
            ThermalZone(
                identifier="core",
                label="Core",
                description="Core torso panel",
                default_heat_load=180.0,
                bounds={"z": (0.45, 0.75), "y": (0.45, 0.7)},
            ),
            ThermalZone(
                identifier="arms",
                label="Arms",
                description="Bilaterial arm sleeves",
                default_heat_load=90.0,
                bounds={"z": (0.4, 0.8), "x": (0.2, 0.8)},
            ),
            ThermalZone(
                identifier="legs",
                label="Legs",
                description="Lower body panels",
                default_heat_load=60.0,
                bounds={"z": (0.0, 0.45)},
            ),
        ),
    )


@pytest.fixture(scope="module")
def sample_vertices() -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    points = rng.random((100, 3))
    points[:, 2] *= 1.8
    points[:, 1] *= 0.8
    points[:, 0] *= 0.6
    return points


def test_plan_balances_capacity_and_flow(
    sample_spec: ThermalZoneSpec, sample_vertices: np.ndarray
) -> None:
    plan = plan_cooling_layout(
        sample_spec,
        sample_vertices,
        medium="liquid",
        total_pcm_capacity=660.0,
        total_flow_rate_l_min=9.0,
    )

    pcm_capacity = plan.zone_pcm_capacity()
    flow_rates = plan.zone_flow_rate()

    assert pcm_capacity["core"] == pytest.approx(360.0)
    assert pcm_capacity["arms"] == pytest.approx(180.0)
    assert pcm_capacity["legs"] == pytest.approx(120.0)

    assert flow_rates["core"] == pytest.approx(4.909, rel=1e-3)
    assert flow_rates["arms"] == pytest.approx(2.455, rel=1e-3)
    assert flow_rates["legs"] == pytest.approx(1.636, rel=1e-3)


def test_routing_manifest_serialises(
    sample_spec: ThermalZoneSpec, sample_vertices: np.ndarray
) -> None:
    plan = plan_cooling_layout(sample_spec, sample_vertices, medium="liquid")
    manifest = plan.to_manifest()

    # ensure JSON serialisable and includes the expected keys
    serialised = json.dumps(manifest)
    assert "supply_manifold" in serialised
    assert manifest["spec"]["name"] == "demo"
    assert {node["id"] for node in manifest["routing"]["nodes"]} >= {
        "supply_manifold",
        "return_manifold",
        "zone_core",
        "zone_arms",
        "zone_legs",
    }
