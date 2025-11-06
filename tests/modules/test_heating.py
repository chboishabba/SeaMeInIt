"""Regression tests for the heating module placement and control logic."""

from __future__ import annotations

import pytest

from modules.heating import (
    HeatingController,
    HeatingMeshPlanner,
    HeaterTemplate,
    LayerWattageConfig,
    PanelGeometry,
    ZoneConfig,
)


@pytest.fixture()
def sample_templates() -> dict[str, HeaterTemplate]:
    return {
        "carbon_large": HeaterTemplate("carbon_large", wattage=24.0, coverage_area=0.036),
        "carbon_small": HeaterTemplate("carbon_small", wattage=12.0, coverage_area=0.018),
    }


@pytest.fixture()
def layered_config() -> tuple[LayerWattageConfig, ...]:
    return (
        LayerWattageConfig(
            layer_id="base",
            template_id="carbon_large",
            coverage_ratio=0.5,
            watt_density=180.0,
        ),
        LayerWattageConfig(
            layer_id="booster",
            template_id="carbon_small",
            coverage_ratio=0.25,
            watt_density=110.0,
        ),
    )


@pytest.fixture()
def torso_panels() -> tuple[PanelGeometry, ...]:
    return (
        PanelGeometry(panel_id="torso_front", area=0.24, zone="core"),
        PanelGeometry(panel_id="torso_back", area=0.22, zone="upper_torso_back"),
    )


def test_heating_mesh_assigns_layered_elements(
    sample_templates: dict[str, HeaterTemplate],
    layered_config: tuple[LayerWattageConfig, ...],
    torso_panels: tuple[PanelGeometry, ...],
) -> None:
    """Placements should honour coverage and watt density constraints."""

    planner = HeatingMeshPlanner(sample_templates, layered_config)
    plan = planner.plan(torso_panels)

    assert plan.panel_layers
    assert plan.total_wattage() == pytest.approx(288.0)

    front_layers = plan.panel_layers["torso_front"]
    assert len(front_layers) == 2

    base_layer = next(layer for layer in front_layers if layer.layer_id == "base")
    assert base_layer.element_count == 4
    assert base_layer.total_wattage == pytest.approx(96.0)
    assert base_layer.coverage_area == pytest.approx(0.144)

    booster_layer = next(layer for layer in front_layers if layer.layer_id == "booster")
    assert booster_layer.element_count == 4
    assert booster_layer.total_wattage == pytest.approx(48.0)

    bom = plan.build_bom()
    assert bom["total_wattage"] == pytest.approx(288.0)
    template_totals = bom["templates"]
    assert template_totals["carbon_large"]["element_count"] == 8
    assert template_totals["carbon_small"]["total_wattage"] == pytest.approx(96.0)


def test_controller_enforces_zone_limits(
    sample_templates: dict[str, HeaterTemplate],
    layered_config: tuple[LayerWattageConfig, ...],
    torso_panels: tuple[PanelGeometry, ...],
) -> None:
    """The controller should compute safe duty cycles for each zone."""

    planner = HeatingMeshPlanner(sample_templates, layered_config)
    plan = planner.plan(torso_panels)

    controller = HeatingController(
        {
            "core": ZoneConfig("core", max_watt_density=500.0, duty_cycle_limit=0.75),
            "upper_torso_back": ZoneConfig(
                "upper_torso_back", max_watt_density=420.0, duty_cycle_limit=0.9
            ),
        },
        global_safety_margin=0.85,
    )

    budgets = controller.derive_zone_budgets(plan)
    assert set(budgets) == {"core", "upper_torso_back"}

    core_budget = budgets["core"]
    assert core_budget.installed_power == pytest.approx(144.0)
    assert core_budget.max_safe_power == pytest.approx(120.0)
    assert core_budget.recommended_power == pytest.approx(102.0)
    assert core_budget.recommended_duty_cycle == pytest.approx(102.0 / 144.0)

    back_budget = budgets["upper_torso_back"]
    assert back_budget.installed_power == pytest.approx(144.0)
    assert back_budget.max_safe_power == pytest.approx(92.4)
    assert back_budget.recommended_power == pytest.approx(92.4 * 0.85)
    assert back_budget.recommended_duty_cycle == pytest.approx((92.4 * 0.85) / 144.0)

    summary = controller.describe()
    assert summary["global_safety_margin"] == pytest.approx(0.85)
    assert any(entry["zone_id"] == "core" for entry in summary["zones"])
