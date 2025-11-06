from __future__ import annotations

import pytest

from src.pipelines.undersuit import UndersuitPipeline
from src.schemas.validators import validate_material_catalog
from src.suit.material_model import MaterialCatalog


@pytest.fixture
def sample_catalog_payload() -> dict[str, object]:
    return {
        "version": "2024.1",
        "description": "Test catalog for undersuit materials",
        "stacks": [
            {
                "id": "baseline_knit",
                "name": "Baseline Knit",
                "layers": [
                    {"material": "nylon brushed", "thickness_mm": 0.6},
                    {
                        "material": "vent foam",
                        "thickness_mm": 1.2,
                        "description": "Open-cell foam for airflow",
                    },
                ],
                "elastic_modulus": {"warp": 45.0, "weft": 30.0, "bias": 35.0, "unit": "MPa"},
                "thermal_resistance": {"clo": 0.8, "r_value": 0.12},
                "pressure_comfort": {
                    "preferred_kpa": 4.5,
                    "max_kpa": 8.0,
                    "notes": "Baseline comfort envelope",
                },
                "applications": ["torso", "arms"],
            },
            {
                "id": "vented_mesh",
                "name": "Vented Mesh",
                "layers": [
                    {"material": "poly mesh", "thickness_mm": 0.4},
                ],
                "elastic_modulus": {"warp": 25.0, "weft": 20.0, "unit": "MPa"},
                "thermal_resistance": {"clo": 0.3},
                "pressure_comfort": {"preferred_kpa": 3.0, "max_kpa": 6.0},
                "applications": ["underarm"],
            },
        ],
    }


def test_material_payload_matches_schema(sample_catalog_payload: dict[str, object]) -> None:
    # Should not raise
    validate_material_catalog(sample_catalog_payload)


def test_material_catalog_parses_layers(sample_catalog_payload: dict[str, object]) -> None:
    catalog = MaterialCatalog.from_payload(sample_catalog_payload)
    stack = catalog.get_stack("baseline_knit")

    assert pytest.approx(1.8, rel=1e-6) == stack.total_thickness_mm
    assert stack.elastic_modulus.warp == 45.0
    assert stack.elastic_modulus.weft == 30.0
    assert stack.pressure_comfort.max_kpa == 8.0
    assert stack.thermal_resistance.clo == 0.8
    assert stack.applications == ("torso", "arms")


def test_stack_interpolation_blends_properties(sample_catalog_payload: dict[str, object]) -> None:
    catalog = MaterialCatalog.from_payload(sample_catalog_payload)

    blend = catalog.blend("baseline_knit", "vented_mesh", 0.25, name="Torso Vent Blend")

    assert blend.name == "Torso Vent Blend"
    assert pytest.approx(0.25, rel=1e-6) == blend.to_metadata()["composition"]["ratio"]
    assert pytest.approx(40.0, rel=1e-6) == blend.elastic_modulus.warp
    assert pytest.approx(27.5, rel=1e-6) == blend.elastic_modulus.weft
    assert pytest.approx(0.675, rel=1e-6) == blend.thermal_resistance.clo
    assert pytest.approx(4.125, rel=1e-6) == blend.pressure_comfort.preferred_kpa
    assert pytest.approx(1.45, rel=1e-6) == blend.total_thickness_mm


def test_interpolation_ratio_bounds(sample_catalog_payload: dict[str, object]) -> None:
    catalog = MaterialCatalog.from_payload(sample_catalog_payload)

    with pytest.raises(ValueError):
        catalog.blend("baseline_knit", "vented_mesh", 1.5)

    with pytest.raises(ValueError):
        catalog.blend("baseline_knit", "vented_mesh", -0.1)


def test_undersuit_pipeline_assigns_material_metadata(
    sample_catalog_payload: dict[str, object],
) -> None:
    catalog = MaterialCatalog.from_payload(sample_catalog_payload)
    pipeline = UndersuitPipeline(
        material_catalog=catalog,
        panel_templates=("torso", "left_arm", "right_arm"),
        default_stack_id="baseline_knit",
        metadata_overrides={"generator": "unit-test"},
    )

    result = pipeline.generate(
        subject_id="tester-001",
        panel_materials={"left_arm": "vented_mesh"},
        panel_blends={"torso": ("baseline_knit", "vented_mesh", 0.5)},
    )

    assert {panel["name"] for panel in result["panels"]} == {"torso", "left_arm", "right_arm"}

    torso_panel = next(panel for panel in result["panels"] if panel["name"] == "torso")
    assert torso_panel["material_stack"]["kind"] == "blend"
    assert torso_panel["material_stack_id"].startswith("blend:baseline_knit:vented_mesh")

    left_arm_panel = next(panel for panel in result["panels"] if panel["name"] == "left_arm")
    assert left_arm_panel["material_stack_id"] == "vented_mesh"

    metadata = result["metadata"]
    assert metadata["subject_id"] == "tester-001"
    assert metadata["generator"] == "unit-test"
    assert metadata["material_catalog_version"] == "2024.1"
    assert metadata["materials"]["left_arm"]["id"] == "vented_mesh"
    assert metadata["materials"]["right_arm"]["id"] == "baseline_knit"
    assert metadata["materials"]["torso"]["kind"] == "blend"
