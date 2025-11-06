"""Integration tests for the undersuit thermal zoning pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from smii.tools import ThermalBrushSession, load_brush_payload, load_weights
from smii.pipelines.undersuit import UndersuitMesh, UndersuitPipeline
from suit.thermal_zones import DEFAULT_THERMAL_ZONE_SPEC


@pytest.fixture
def sample_vertices() -> list[tuple[float, float, float]]:
    """Provide a small vertex cloud approximating an undersuit body."""

    return [
        (0.5, 0.5, 0.95),  # head
        (0.45, 0.8, 0.72),  # upper torso front
        (0.55, 0.2, 0.72),  # upper torso back
        (0.5, 0.5, 0.5),  # core
        (0.5, 0.5, 0.38),  # pelvis
        (0.2, 0.5, 0.25),  # left leg
        (0.8, 0.5, 0.25),  # right leg
        (0.15, 0.6, 0.65),  # left arm
        (0.85, 0.6, 0.65),  # right arm
        (0.5, 0.5, 0.1),  # feet / fallback
    ]


def test_default_spec_assigns_vertices(sample_vertices: list[tuple[float, float, float]]) -> None:
    """Ensure every vertex is partitioned into a thermal zone."""

    spec = DEFAULT_THERMAL_ZONE_SPEC
    assignment = spec.assign_vertices(sample_vertices)

    assert assignment.total_vertices() == len(sample_vertices)

    fallback_indices = assignment.vertex_indices("fallback")
    assert fallback_indices, "fallback zone should capture outliers"

    covered = {
        zone.identifier: assignment.vertex_indices(zone.identifier)
        for zone in spec.zones
    }
    assert any(covered[zone.identifier] for zone in spec.zones if not zone.fallback)


def test_brush_session_roundtrip(tmp_path: Path) -> None:
    """Verify that brush edits persist and reload correctly."""

    session = ThermalBrushSession()
    session.apply_brush("core", 1.75)
    session.apply_brush("left_leg", 0.5)

    payload_path = tmp_path / "weights.json"
    session.save(payload_path)

    payload = load_brush_payload(payload_path)
    assert payload["weights"]["core"] == pytest.approx(1.75)

    loaded_session = ThermalBrushSession(weights=load_weights(payload_path))
    assert loaded_session.weights["left_leg"] == pytest.approx(0.5)


def test_pipeline_integration_with_weights(sample_vertices: list[tuple[float, float, float]], tmp_path: Path) -> None:
    """The undersuit pipeline should combine mesh data and brush weights."""

    mesh = UndersuitMesh(vertices=tuple(sample_vertices))

    brush_session = ThermalBrushSession()
    brush_session.apply_brush("core", 1.6)
    brush_path = tmp_path / "core_weights.json"
    brush_session.save(brush_path)

    pipeline = UndersuitPipeline()
    result = pipeline.generate(
        mesh,
        brush_path=brush_path,
        weight_overrides={"head_neck": 2.0},
    )

    payload = result.to_payload()
    assert "mesh" in payload and "thermal" in payload

    mesh_payload = payload["mesh"]
    assert len(mesh_payload["vertices"]) == len(sample_vertices)

    thermal = payload["thermal"]
    zone_entries = thermal["zones"]
    assert zone_entries, "thermal payload should include zone entries"

    head_zone = next(zone for zone in zone_entries if zone["id"] == "head_neck")
    assert head_zone["priority_weight"] == pytest.approx(2.0)
    assert head_zone["cooling_target"] == pytest.approx(
        head_zone["default_heat_load"] * head_zone["priority_weight"]
    )

    core_zone = next(zone for zone in zone_entries if zone["id"] == "core")
    assert core_zone["priority_weight"] == pytest.approx(1.6)

    total_target = sum(zone["cooling_target"] for zone in zone_entries)
    assert thermal["total_effective_load"] == pytest.approx(total_target)
