import math
from typing import Mapping

import pytest

from src.pipelines.hard_shell import HardShellPipeline, generate_hard_shell_payload
from src.schemas.validators import validate_attachment_catalog
from src.suit_hard.attachments import AttachmentPlanner, PanelSegment


@pytest.fixture()
def attachment_catalog() -> dict[str, object]:
    return {
        "version": "1.0.0",
        "description": "Test hard-shell attachment interfaces",
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
            },
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
            },
            {
                "panel_tags_any": ["vented"],
                "allowed_connector_types": ["rail"],
                "required_compatibility": ["utility"],
                "max_per_panel": 1,
                "edge_clearance": 0.03,
                "harness": "ancillary",
                "routing_tags": ["utility"],
            },
        ],
        "compatibility_tags": [
            {"name": "power", "description": "High current bus"},
            {"name": "data", "description": "Digital signaling"},
            {"name": "utility", "description": "General purpose"},
        ],
    }


@pytest.fixture()
def panel_segments() -> list[PanelSegment]:
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
            tags={"primary", "front"},
        ),
        PanelSegment(
            "shoulder_left",
            outline=[
                (0.4, 0.4, 0.0),
                (0.65, 0.4, 0.0),
                (0.65, 0.65, 0.0),
                (0.4, 0.65, 0.0),
            ],
            normal=(0.0, 0.0, 1.0),
            tags={"vented", "upper"},
        ),
        PanelSegment(
            "torso_back",
            outline=[
                (0.0, -0.5, 0.0),
                (0.4, -0.5, 0.0),
                (0.4, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ],
            normal=(0.0, 0.0, 1.0),
            tags={"rear"},
        ),
    ]


def test_catalog_validates_against_schema(attachment_catalog: Mapping[str, object]) -> None:
    validate_attachment_catalog(attachment_catalog)


def test_attachment_planner_respects_rules(
    attachment_catalog: Mapping[str, object], panel_segments: list[PanelSegment]
) -> None:
    planner = AttachmentPlanner(attachment_catalog)
    layout = planner.place(panel_segments)

    torso_attachments = [entry for entry in layout.attachments if entry.panel_id == "torso_front"]
    shoulder_attachments = [entry for entry in layout.attachments if entry.panel_id == "shoulder_left"]

    assert len(torso_attachments) == 2
    assert len(shoulder_attachments) == 1
    assert all(att.connector_type == "rail" for att in torso_attachments)
    assert all(att.routing.harness == "front_bus" for att in torso_attachments)
    assert all("power" in att.compatibility_tags for att in torso_attachments)

    y_positions = sorted(att.position[1] for att in torso_attachments)
    assert math.isclose(y_positions[1] - y_positions[0], 0.18, rel_tol=1e-2)

    shoulder_attachment = shoulder_attachments[0]
    assert shoulder_attachment.routing.harness == "ancillary"
    assert shoulder_attachment.routing.bundle_tags == ("utility",)

    assert layout.panel_summary["torso_front"]["attachment_count"] == 2
    assert layout.panel_summary["shoulder_left"]["attachment_count"] == 1
    assert layout.panel_summary["torso_back"]["attachment_count"] == 0


def test_pipeline_serialization_emits_visualization(
    attachment_catalog: Mapping[str, object], panel_segments: list[PanelSegment]
) -> None:
    pipeline = HardShellPipeline(catalog=attachment_catalog)
    payload = pipeline.generate(panel_segments)

    placements = payload["attachments"]["placements"]
    assert len(placements) == 3
    assert payload["attachments"]["panel_summary"]["torso_front"]["attachment_count"] == 2
    assert len(payload["visualization"]["attachment_markers"]) == len(placements)

    for marker in payload["visualization"]["attachment_markers"]:
        assert marker["panel_id"] in {panel.panel_id for panel in panel_segments}

    for placement in placements:
        assert placement["routing"]["estimated_length"] > 0
        assert placement["fastener_pattern"]["count"] >= 1

    validate_attachment_catalog(payload["attachments"]["catalog"])

    functional_payload = generate_hard_shell_payload(panel_segments, catalog=attachment_catalog)
    assert functional_payload["attachments"]["placements"] == placements
