"""Tests for panel payload adapters."""

from __future__ import annotations

import pytest

from suit import Panel, PanelPayloadSource, SurfacePatch, panel_to_payload


def test_panel_to_payload_from_vertex_indices() -> None:
    panel = Panel(
        panel_id="front",
        surface_patch=SurfacePatch(vertex_indices=(0, 1, 2, 3)),
    )
    source = PanelPayloadSource(
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )

    payload = panel_to_payload(panel, source)

    assert payload.name == "front"
    assert payload.vertices == source.vertices
    assert payload.faces == ((0, 1, 2), (0, 2, 3))


def test_panel_to_payload_from_face_indices() -> None:
    panel = Panel(
        panel_id="front",
        surface_patch=SurfacePatch(face_indices=(0, 1)),
    )
    source = PanelPayloadSource(
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        faces=((0, 1, 2), (0, 2, 3)),
    )

    payload = panel_to_payload(panel, source)

    assert payload.vertices == source.vertices
    assert payload.faces == source.faces


def test_panel_to_payload_can_omit_surface_metadata() -> None:
    panel = Panel(
        panel_id="front",
        surface_patch=SurfacePatch(
            vertex_indices=(0, 1, 2),
            metadata={"panel_curvature": 0.2},
        ),
    )
    source = PanelPayloadSource(
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
    )

    payload = panel_to_payload(panel, source, include_surface_metadata=False)

    assert payload.to_mapping() == {
        "name": "front",
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        "faces": [[0, 1, 2]],
    }


def test_panel_to_payload_requires_surface_patch() -> None:
    panel = Panel(panel_id="front")
    source = PanelPayloadSource(vertices=((0.0, 0.0, 0.0),))

    with pytest.raises(ValueError, match="surface_patch"):
        panel_to_payload(panel, source)
