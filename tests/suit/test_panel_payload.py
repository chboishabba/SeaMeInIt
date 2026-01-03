"""Tests for panel payload serialization shape."""

from __future__ import annotations

import json
from pathlib import Path

from suit.panel_payload import PanelPayload


def _shape(value):
    if isinstance(value, dict):
        return {key: _shape(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_shape(value[0])] if value else []
    return type(value).__name__


def test_panel_payload_schema_is_stable() -> None:
    payload = PanelPayload(
        name="front",
        vertices=((0.0, 1.0, 2.0), (0.5, 1.5, 2.5), (1.0, 2.0, 3.0)),
        faces=((0, 1, 2),),
        metadata={"source": "demo", "scale": 1.0},
    )

    fixture_path = Path(__file__).parents[1] / "fixtures" / "panel_payload_shape.json"
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert _shape(payload.to_mapping()) == expected


def test_panel_payload_omits_empty_metadata() -> None:
    payload = PanelPayload(
        name="front",
        vertices=((0.0, 1.0, 2.0), (0.5, 1.5, 2.5), (1.0, 2.0, 3.0)),
        faces=((0, 1, 2),),
    )

    assert "metadata" not in payload.to_mapping()
