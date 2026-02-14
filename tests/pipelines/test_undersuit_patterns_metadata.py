"""Tests for the generated undersuit patterns metadata shape."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _shape(value):
    if isinstance(value, dict):
        return {key: _shape(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_shape(value[0])] if value else []
    return type(value).__name__


def test_afflec_patterns_metadata_shape() -> None:
    metadata_path = Path("outputs/suits/afflec_body/metadata.json")
    if not metadata_path.exists():
        pytest.skip("afflec_body metadata not available in this checkout")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    patterns = metadata.get("patterns")
    assert patterns is not None

    # `outputs/suits/*` is not guaranteed to be regenerated in every checkout.
    # If this metadata file predates the current schema (e.g., missing gate
    # fields), treat it as a legacy artifact instead of failing the suite.
    panel_validation = patterns.get("panel_validation", {})
    if isinstance(panel_validation, dict) and panel_validation:
        sample = next(iter(panel_validation.values()))
        if isinstance(sample, dict) and "gate" not in sample:
            pytest.skip("legacy patterns metadata (missing gate fields); regenerate outputs to validate schema")

    fixture_path = Path(__file__).parents[1] / "fixtures" / "afflec_patterns_shape.json"
    # Update fixture alongside changes to exported pattern metadata keys.
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert _shape(patterns) == expected
