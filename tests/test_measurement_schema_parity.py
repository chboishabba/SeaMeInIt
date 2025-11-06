"""Tests that ensure measurement definitions stay synchronized."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from schemas.validators import load_measurement_catalog
from smii.pipelines.fit_from_measurements import (
    MEASUREMENT_MODELS,
    load_schema as load_measurement_schema,
    required_measurements,
)


@pytest.fixture(scope="module")
def measurement_catalog() -> dict[str, object]:
    return load_measurement_catalog()


def test_body_measurement_artifact_matches_unified_schema(measurement_catalog: dict[str, object]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_path = repo_root / "data" / "schemas" / "body_measurements.json"
    with artifact_path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)

    manual_entries = cast(list[dict[str, Any]], measurement_catalog["manual_measurements"])
    scan_entries = cast(list[dict[str, Any]], measurement_catalog["scan_landmarks"])

    assert artifact["manual_measurements"] == manual_entries
    assert artifact["scan_landmarks"] == scan_entries
    assert artifact.get("version") == measurement_catalog.get("version")


def test_measurement_models_align_with_schema(measurement_catalog: dict[str, object]) -> None:
    manual_entries = cast(list[dict[str, Any]], measurement_catalog["manual_measurements"])
    manual_names = {entry["name"] for entry in manual_entries}
    model_names = {model.name for model in MEASUREMENT_MODELS}
    assert model_names <= manual_names


def test_required_measurements_match_canonical(measurement_catalog: dict[str, object]) -> None:
    schema = load_measurement_schema()
    manual_entries = cast(list[dict[str, Any]], measurement_catalog["manual_measurements"])
    canonical_required = {entry["name"] for entry in manual_entries if entry.get("required")}
    assert required_measurements(schema) == canonical_required
