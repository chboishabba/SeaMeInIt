"""Utilities for validating SeaMeInIt schema instances."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import json

try:
    import yaml  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]
from jsonschema import Draft202012Validator, ValidationError

DEFAULT_SCHEMA_NAME = "body_unified.yaml"
SUIT_MATERIAL_SCHEMA_NAME = "suit_materials.yaml"
HARD_LAYER_ATTACHMENT_SCHEMA_NAME = "hard_layer_interfaces.yaml"

__all__ = [
    "DEFAULT_SCHEMA_NAME",
    "SUIT_MATERIAL_SCHEMA_NAME",
    "SchemaValidationError",
    "load_schema",
    "load_measurement_catalog",
    "load_material_catalog",
    "load_attachment_catalog",
    "load_payload",
    "validate_body_payload",
    "validate_material_catalog",
    "validate_attachment_catalog",
    "validate_file",
]


class SchemaValidationError(RuntimeError):
    """Raised when an instance fails schema validation."""

    def __init__(self, errors: Iterable[ValidationError]):
        self.errors = tuple(errors)
        message = "Schema validation failed:\n" + "\n".join(_format_error(e) for e in self.errors)
        super().__init__(message)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _schema_dir() -> Path:
    return _repository_root() / "schemas"


def _measurement_schema_path() -> Path:
    return _repository_root() / "data" / "schemas" / "body_measurements.json"


@lru_cache(maxsize=4)
def load_schema(name: str = DEFAULT_SCHEMA_NAME) -> Mapping[str, Any]:
    """Load and cache a schema definition by name."""

    schema_path = _schema_dir() / name
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema '{name}' not found at {schema_path}")

    with schema_path.open("r", encoding="utf-8") as handle:
        schema_text = handle.read()

    schema = _parse_yaml_text(schema_text, source=schema_path)

    if not isinstance(schema, Mapping):
        raise TypeError(f"Schema '{name}' must decode to a mapping, received {type(schema)!r}")

    return schema


def load_measurement_catalog(
    *,
    schema_name: str = DEFAULT_SCHEMA_NAME,
    measurement_path: Path | None = None,
) -> dict[str, Any]:
    """Load measurement metadata from the unified schema and validate artifacts.

    Parameters
    ----------
    schema_name:
        The unified YAML schema containing authoritative measurement definitions.
    measurement_path:
        Optional override for the generated JSON artifact. When provided, the
        artifact is checked for parity with the unified schema. Defaults to the
        repository's ``data/schemas/body_measurements.json`` file.
    """

    schema = load_schema(schema_name)
    extension = schema.get("x-measurements")
    if not isinstance(extension, Mapping):
        raise KeyError("Unified schema is missing the 'x-measurements' extension.")

    manual = _normalize_measurements(extension.get("manual", []), context="manual")
    scan = _normalize_measurements(extension.get("scan_landmarks", []), context="scan_landmarks")

    catalog: dict[str, Any] = {
        "version": str(extension.get("version", "")),
        "description": str(extension.get("description", "")),
        "manual_measurements": manual,
        "scan_landmarks": scan,
    }

    artifact_path = measurement_path or _measurement_schema_path()
    artifact = _load_measurement_artifact(artifact_path)

    _assert_measurement_parity(manual, artifact.get("manual_measurements", []), context="manual_measurements")
    _assert_measurement_parity(scan, artifact.get("scan_landmarks", []), context="scan_landmarks")

    if not catalog["version"] and isinstance(artifact.get("version"), str):
        catalog["version"] = artifact["version"]
    elif catalog["version"] and artifact.get("version") not in {None, catalog["version"]}:
        raise ValueError(
            "Measurement artifact version does not match unified schema: "
            f"{artifact.get('version')!r} != {catalog['version']!r}"
        )

    if not catalog["description"] and isinstance(artifact.get("description"), str):
        catalog["description"] = artifact["description"]

    return catalog


def _normalize_measurements(values: Any, *, context: str) -> list[dict[str, Any]]:
    if not isinstance(values, Iterable):
        raise TypeError(f"'{context}' must be an iterable of measurement mappings.")

    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(values):
        if not isinstance(entry, Mapping):
            raise TypeError(f"Entry {index} in '{context}' must be a mapping, received {type(entry)!r}.")
        if "name" not in entry:
            raise KeyError(f"Entry {index} in '{context}' is missing a 'name' field.")
        if "unit" not in entry:
            raise KeyError(f"Entry {index} in '{context}' is missing a 'unit' field.")

        record: dict[str, Any] = {
            "name": str(entry["name"]),
            "unit": str(entry["unit"]),
            "required": bool(entry.get("required", False)),
        }
        for optional_key in ("category", "description"):
            if optional_key in entry:
                record[optional_key] = entry[optional_key]
        normalized.append(record)

    return normalized


def _load_measurement_artifact(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Measurement artifact not found at {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise TypeError(f"Measurement artifact at {path} must decode to a mapping.")

    return payload


def _assert_measurement_parity(
    unified: Iterable[Mapping[str, Any]],
    artifact: Iterable[Mapping[str, Any]],
    *,
    context: str,
) -> None:
    unified_index = {entry["name"]: entry for entry in _normalize_measurements(unified, context=context + "_unified")}
    artifact_index = {entry["name"]: entry for entry in _normalize_measurements(artifact, context=context + "_artifact")}

    if unified_index.keys() != artifact_index.keys():
        missing_in_artifact = sorted(set(unified_index) - set(artifact_index))
        missing_in_unified = sorted(set(artifact_index) - set(unified_index))
        raise ValueError(
            f"Mismatch in {context} identifiers. Missing in artifact: {missing_in_artifact}; "
            f"missing in unified schema: {missing_in_unified}"
        )

    for name, unified_entry in unified_index.items():
        artifact_entry = artifact_index[name]
        if unified_entry.get("unit") != artifact_entry.get("unit"):
            raise ValueError(
                f"Unit mismatch for {name!r} in {context}: {artifact_entry.get('unit')!r} != {unified_entry.get('unit')!r}"
            )
        if bool(artifact_entry.get("required", False)) != bool(unified_entry.get("required", False)):
            raise ValueError(
                f"Required flag mismatch for {name!r} in {context}: "
                f"{artifact_entry.get('required')!r} != {unified_entry.get('required')!r}"
            )

def load_payload(path: Path) -> Any:
    """Load a JSON or YAML payload from disk."""

    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix in {".yaml", ".yml"}:
            return _parse_yaml_text(handle.read(), source=path)
        if suffix == ".json":
            return json.load(handle)
    raise ValueError(f"Unsupported payload extension '{suffix}' for {path}")


def validate_body_payload(instance: Any, *, schema_name: str = DEFAULT_SCHEMA_NAME) -> None:
    """Validate *instance* against the SeaMeInIt body schema."""

    schema = load_schema(schema_name)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda exc: exc.path)
    if errors:
        raise SchemaValidationError(errors)


def validate_material_catalog(instance: Any, *, schema_name: str = SUIT_MATERIAL_SCHEMA_NAME) -> None:
    """Validate *instance* against the suit material catalog schema."""

    schema = load_schema(schema_name)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda exc: exc.path)
    if errors:
        raise SchemaValidationError(errors)


def validate_attachment_catalog(instance: Any, *, schema_name: str = HARD_LAYER_ATTACHMENT_SCHEMA_NAME) -> None:
    """Validate *instance* against the hard-layer attachment schema."""

    schema = load_schema(schema_name)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda exc: exc.path)
    if errors:
        raise SchemaValidationError(errors)


def load_material_catalog(path: Path) -> Mapping[str, Any]:
    """Load and validate a suit material catalog payload from *path*."""

    instance = load_payload(path)
    if not isinstance(instance, Mapping):
        raise TypeError("Material catalog payload must be a mapping.")

    validate_material_catalog(instance)
    return instance


def load_attachment_catalog(path: Path) -> Mapping[str, Any]:
    """Load and validate a hard-layer attachment catalog payload from *path*."""

    instance = load_payload(path)
    if not isinstance(instance, Mapping):
        raise TypeError("Attachment catalog payload must be a mapping.")

    validate_attachment_catalog(instance)
    return instance


def validate_file(path: Path, *, schema_name: str = DEFAULT_SCHEMA_NAME) -> Any:
    """Load a payload from *path*, validate it, and return the parsed instance."""

    instance = load_payload(path)
    validate_body_payload(instance, schema_name=schema_name)
    return instance


def _format_error(error: ValidationError) -> str:
    location = " / ".join(str(component) for component in error.absolute_path)
    prefix = f"[{location}] " if location else ""
    return f"{prefix}{error.message}"


def _parse_yaml_text(text: str, *, source: Path | None = None) -> Any:
    if yaml is not None:
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - exercised when PyYAML missing
        raise ModuleNotFoundError(
            "PyYAML is required to parse YAML files; install the 'pyyaml' extra to proceed."
        ) from exc
