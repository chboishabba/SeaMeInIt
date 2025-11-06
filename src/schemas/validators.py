"""Utilities for validating SeaMeInIt schema instances."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import json

import yaml
from jsonschema import Draft202012Validator, ValidationError

DEFAULT_SCHEMA_NAME = "body_unified.yaml"

__all__ = [
    "DEFAULT_SCHEMA_NAME",
    "SchemaValidationError",
    "load_schema",
    "load_payload",
    "validate_body_payload",
    "validate_file",
]


class SchemaValidationError(RuntimeError):
    """Raised when an instance fails schema validation."""

    def __init__(self, errors: Iterable[ValidationError]):
        self.errors = tuple(errors)
        message = "Schema validation failed:\n" + "\n".join(_format_error(e) for e in self.errors)
        super().__init__(message)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _schema_dir() -> Path:
    return _repository_root() / "schemas"


@lru_cache(maxsize=4)
def load_schema(name: str = DEFAULT_SCHEMA_NAME) -> Mapping[str, Any]:
    """Load and cache a schema definition by name."""

    schema_path = _schema_dir() / name
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema '{name}' not found at {schema_path}")

    with schema_path.open("r", encoding="utf-8") as handle:
        schema = yaml.safe_load(handle)

    if not isinstance(schema, Mapping):
        raise TypeError(f"Schema '{name}' must decode to a mapping, received {type(schema)!r}")

    return schema


def load_payload(path: Path) -> Any:
    """Load a JSON or YAML payload from disk."""

    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
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


def validate_file(path: Path, *, schema_name: str = DEFAULT_SCHEMA_NAME) -> Any:
    """Load a payload from *path*, validate it, and return the parsed instance."""

    instance = load_payload(path)
    validate_body_payload(instance, schema_name=schema_name)
    return instance


def _format_error(error: ValidationError) -> str:
    location = " / ".join(str(component) for component in error.absolute_path)
    prefix = f"[{location}] " if location else ""
    return f"{prefix}{error.message}"
