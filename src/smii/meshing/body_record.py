"""Loaders for fitted body mesh records used across tooling."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["load_body_record"]


def _normalise_mesh_array(array: np.ndarray, *, dtype: object) -> np.ndarray:
    result = np.asarray(array, dtype=dtype)
    if result.ndim == 3:
        result = result[0]
    return result


def _coerce_npz_value(value: np.ndarray) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            converted = value.tolist()
            if isinstance(converted, list) and len(converted) == 1:
                return converted[0]
            return converted
        if value.ndim == 0:
            return value.item()
    return value


def load_body_record(path: Path) -> dict[str, Any]:
    """Load a fitted body mesh from JSON or NPZ disk representations."""

    suffix = path.suffix.lower()
    record: dict[str, Any] = {}

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, Mapping):
            raise TypeError("Body record JSON must be an object with vertices/faces arrays.")
        for key, value in payload.items():
            if key == "vertices":
                record[key] = _normalise_mesh_array(value, dtype=float)
            elif key == "faces":
                record[key] = _normalise_mesh_array(value, dtype=int)
            else:
                record[key] = value
    elif suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            for key in data.files:
                value = data[key]
                if key == "vertices":
                    record[key] = _normalise_mesh_array(value, dtype=float)
                elif key == "faces":
                    record[key] = _normalise_mesh_array(value, dtype=int)
                else:
                    record[key] = _coerce_npz_value(value)
    else:
        raise ValueError(f"Unsupported body record format: {path.suffix}")

    return record
