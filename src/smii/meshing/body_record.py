"""Loaders for fitted body mesh records used across tooling."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

__all__ = ["load_body_record"]


def load_body_record(path: Path) -> dict[str, np.ndarray]:
    """Load a fitted body mesh from JSON or NPZ disk representations."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, Mapping):
            raise TypeError("Body record JSON must be an object with vertices/faces arrays.")
        vertices = np.asarray(payload.get("vertices"), dtype=float)
        faces = np.asarray(payload.get("faces"), dtype=int)
    elif suffix == ".npz":
        data = np.load(path)
        vertices = np.asarray(data["vertices"], dtype=float)
        faces = np.asarray(data["faces"], dtype=int)
    else:
        raise ValueError(f"Unsupported body record format: {path.suffix}")

    if vertices.ndim == 3:
        vertices = vertices[0]
    if faces.ndim == 3:
        faces = faces[0]

    return {"vertices": vertices, "faces": faces}
