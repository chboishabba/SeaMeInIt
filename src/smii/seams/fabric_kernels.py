"""Fabric profile loaders and kernel penalty helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np

__all__ = [
    "FabricCompliance",
    "FabricAssignment",
    "FabricConstraints",
    "FabricProfile",
    "fabric_penalty",
    "load_fabric_profile",
    "load_fabrics_from_dir",
    "rotate_grain",
]


@dataclass(frozen=True, slots=True)
class FabricCompliance:
    """Directional compliance used when evaluating grain alignment."""

    s_parallel: float
    s_perp: float
    s_shear: float


@dataclass(frozen=True, slots=True)
class FabricAssignment:
    """Panel-level fabric selection and grain rotation."""

    fabric_id: str | None
    grain_rotation_deg: float = 0.0


@dataclass(frozen=True, slots=True)
class FabricConstraints:
    """Manufacturability limits for a fabric regime."""

    max_grain_rotation_deg: float | None = None
    allow_bias: bool = True


@dataclass(frozen=True, slots=True)
class FabricProfile:
    """Fabric properties parsed from configuration."""

    fabric_id: str
    description: str | None
    compliance: FabricCompliance
    mdl_modifiers: Mapping[str, float]
    constraints: FabricConstraints

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FabricProfile":
        compliance_payload = payload.get("compliance", {})
        constraints_payload = payload.get("constraints", {})
        compliance = FabricCompliance(
            s_parallel=float(compliance_payload.get("s_parallel", 1.0)),
            s_perp=float(compliance_payload.get("s_perp", 1.0)),
            s_shear=float(compliance_payload.get("s_shear", 1.0)),
        )
        constraints = FabricConstraints(
            max_grain_rotation_deg=float(constraints_payload["max_grain_rotation_per_panel_deg"])
            if "max_grain_rotation_per_panel_deg" in constraints_payload
            else None,
            allow_bias=bool(constraints_payload.get("allow_bias", True)),
        )
        mdl_modifiers = {
            "seam_length": float(payload.get("mdl_modifiers", {}).get("seam_length", 1.0)),
            "seam_count": float(payload.get("mdl_modifiers", {}).get("seam_count", 1.0)),
            "panel_count": float(payload.get("mdl_modifiers", {}).get("panel_count", 1.0)),
        }
        return cls(
            fabric_id=str(payload.get("fabric_id") or payload.get("id") or "unknown_fabric"),
            description=str(payload.get("description")) if payload.get("description") is not None else None,
            compliance=compliance,
            mdl_modifiers=mdl_modifiers,
            constraints=constraints,
        )


def _load_yaml_like(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return json.loads(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_fabric_profile(path: str | Path) -> FabricProfile:
    """Load a fabric profile from YAML/JSON."""

    fabric_path = Path(path)
    payload = _load_yaml_like(fabric_path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Fabric profile at '{fabric_path}' must be a mapping.")
    return FabricProfile.from_mapping(payload)


def load_fabrics_from_dir(directory: str | Path) -> Mapping[str, FabricProfile]:
    """Load all fabric profiles from a directory."""

    base = Path(directory)
    profiles: MutableMapping[str, FabricProfile] = {}
    for path in sorted(base.glob("*.yaml")):
        profile = load_fabric_profile(path)
        profiles[profile.fabric_id] = profile
    for path in sorted(base.glob("*.json")):
        profile = load_fabric_profile(path)
        profiles[profile.fabric_id] = profile
    if not profiles:
        raise FileNotFoundError(f"No fabric profiles found in '{base}'.")
    return profiles


def rotate_grain(grain: np.ndarray, rotation_deg: float, *, axis: np.ndarray | None = None) -> np.ndarray:
    """Rotate a grain vector around an axis (defaults to global z)."""

    vector = np.asarray(grain, dtype=float).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12 or abs(rotation_deg) <= 1e-9:
        return vector

    unit_grain = vector / norm
    axis_vec = np.asarray(axis if axis is not None else np.array([0.0, 0.0, 1.0]), dtype=float).reshape(-1)
    axis_norm = np.linalg.norm(axis_vec)
    if axis_norm <= 1e-12:
        return unit_grain
    axis_unit = axis_vec / axis_norm

    theta = np.deg2rad(rotation_deg)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    cross = np.cross(axis_unit, unit_grain)
    dot = float(np.dot(axis_unit, unit_grain))
    rotated = unit_grain * cos_t + cross * sin_t + axis_unit * dot * (1.0 - cos_t)
    return rotated / max(np.linalg.norm(rotated), 1e-12)


def fabric_penalty(edge_vector: np.ndarray, grain_vector: np.ndarray, profile: FabricProfile) -> float:
    """Compute a normalized penalty for cutting along an edge relative to fabric grain."""

    edge = np.asarray(edge_vector, dtype=float).reshape(-1)
    grain = np.asarray(grain_vector, dtype=float).reshape(-1)
    edge_norm = np.linalg.norm(edge)
    grain_norm = np.linalg.norm(grain)
    if edge_norm <= 1e-12 or grain_norm <= 1e-12:
        return 0.0

    edge_unit = edge / edge_norm
    grain_unit = grain / grain_norm
    alignment = abs(float(np.dot(edge_unit, grain_unit)))
    alignment = float(np.clip(alignment, 0.0, 1.0))
    if alignment >= 1.0 - 1e-6:
        return 0.0

    comp = profile.compliance
    max_compliance = max(comp.s_parallel, comp.s_perp, comp.s_shear, 1e-6)
    stretch_compliance = alignment * comp.s_parallel + (1.0 - alignment) * comp.s_perp
    stretch_penalty = 1.0 - min(stretch_compliance / max_compliance, 1.0)

    shear_component = 1.0 - alignment
    bias_multiplier = 1.0 if profile.constraints.allow_bias else 1.5
    shear_penalty = shear_component * (comp.s_shear / max_compliance) * bias_multiplier

    penalty = float(np.clip(stretch_penalty + shear_penalty, 0.0, 1.0))
    return penalty
