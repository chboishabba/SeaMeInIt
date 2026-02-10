"""Task profile loader and task-weighted ROM aggregation hook."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from smii.rom.aggregation import RomAggregation, RomSample, aggregate_fields
from smii.rom.basis import KernelProjector
from smii.rom.gates import RomGate

__all__ = [
    "TaskAggregationConfig",
    "TaskMixtureComponent",
    "TaskProfile",
    "aggregate_rom_for_task",
    "load_task_profile",
]


@dataclass(frozen=True, slots=True)
class TaskMixtureComponent:
    """Weighted component within a task profile."""

    name: str
    weight: float
    source: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TaskAggregationConfig:
    """Aggregation-time options for a task profile."""

    statistic: str = "mean"
    temperature: float = 1.0
    per_sweep_normalize: bool = False


@dataclass(frozen=True, slots=True)
class TaskProfile:
    """Task definition used to weight ROM aggregation."""

    task_id: str
    description: str | None
    mixture: tuple[TaskMixtureComponent, ...]
    aggregation: TaskAggregationConfig

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TaskProfile":
        mixture: list[TaskMixtureComponent] = []
        for entry in payload.get("mixture", []):
            if not isinstance(entry, Mapping):
                continue
            name = str(entry.get("name", f"component_{len(mixture)}"))
            weight = float(entry.get("weight", 1.0))
            source = entry.get("source", {})
            mixture.append(TaskMixtureComponent(name=name, weight=weight, source=source if isinstance(source, Mapping) else {}))

        aggregation_payload = payload.get("aggregation", {})
        aggregation = TaskAggregationConfig(
            statistic=str(aggregation_payload.get("statistic", "mean")),
            temperature=float(aggregation_payload.get("temperature", 1.0)),
            per_sweep_normalize=bool(aggregation_payload.get("per_sweep_normalize", False)),
        )
        return cls(
            task_id=str(payload.get("task_id", "task_profile")),
            description=str(payload.get("description")) if payload.get("description") is not None else None,
            mixture=tuple(mixture),
            aggregation=aggregation,
        )

    @property
    def normalized_weights(self) -> Mapping[str, float]:
        if not self.mixture:
            return {}
        total = float(sum(max(component.weight, 0.0) for component in self.mixture))
        if total <= 0.0:
            total = float(len(self.mixture))
        return {component.name: max(component.weight, 0.0) / total for component in self.mixture}

    def weight_for_sample(self, sample: RomSample) -> float:
        """Return the normalized weight for a sample based on its observations."""

        if not self.mixture:
            return 1.0
        observations = sample.observations or {}
        component_name = None
        for key in ("task_component", "component", "mixture_component", "task"):
            if key in observations:
                component_name = str(observations[key])
                break
        weights = self.normalized_weights
        base_weight = weights.get(component_name, 1.0 / max(len(self.mixture), 1))
        temp = max(self.aggregation.temperature, 1e-6)
        adjusted = float(np.power(base_weight, 1.0 / temp))
        if "task_weight" in observations:
            try:
                adjusted *= float(observations["task_weight"])
            except Exception:
                pass
        return adjusted


def _load_yaml_like(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return json.loads(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_task_profile(path: str | Path) -> TaskProfile:
    """Load a task profile YAML/JSON document."""

    profile_path = Path(path)
    payload = _load_yaml_like(profile_path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Task profile at '{profile_path}' must be a mapping.")
    return TaskProfile.from_mapping(payload)


def _normalize_weights(weight_map: Mapping[str, float]) -> Mapping[str, float]:
    if not weight_map:
        return weight_map
    total = float(sum(weight_map.values()))
    if total <= 0.0:
        return {key: 1.0 for key in weight_map}
    scale = len(weight_map) / total
    return {key: value * scale for key, value in weight_map.items()}


def aggregate_rom_for_task(
    samples: Iterable[RomSample],
    projector: KernelProjector,
    task_profile: TaskProfile,
    *,
    field_keys: Sequence[str] | None = None,
    optional_fields: Sequence[str] | None = None,
    include_observed_fields: bool = True,
    edges: Sequence[tuple[int, int]] | None = None,
    gate: RomGate | None = None,
    diagnostics_top_k: int = 5,
) -> RomAggregation:
    """Aggregate ROM samples using task-defined weights."""

    sample_list = list(samples)
    weight_map = {
        sample.pose_id: task_profile.weight_for_sample(sample)
        for sample in sample_list
    }
    normalized_weights = _normalize_weights(weight_map)
    return aggregate_fields(
        sample_list,
        projector,
        field_keys=field_keys,
        optional_fields=optional_fields,
        include_observed_fields=include_observed_fields,
        edges=edges,
        gate=gate,  # type: ignore[arg-type]
        diagnostics_top_k=diagnostics_top_k,
        sample_weights=normalized_weights,
    )
