"""Lightweight PDA-style gate helpers for ROM and seam couplings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class CouplingRule:
    """Defines a coupling obligation or violation condition."""

    id: str
    description: str
    threshold: float = 0.0
    block: bool = True
    severity: str = "error"


@dataclass(frozen=True, slots=True)
class CouplingManifest:
    """Object map plus coupling rules loaded from disk."""

    object_map: Mapping[str, Mapping[str, str]]
    rules: tuple[CouplingRule, ...]


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Result of evaluating a set of coupling rules."""

    accepted: bool
    reasons: tuple["GateReason", ...]

    @property
    def blocking_reasons(self) -> tuple["GateReason", ...]:
        return tuple(reason for reason in self.reasons if reason.blocking)


@dataclass(frozen=True, slots=True)
class GateReason:
    """Structured reason for a gate decision."""

    id: str
    description: str
    severity: str
    blocking: bool


@dataclass(slots=True)
class GateRuntimeState:
    """Mutable runtime state for sequence-aware gate derivations."""

    evaluated_samples: int = 0
    seam_tear_damage: float = 0.0


class RomGate:
    """Evaluates coupling rules against observed seam/ROM metrics."""

    def __init__(
        self,
        rules: Sequence[CouplingRule],
        *,
        fatigue_floor: float = 0.35,
        fatigue_horizon: float = 8.0,
    ) -> None:
        self.rules = {rule.id: rule for rule in rules}
        self._fatigue_floor = max(0.0, min(float(fatigue_floor), 0.99))
        self._fatigue_horizon = max(float(fatigue_horizon), 1.0)

    def new_runtime_state(self) -> GateRuntimeState:
        """Create mutable state for stream-aware gate evaluation."""

        return GateRuntimeState()

    @staticmethod
    def _as_metric(value: float | bool | object) -> float | None:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @classmethod
    def _derive_observations(
        cls,
        observations: Mapping[str, float | bool],
    ) -> dict[str, float | bool]:
        derived: dict[str, float | bool] = dict(observations)
        if "seam_tear_risk" in derived:
            return derived

        risk = 0.0
        has_signal = False

        shear = cls._as_metric(derived.get("shear_hotspot", 0.0))
        if shear is not None:
            risk = max(risk, max(shear, 0.0))
            has_signal = has_signal or bool(shear > 0.0)

        pressure = cls._as_metric(derived.get("pressure_hotspot", 0.0))
        if pressure is not None:
            # Pressure is a secondary contributor to tear risk.
            risk = max(risk, max(pressure, 0.0) * 0.85)
            has_signal = has_signal or bool(pressure > 0.0)

        self_intersection = cls._as_metric(derived.get("edge_self_intersection", 0.0))
        if self_intersection is not None and self_intersection >= 1.0:
            risk = 1.0
            has_signal = True

        forbidden_hit = cls._as_metric(derived.get("forbidden_vertices", 0.0))
        if forbidden_hit is not None and forbidden_hit >= 1.0:
            risk = max(risk, 1.0)
            has_signal = True

        if has_signal:
            derived["seam_tear_risk"] = risk
        return derived

    def _derive_dynamic_tear_risk(
        self,
        observations: dict[str, float | bool],
        runtime_state: GateRuntimeState,
    ) -> None:
        tear_risk = self._as_metric(observations.get("seam_tear_risk", 0.0))
        risk_value = 0.0 if tear_risk is None else max(0.0, min(float(tear_risk), 1.0))
        normalized_excess = max(risk_value - self._fatigue_floor, 0.0) / max(
            1.0 - self._fatigue_floor,
            1e-6,
        )
        runtime_state.evaluated_samples += 1
        runtime_state.seam_tear_damage = min(
            1.0,
            runtime_state.seam_tear_damage + normalized_excess / self._fatigue_horizon,
        )
        observations["seam_tear_risk_dynamic"] = runtime_state.seam_tear_damage

    def prepare_observations(
        self,
        observations: Mapping[str, float | bool],
        *,
        runtime_state: GateRuntimeState | None = None,
    ) -> dict[str, float | bool]:
        """Normalize and derive gate observations before threshold checks."""

        prepared = self._derive_observations(observations)
        if runtime_state is not None:
            self._derive_dynamic_tear_risk(prepared, runtime_state)
        return prepared

    def evaluate(self, observations: Mapping[str, float | bool]) -> GateDecision:
        """Check observations against configured rules."""

        observed = self.prepare_observations(observations)
        accepted = True
        reasons: list[GateReason] = []
        for rule_id, rule in self.rules.items():
            if rule_id not in observed:
                continue
            value = observed[rule_id]
            triggered = bool(value) if isinstance(value, bool) else float(value) >= rule.threshold
            if not triggered:
                continue
            reasons.append(
                GateReason(
                    id=rule.id,
                    description=rule.description,
                    severity=rule.severity,
                    blocking=rule.block,
                )
            )
            if rule.block:
                accepted = False
        return GateDecision(accepted=accepted, reasons=tuple(reasons))


def load_coupling_manifest(path: str | Path) -> CouplingManifest:
    """Load coupling manifest JSON with object map and rules."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Coupling manifest '{manifest_path}' must be a JSON object.")

    raw_object_map = payload.get("object_map", {})
    if not isinstance(raw_object_map, Mapping):
        raw_object_map = {}
    object_map = {
        "objects": {str(k): str(v) for k, v in raw_object_map.get("objects", {}).items()}
        if isinstance(raw_object_map.get("objects"), Mapping)
        else {},
        "observations": {str(k): str(v) for k, v in raw_object_map.get("observations", {}).items()}
        if isinstance(raw_object_map.get("observations"), Mapping)
        else {},
    }

    rules_payload = payload.get("rules", [])
    if not isinstance(rules_payload, Sequence):
        raise TypeError("Coupling manifest 'rules' must be an array.")

    rules: list[CouplingRule] = []
    for entry in rules_payload:
        if not isinstance(entry, Mapping):
            continue
        rule_id = entry.get("id")
        description = entry.get("description", "")
        if not rule_id:
            continue
        rules.append(
            CouplingRule(
                id=str(rule_id),
                description=str(description),
                threshold=float(entry.get("threshold", 0.0)),
                block=bool(entry.get("block", True)),
                severity=str(entry.get("severity", "error")),
            )
        )

    if not rules:
        raise ValueError("Coupling manifest must define at least one rule.")

    return CouplingManifest(object_map=object_map, rules=tuple(rules))


def build_gate_from_manifest(manifest: CouplingManifest) -> RomGate:
    """Construct a ROM gate from a loaded coupling manifest."""

    return RomGate(manifest.rules)
