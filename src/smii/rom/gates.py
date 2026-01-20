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


class RomGate:
    """Evaluates coupling rules against observed seam/ROM metrics."""

    def __init__(self, rules: Sequence[CouplingRule]) -> None:
        self.rules = {rule.id: rule for rule in rules}

    def evaluate(self, observations: Mapping[str, float | bool]) -> GateDecision:
        """Check observations against configured rules."""

        accepted = True
        reasons: list[GateReason] = []
        for rule_id, rule in self.rules.items():
            if rule_id not in observations:
                continue
            value = observations[rule_id]
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
