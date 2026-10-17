"""Fail-closed receipts from physical ROM closure to adaptive suit deployment.

These receipts deliberately sit *after* `PhysicalROMValidationReceipt`.  A suit
that matches its declared ROM/field model has not thereby proved that its
sensors, actuators, controller, power system, or deployment context are safe.

The authority order is:

    PhysicalROMValidationReceipt
      -> AdaptiveModuleReceipt
      -> ControllerSafetyReceipt
      -> SuitDeploymentReceipt

`SuitDeploymentReceipt` additionally requires an explicit external/manual
authority review gate.  No upstream theorem or measurement can synthesize that
approval automatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from .suit_closure_receipt import (
    PhysicalROMValidationReceipt,
    can_consume_physical_rom_validation_receipt,
)

Promotion = Literal[-1, 0, 1]

ADAPTIVE_MODULE_SCHEMA_VERSION = "smii.adaptive_module_receipt.v1"
CONTROLLER_SAFETY_SCHEMA_VERSION = "smii.controller_safety_receipt.v1"
SUIT_DEPLOYMENT_SCHEMA_VERSION = "smii.suit_deployment_receipt.v1"

ADAPTIVE_MODULE_CLAIM_BOUNDARY = (
    "module_receipt_checks_declared_hardware_envelope_not_general_device_safety"
)
CONTROLLER_SAFETY_CLAIM_BOUNDARY = (
    "controller_receipt_checks_declared_scenarios_not_universal_closed_loop_safety"
)
SUIT_DEPLOYMENT_CLAIM_BOUNDARY = (
    "deployment_requires_separate_authority_review_and_is_profile_specific"
)

__all__ = [
    "ADAPTIVE_MODULE_SCHEMA_VERSION",
    "CONTROLLER_SAFETY_SCHEMA_VERSION",
    "SUIT_DEPLOYMENT_SCHEMA_VERSION",
    "AdaptiveModuleReceipt",
    "ControllerSafetyReceipt",
    "SuitDeploymentReceipt",
    "derive_adaptive_module_receipt",
    "derive_controller_safety_receipt",
    "derive_suit_deployment_receipt",
    "can_consume_adaptive_module_receipt",
    "can_consume_controller_safety_receipt",
    "can_consume_suit_deployment_receipt",
    "load_adaptive_module_receipt",
    "load_controller_safety_receipt",
    "load_suit_deployment_receipt",
]


def _promotion(value: object) -> Promotion:
    if isinstance(value, bool):
        raise ValueError("promotion must be one of -1, 0, 1")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value in {"-1", "0", "1"}:
        result = int(value)
    elif isinstance(value, float) and value.is_integer():
        result = int(value)
    else:
        raise ValueError("promotion must be one of -1, 0, 1")
    if result not in {-1, 0, 1}:
        raise ValueError("promotion must be one of -1, 0, 1")
    return cast(Promotion, result)


def _required_str(value: object, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _non_negative_float(value: object, key: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{key} must be numeric")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{key} must be numeric") from exc
    if number < 0.0 or number != number or number in {float("inf"), float("-inf")}:
        raise ValueError(f"{key} must be finite and non-negative")
    return number


def _fraction(value: object, key: str) -> float:
    number = _non_negative_float(value, key)
    if number > 1.0:
        raise ValueError(f"{key} must lie in [0, 1]")
    return number


def _non_negative_int(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{key} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"{key} must be an integer")
    if number < 0:
        raise ValueError(f"{key} must be non-negative")
    return number


def _strings(value: object, key: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    result = [str(item) for item in value]
    if any(not item for item in result):
        raise ValueError(f"{key} may not contain empty strings")
    return result


def _blocked(promotion: Promotion, values: list[str], defaults: tuple[str, ...]) -> list[str]:
    return list(defaults) if promotion != 1 and not values else values


def _state(promotion: Promotion) -> str:
    return {1: "promoted", 0: "diagnostic_only", -1: "rejected"}[promotion]


@dataclass(frozen=True, slots=True)
class AdaptiveModuleReceipt:
    physical_rom_receipt_hash: str
    module_manifest_hash: str
    sensor_map_hash: str
    actuator_map_hash: str
    power_budget_hash: str
    fail_safe_evidence_hash: str
    module_count: int
    sensor_count: int
    actuator_count: int
    command_envelope_violation_rate: float
    command_envelope_violation_threshold: float
    fail_safe_test_count: int
    fail_safe_failure_count: int
    promotion: Promotion
    blocked_consumers: list[str]
    blocker_log: list[str]
    claim_boundary: str = ADAPTIVE_MODULE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        for key in (
            "physical_rom_receipt_hash",
            "module_manifest_hash",
            "sensor_map_hash",
            "actuator_map_hash",
            "power_budget_hash",
            "fail_safe_evidence_hash",
            "claim_boundary",
        ):
            object.__setattr__(self, key, _required_str(getattr(self, key), key))
        for key in ("module_count", "sensor_count", "actuator_count", "fail_safe_test_count", "fail_safe_failure_count"):
            object.__setattr__(self, key, _non_negative_int(getattr(self, key), key))
        object.__setattr__(
            self,
            "command_envelope_violation_rate",
            _fraction(self.command_envelope_violation_rate, "command_envelope_violation_rate"),
        )
        object.__setattr__(
            self,
            "command_envelope_violation_threshold",
            _fraction(
                self.command_envelope_violation_threshold,
                "command_envelope_violation_threshold",
            ),
        )
        promotion = _promotion(self.promotion)
        object.__setattr__(self, "promotion", promotion)
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked(
                promotion,
                _strings(self.blocked_consumers, "blocked_consumers"),
                ("controller_safety", "deployment"),
            ),
        )
        object.__setattr__(self, "blocker_log", _strings(self.blocker_log, "blocker_log"))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ADAPTIVE_MODULE_SCHEMA_VERSION,
            "physical_rom_receipt_hash": self.physical_rom_receipt_hash,
            "module_manifest_hash": self.module_manifest_hash,
            "sensor_map_hash": self.sensor_map_hash,
            "actuator_map_hash": self.actuator_map_hash,
            "power_budget_hash": self.power_budget_hash,
            "fail_safe_evidence_hash": self.fail_safe_evidence_hash,
            "module_count": self.module_count,
            "sensor_count": self.sensor_count,
            "actuator_count": self.actuator_count,
            "command_envelope_violation_rate": self.command_envelope_violation_rate,
            "command_envelope_violation_threshold": self.command_envelope_violation_threshold,
            "fail_safe_test_count": self.fail_safe_test_count,
            "fail_safe_failure_count": self.fail_safe_failure_count,
            "promotion": int(self.promotion),
            "promotion_state": _state(self.promotion),
            "blocked_consumers": list(self.blocked_consumers),
            "blocker_log": list(self.blocker_log),
            "claim_boundary": self.claim_boundary,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AdaptiveModuleReceipt":
        return cls(
            physical_rom_receipt_hash=_required_str(payload.get("physical_rom_receipt_hash"), "physical_rom_receipt_hash"),
            module_manifest_hash=_required_str(payload.get("module_manifest_hash"), "module_manifest_hash"),
            sensor_map_hash=_required_str(payload.get("sensor_map_hash"), "sensor_map_hash"),
            actuator_map_hash=_required_str(payload.get("actuator_map_hash"), "actuator_map_hash"),
            power_budget_hash=_required_str(payload.get("power_budget_hash"), "power_budget_hash"),
            fail_safe_evidence_hash=_required_str(payload.get("fail_safe_evidence_hash"), "fail_safe_evidence_hash"),
            module_count=_non_negative_int(payload.get("module_count"), "module_count"),
            sensor_count=_non_negative_int(payload.get("sensor_count"), "sensor_count"),
            actuator_count=_non_negative_int(payload.get("actuator_count"), "actuator_count"),
            command_envelope_violation_rate=_fraction(payload.get("command_envelope_violation_rate"), "command_envelope_violation_rate"),
            command_envelope_violation_threshold=_fraction(payload.get("command_envelope_violation_threshold"), "command_envelope_violation_threshold"),
            fail_safe_test_count=_non_negative_int(payload.get("fail_safe_test_count"), "fail_safe_test_count"),
            fail_safe_failure_count=_non_negative_int(payload.get("fail_safe_failure_count"), "fail_safe_failure_count"),
            promotion=_promotion(payload.get("promotion")),
            blocked_consumers=_strings(payload.get("blocked_consumers", []), "blocked_consumers"),
            blocker_log=_strings(payload.get("blocker_log", []), "blocker_log"),
            claim_boundary=_required_str(payload.get("claim_boundary", ADAPTIVE_MODULE_CLAIM_BOUNDARY), "claim_boundary"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "AdaptiveModuleReceipt":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("AdaptiveModuleReceipt JSON must contain an object")
        return cls.from_mapping(payload)

    def to_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


@dataclass(frozen=True, slots=True)
class ControllerSafetyReceipt:
    adaptive_module_receipt_hash: str
    controller_artifact_hash: str
    safety_envelope_hash: str
    scenario_manifest_hash: str
    controller_test_evidence_hash: str
    scenario_count: int
    hard_violation_count: int
    max_command_latency_ms: float
    command_latency_threshold_ms: float
    emergency_stop_test_count: int
    emergency_stop_failure_count: int
    promotion: Promotion
    blocked_consumers: list[str]
    blocker_log: list[str]
    claim_boundary: str = CONTROLLER_SAFETY_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        for key in (
            "adaptive_module_receipt_hash",
            "controller_artifact_hash",
            "safety_envelope_hash",
            "scenario_manifest_hash",
            "controller_test_evidence_hash",
            "claim_boundary",
        ):
            object.__setattr__(self, key, _required_str(getattr(self, key), key))
        for key in (
            "scenario_count",
            "hard_violation_count",
            "emergency_stop_test_count",
            "emergency_stop_failure_count",
        ):
            object.__setattr__(self, key, _non_negative_int(getattr(self, key), key))
        for key in ("max_command_latency_ms", "command_latency_threshold_ms"):
            object.__setattr__(self, key, _non_negative_float(getattr(self, key), key))
        promotion = _promotion(self.promotion)
        object.__setattr__(self, "promotion", promotion)
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked(
                promotion,
                _strings(self.blocked_consumers, "blocked_consumers"),
                ("deployment",),
            ),
        )
        object.__setattr__(self, "blocker_log", _strings(self.blocker_log, "blocker_log"))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CONTROLLER_SAFETY_SCHEMA_VERSION,
            "adaptive_module_receipt_hash": self.adaptive_module_receipt_hash,
            "controller_artifact_hash": self.controller_artifact_hash,
            "safety_envelope_hash": self.safety_envelope_hash,
            "scenario_manifest_hash": self.scenario_manifest_hash,
            "controller_test_evidence_hash": self.controller_test_evidence_hash,
            "scenario_count": self.scenario_count,
            "hard_violation_count": self.hard_violation_count,
            "max_command_latency_ms": self.max_command_latency_ms,
            "command_latency_threshold_ms": self.command_latency_threshold_ms,
            "emergency_stop_test_count": self.emergency_stop_test_count,
            "emergency_stop_failure_count": self.emergency_stop_failure_count,
            "promotion": int(self.promotion),
            "promotion_state": _state(self.promotion),
            "blocked_consumers": list(self.blocked_consumers),
            "blocker_log": list(self.blocker_log),
            "claim_boundary": self.claim_boundary,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ControllerSafetyReceipt":
        return cls(
            adaptive_module_receipt_hash=_required_str(payload.get("adaptive_module_receipt_hash"), "adaptive_module_receipt_hash"),
            controller_artifact_hash=_required_str(payload.get("controller_artifact_hash"), "controller_artifact_hash"),
            safety_envelope_hash=_required_str(payload.get("safety_envelope_hash"), "safety_envelope_hash"),
            scenario_manifest_hash=_required_str(payload.get("scenario_manifest_hash"), "scenario_manifest_hash"),
            controller_test_evidence_hash=_required_str(payload.get("controller_test_evidence_hash"), "controller_test_evidence_hash"),
            scenario_count=_non_negative_int(payload.get("scenario_count"), "scenario_count"),
            hard_violation_count=_non_negative_int(payload.get("hard_violation_count"), "hard_violation_count"),
            max_command_latency_ms=_non_negative_float(payload.get("max_command_latency_ms"), "max_command_latency_ms"),
            command_latency_threshold_ms=_non_negative_float(payload.get("command_latency_threshold_ms"), "command_latency_threshold_ms"),
            emergency_stop_test_count=_non_negative_int(payload.get("emergency_stop_test_count"), "emergency_stop_test_count"),
            emergency_stop_failure_count=_non_negative_int(payload.get("emergency_stop_failure_count"), "emergency_stop_failure_count"),
            promotion=_promotion(payload.get("promotion")),
            blocked_consumers=_strings(payload.get("blocked_consumers", []), "blocked_consumers"),
            blocker_log=_strings(payload.get("blocker_log", []), "blocker_log"),
            claim_boundary=_required_str(payload.get("claim_boundary", CONTROLLER_SAFETY_CLAIM_BOUNDARY), "claim_boundary"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ControllerSafetyReceipt":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("ControllerSafetyReceipt JSON must contain an object")
        return cls.from_mapping(payload)

    def to_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


@dataclass(frozen=True, slots=True)
class SuitDeploymentReceipt:
    controller_safety_receipt_hash: str
    deployment_profile_hash: str
    deployment_evidence_hash: str
    authority_review_receipt_hash: str | None
    authority_review_passed: bool | None
    promotion: Promotion
    blocked_consumers: list[str]
    blocker_log: list[str]
    claim_boundary: str = SUIT_DEPLOYMENT_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        for key in (
            "controller_safety_receipt_hash",
            "deployment_profile_hash",
            "deployment_evidence_hash",
            "claim_boundary",
        ):
            object.__setattr__(self, key, _required_str(getattr(self, key), key))
        if self.authority_review_receipt_hash is not None:
            object.__setattr__(
                self,
                "authority_review_receipt_hash",
                _required_str(self.authority_review_receipt_hash, "authority_review_receipt_hash"),
            )
        if self.authority_review_passed is not None and not isinstance(self.authority_review_passed, bool):
            raise TypeError("authority_review_passed must be boolean or null")
        promotion = _promotion(self.promotion)
        object.__setattr__(self, "promotion", promotion)
        object.__setattr__(
            self,
            "blocked_consumers",
            _blocked(
                promotion,
                _strings(self.blocked_consumers, "blocked_consumers"),
                ("field_deployment",),
            ),
        )
        object.__setattr__(self, "blocker_log", _strings(self.blocker_log, "blocker_log"))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SUIT_DEPLOYMENT_SCHEMA_VERSION,
            "controller_safety_receipt_hash": self.controller_safety_receipt_hash,
            "deployment_profile_hash": self.deployment_profile_hash,
            "deployment_evidence_hash": self.deployment_evidence_hash,
            "authority_review_receipt_hash": self.authority_review_receipt_hash,
            "authority_review_passed": self.authority_review_passed,
            "promotion": int(self.promotion),
            "promotion_state": _state(self.promotion),
            "blocked_consumers": list(self.blocked_consumers),
            "blocker_log": list(self.blocker_log),
            "claim_boundary": self.claim_boundary,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SuitDeploymentReceipt":
        authority_hash = payload.get("authority_review_receipt_hash")
        if authority_hash is not None:
            authority_hash = _required_str(authority_hash, "authority_review_receipt_hash")
        authority_passed = payload.get("authority_review_passed")
        if authority_passed is not None and not isinstance(authority_passed, bool):
            raise TypeError("authority_review_passed must be boolean or null")
        return cls(
            controller_safety_receipt_hash=_required_str(payload.get("controller_safety_receipt_hash"), "controller_safety_receipt_hash"),
            deployment_profile_hash=_required_str(payload.get("deployment_profile_hash"), "deployment_profile_hash"),
            deployment_evidence_hash=_required_str(payload.get("deployment_evidence_hash"), "deployment_evidence_hash"),
            authority_review_receipt_hash=cast(str | None, authority_hash),
            authority_review_passed=cast(bool | None, authority_passed),
            promotion=_promotion(payload.get("promotion")),
            blocked_consumers=_strings(payload.get("blocked_consumers", []), "blocked_consumers"),
            blocker_log=_strings(payload.get("blocker_log", []), "blocker_log"),
            claim_boundary=_required_str(payload.get("claim_boundary", SUIT_DEPLOYMENT_CLAIM_BOUNDARY), "claim_boundary"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SuitDeploymentReceipt":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("SuitDeploymentReceipt JSON must contain an object")
        return cls.from_mapping(payload)

    def to_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


def derive_adaptive_module_receipt(
    *,
    physical_rom_receipt_hash: str,
    physical_rom_receipt: PhysicalROMValidationReceipt,
    module_manifest_hash: str,
    sensor_map_hash: str,
    actuator_map_hash: str,
    power_budget_hash: str,
    fail_safe_evidence_hash: str,
    module_count: int,
    sensor_count: int,
    actuator_count: int,
    command_envelope_violation_rate: float,
    command_envelope_violation_threshold: float,
    fail_safe_test_count: int,
    fail_safe_failure_count: int,
) -> AdaptiveModuleReceipt:
    blockers: list[str] = []
    hard_reject = False
    if not can_consume_physical_rom_validation_receipt(physical_rom_receipt, "deployment"):
        blockers.append("physical_rom_validation_not_promoted")
    if module_count <= 0 or sensor_count <= 0 or actuator_count <= 0:
        blockers.append("adaptive_module_inventory_incomplete")
    if fail_safe_test_count <= 0:
        blockers.append("fail_safe_tests_missing")
    if command_envelope_violation_rate > command_envelope_violation_threshold:
        blockers.append("command_envelope_violation_rate_exceeds_threshold")
        hard_reject = True
    if fail_safe_failure_count > 0:
        blockers.append("fail_safe_test_failure")
        hard_reject = True
    promotion: Promotion = -1 if hard_reject else 0 if blockers else 1
    return AdaptiveModuleReceipt(
        physical_rom_receipt_hash=physical_rom_receipt_hash,
        module_manifest_hash=module_manifest_hash,
        sensor_map_hash=sensor_map_hash,
        actuator_map_hash=actuator_map_hash,
        power_budget_hash=power_budget_hash,
        fail_safe_evidence_hash=fail_safe_evidence_hash,
        module_count=module_count,
        sensor_count=sensor_count,
        actuator_count=actuator_count,
        command_envelope_violation_rate=command_envelope_violation_rate,
        command_envelope_violation_threshold=command_envelope_violation_threshold,
        fail_safe_test_count=fail_safe_test_count,
        fail_safe_failure_count=fail_safe_failure_count,
        promotion=promotion,
        blocked_consumers=[],
        blocker_log=blockers,
    )


def derive_controller_safety_receipt(
    *,
    adaptive_module_receipt_hash: str,
    adaptive_module_receipt: AdaptiveModuleReceipt,
    controller_artifact_hash: str,
    safety_envelope_hash: str,
    scenario_manifest_hash: str,
    controller_test_evidence_hash: str,
    scenario_count: int,
    hard_violation_count: int,
    max_command_latency_ms: float,
    command_latency_threshold_ms: float,
    emergency_stop_test_count: int,
    emergency_stop_failure_count: int,
) -> ControllerSafetyReceipt:
    blockers: list[str] = []
    hard_reject = False
    if not can_consume_adaptive_module_receipt(adaptive_module_receipt, "controller_safety"):
        blockers.append("adaptive_module_receipt_not_promoted")
    if scenario_count <= 0:
        blockers.append("controller_scenarios_missing")
    if emergency_stop_test_count <= 0:
        blockers.append("emergency_stop_tests_missing")
    if hard_violation_count > 0:
        blockers.append("controller_hard_safety_violation")
        hard_reject = True
    if max_command_latency_ms > command_latency_threshold_ms:
        blockers.append("controller_latency_exceeds_threshold")
        hard_reject = True
    if emergency_stop_failure_count > 0:
        blockers.append("emergency_stop_failure")
        hard_reject = True
    promotion: Promotion = -1 if hard_reject else 0 if blockers else 1
    return ControllerSafetyReceipt(
        adaptive_module_receipt_hash=adaptive_module_receipt_hash,
        controller_artifact_hash=controller_artifact_hash,
        safety_envelope_hash=safety_envelope_hash,
        scenario_manifest_hash=scenario_manifest_hash,
        controller_test_evidence_hash=controller_test_evidence_hash,
        scenario_count=scenario_count,
        hard_violation_count=hard_violation_count,
        max_command_latency_ms=max_command_latency_ms,
        command_latency_threshold_ms=command_latency_threshold_ms,
        emergency_stop_test_count=emergency_stop_test_count,
        emergency_stop_failure_count=emergency_stop_failure_count,
        promotion=promotion,
        blocked_consumers=[],
        blocker_log=blockers,
    )


def derive_suit_deployment_receipt(
    *,
    controller_safety_receipt_hash: str,
    controller_safety_receipt: ControllerSafetyReceipt,
    deployment_profile_hash: str,
    deployment_evidence_hash: str,
    authority_review_receipt_hash: str | None,
    authority_review_passed: bool | None,
) -> SuitDeploymentReceipt:
    blockers: list[str] = []
    hard_reject = False
    if not can_consume_controller_safety_receipt(controller_safety_receipt, "deployment"):
        blockers.append("controller_safety_receipt_not_promoted")
    if authority_review_passed is None or authority_review_receipt_hash is None:
        blockers.append("external_authority_review_missing")
    elif not authority_review_passed:
        blockers.append("external_authority_review_rejected")
        hard_reject = True
    promotion: Promotion = -1 if hard_reject else 0 if blockers else 1
    return SuitDeploymentReceipt(
        controller_safety_receipt_hash=controller_safety_receipt_hash,
        deployment_profile_hash=deployment_profile_hash,
        deployment_evidence_hash=deployment_evidence_hash,
        authority_review_receipt_hash=authority_review_receipt_hash,
        authority_review_passed=authority_review_passed,
        promotion=promotion,
        blocked_consumers=[],
        blocker_log=blockers,
    )


def can_consume_adaptive_module_receipt(
    receipt: AdaptiveModuleReceipt, consumer: str | None = None
) -> bool:
    return receipt.promotion == 1 and (
        not receipt.blocked_consumers if consumer is None else consumer not in receipt.blocked_consumers
    )


def can_consume_controller_safety_receipt(
    receipt: ControllerSafetyReceipt, consumer: str | None = None
) -> bool:
    return receipt.promotion == 1 and (
        not receipt.blocked_consumers if consumer is None else consumer not in receipt.blocked_consumers
    )


def can_consume_suit_deployment_receipt(
    receipt: SuitDeploymentReceipt, consumer: str | None = None
) -> bool:
    return receipt.promotion == 1 and (
        not receipt.blocked_consumers if consumer is None else consumer not in receipt.blocked_consumers
    )


def load_adaptive_module_receipt(path: str | Path) -> AdaptiveModuleReceipt:
    return AdaptiveModuleReceipt.from_json(path)


def load_controller_safety_receipt(path: str | Path) -> ControllerSafetyReceipt:
    return ControllerSafetyReceipt.from_json(path)


def load_suit_deployment_receipt(path: str | Path) -> SuitDeploymentReceipt:
    return SuitDeploymentReceipt.from_json(path)
