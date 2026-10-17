#!/usr/bin/env python3
"""Emit adaptive-module, controller-safety, and deployment receipts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

from smii.seams.adaptive_suit_receipt import (
    derive_adaptive_module_receipt,
    derive_controller_safety_receipt,
    derive_suit_deployment_receipt,
    load_adaptive_module_receipt,
    load_controller_safety_receipt,
)
from smii.seams.suit_closure_receipt import load_physical_rom_validation_receipt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit_adaptive(args: argparse.Namespace) -> None:
    physical = load_physical_rom_validation_receipt(args.physical_rom_receipt)
    receipt = derive_adaptive_module_receipt(
        physical_rom_receipt_hash=_sha256(args.physical_rom_receipt),
        physical_rom_receipt=physical,
        module_manifest_hash=_sha256(args.module_manifest),
        sensor_map_hash=_sha256(args.sensor_map),
        actuator_map_hash=_sha256(args.actuator_map),
        power_budget_hash=_sha256(args.power_budget),
        fail_safe_evidence_hash=_sha256(args.fail_safe_evidence),
        module_count=args.module_count,
        sensor_count=args.sensor_count,
        actuator_count=args.actuator_count,
        command_envelope_violation_rate=args.command_envelope_violation_rate,
        command_envelope_violation_threshold=args.command_envelope_violation_threshold,
        fail_safe_test_count=args.fail_safe_test_count,
        fail_safe_failure_count=args.fail_safe_failure_count,
    )
    receipt.to_json(args.out)
    print(f"Wrote AdaptiveModuleReceipt to {args.out} ({receipt.promotion})")


def emit_controller(args: argparse.Namespace) -> None:
    adaptive = load_adaptive_module_receipt(args.adaptive_module_receipt)
    receipt = derive_controller_safety_receipt(
        adaptive_module_receipt_hash=_sha256(args.adaptive_module_receipt),
        adaptive_module_receipt=adaptive,
        controller_artifact_hash=_sha256(args.controller_artifact),
        safety_envelope_hash=_sha256(args.safety_envelope),
        scenario_manifest_hash=_sha256(args.scenario_manifest),
        controller_test_evidence_hash=_sha256(args.controller_test_evidence),
        scenario_count=args.scenario_count,
        hard_violation_count=args.hard_violation_count,
        max_command_latency_ms=args.max_command_latency_ms,
        command_latency_threshold_ms=args.command_latency_threshold_ms,
        emergency_stop_test_count=args.emergency_stop_test_count,
        emergency_stop_failure_count=args.emergency_stop_failure_count,
    )
    receipt.to_json(args.out)
    print(f"Wrote ControllerSafetyReceipt to {args.out} ({receipt.promotion})")


def emit_deployment(args: argparse.Namespace) -> None:
    controller = load_controller_safety_receipt(args.controller_safety_receipt)
    review_hash = _sha256(args.authority_review_receipt) if args.authority_review_receipt else None
    passed = None if args.authority_review is None else args.authority_review == "pass"
    receipt = derive_suit_deployment_receipt(
        controller_safety_receipt_hash=_sha256(args.controller_safety_receipt),
        controller_safety_receipt=controller,
        deployment_profile_hash=_sha256(args.deployment_profile),
        deployment_evidence_hash=_sha256(args.deployment_evidence),
        authority_review_receipt_hash=review_hash,
        authority_review_passed=passed,
    )
    receipt.to_json(args.out)
    print(f"Wrote SuitDeploymentReceipt to {args.out} ({receipt.promotion})")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)

    adaptive = subs.add_parser("adaptive-module")
    adaptive.add_argument("--physical-rom-receipt", type=Path, required=True)
    adaptive.add_argument("--module-manifest", type=Path, required=True)
    adaptive.add_argument("--sensor-map", type=Path, required=True)
    adaptive.add_argument("--actuator-map", type=Path, required=True)
    adaptive.add_argument("--power-budget", type=Path, required=True)
    adaptive.add_argument("--fail-safe-evidence", type=Path, required=True)
    adaptive.add_argument("--module-count", type=int, required=True)
    adaptive.add_argument("--sensor-count", type=int, required=True)
    adaptive.add_argument("--actuator-count", type=int, required=True)
    adaptive.add_argument("--command-envelope-violation-rate", type=float, required=True)
    adaptive.add_argument("--command-envelope-violation-threshold", type=float, required=True)
    adaptive.add_argument("--fail-safe-test-count", type=int, required=True)
    adaptive.add_argument("--fail-safe-failure-count", type=int, required=True)
    adaptive.add_argument("--out", type=Path, required=True)
    adaptive.set_defaults(func=emit_adaptive)

    controller = subs.add_parser("controller-safety")
    controller.add_argument("--adaptive-module-receipt", type=Path, required=True)
    controller.add_argument("--controller-artifact", type=Path, required=True)
    controller.add_argument("--safety-envelope", type=Path, required=True)
    controller.add_argument("--scenario-manifest", type=Path, required=True)
    controller.add_argument("--controller-test-evidence", type=Path, required=True)
    controller.add_argument("--scenario-count", type=int, required=True)
    controller.add_argument("--hard-violation-count", type=int, required=True)
    controller.add_argument("--max-command-latency-ms", type=float, required=True)
    controller.add_argument("--command-latency-threshold-ms", type=float, required=True)
    controller.add_argument("--emergency-stop-test-count", type=int, required=True)
    controller.add_argument("--emergency-stop-failure-count", type=int, required=True)
    controller.add_argument("--out", type=Path, required=True)
    controller.set_defaults(func=emit_controller)

    deployment = subs.add_parser("deployment")
    deployment.add_argument("--controller-safety-receipt", type=Path, required=True)
    deployment.add_argument("--deployment-profile", type=Path, required=True)
    deployment.add_argument("--deployment-evidence", type=Path, required=True)
    deployment.add_argument("--authority-review-receipt", type=Path, default=None)
    deployment.add_argument("--authority-review", choices=("pass", "reject"), default=None)
    deployment.add_argument("--out", type=Path, required=True)
    deployment.set_defaults(func=emit_deployment)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
