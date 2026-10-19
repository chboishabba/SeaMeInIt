"""Receipt DAG from physical ROM validation through adaptive deployment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from smii.seams.adaptive_suit_receipt import (
    AdaptiveModuleReceipt,
    ControllerSafetyReceipt,
    SuitDeploymentReceipt,
    load_adaptive_module_receipt,
    load_controller_safety_receipt,
    load_suit_deployment_receipt,
)

from .suit_closure_dag import (
    PHYSICAL_ROM_VALIDATION_RECEIPT,
    SuitClosureDagState,
    read_suit_closure_dag,
)

ADAPTIVE_MODULE_RECEIPT = "adaptive_module_receipt.json"
CONTROLLER_SAFETY_RECEIPT = "controller_safety_receipt.json"
SUIT_DEPLOYMENT_RECEIPT = "suit_deployment_receipt.json"

__all__ = [
    "ADAPTIVE_MODULE_RECEIPT",
    "CONTROLLER_SAFETY_RECEIPT",
    "SUIT_DEPLOYMENT_RECEIPT",
    "AdaptiveSuitDagState",
    "read_adaptive_suit_dag",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_optional(path: Path, loader):
    return loader(path) if path.exists() else None


@dataclass(frozen=True, slots=True)
class AdaptiveSuitDagState:
    physical: SuitClosureDagState
    adaptive_module: int
    controller_safety: int
    deployment: int
    first_blocker: str | None
    adaptive_module_receipt: AdaptiveModuleReceipt | None = None
    controller_safety_receipt: ControllerSafetyReceipt | None = None
    deployment_receipt: SuitDeploymentReceipt | None = None
    hash_chain_errors: tuple[str, ...] = ()

    def hash_chain_valid(self) -> bool:
        return self.physical.hash_chain_valid() and not self.hash_chain_errors

    def can_validate_controller(self) -> bool:
        return (
            self.physical.can_validate_physical_rom()
            and self.physical.physical_rom == 1
            and self.adaptive_module == 1
            and self.hash_chain_valid()
        )

    def can_request_deployment_authority(self) -> bool:
        return self.can_validate_controller() and self.controller_safety == 1

    def can_field_deploy(self) -> bool:
        return self.can_request_deployment_authority() and self.deployment == 1


def _hash_errors(
    root: Path,
    adaptive: AdaptiveModuleReceipt | None,
    controller: ControllerSafetyReceipt | None,
    deployment: SuitDeploymentReceipt | None,
) -> tuple[str, ...]:
    errors: list[str] = []
    physical_path = root / PHYSICAL_ROM_VALIDATION_RECEIPT
    adaptive_path = root / ADAPTIVE_MODULE_RECEIPT
    controller_path = root / CONTROLLER_SAFETY_RECEIPT

    if adaptive is not None and physical_path.exists():
        actual = _sha256(physical_path)
        if adaptive.physical_rom_receipt_hash != actual:
            errors.append(
                "adaptive_module.physical_rom_receipt_hash mismatch: "
                f"expected {adaptive.physical_rom_receipt_hash}, actual {actual}"
            )
    if controller is not None and adaptive_path.exists():
        actual = _sha256(adaptive_path)
        if controller.adaptive_module_receipt_hash != actual:
            errors.append(
                "controller_safety.adaptive_module_receipt_hash mismatch: "
                f"expected {controller.adaptive_module_receipt_hash}, actual {actual}"
            )
    if deployment is not None and controller_path.exists():
        actual = _sha256(controller_path)
        if deployment.controller_safety_receipt_hash != actual:
            errors.append(
                "deployment.controller_safety_receipt_hash mismatch: "
                f"expected {deployment.controller_safety_receipt_hash}, actual {actual}"
            )
    return tuple(errors)


def _first_blocker(
    physical: SuitClosureDagState,
    adaptive: AdaptiveModuleReceipt | None,
    controller: ControllerSafetyReceipt | None,
    deployment: SuitDeploymentReceipt | None,
    errors: tuple[str, ...],
) -> str | None:
    if physical.first_blocker is not None:
        return physical.first_blocker
    if physical.physical_rom != 1:
        return "physical_rom"
    if errors:
        return "adaptive_hash_chain"
    if adaptive is None or adaptive.promotion != 1:
        return "adaptive_module"
    if controller is None or controller.promotion != 1:
        return "controller_safety"
    if deployment is None or deployment.promotion != 1:
        return "deployment_authority"
    return None


def read_adaptive_suit_dag(
    run_dir: str | Path,
    *,
    solve_domain: str | None = None,
) -> AdaptiveSuitDagState:
    root = Path(run_dir)
    physical = read_suit_closure_dag(root, solve_domain=solve_domain)
    adaptive = _load_optional(root / ADAPTIVE_MODULE_RECEIPT, load_adaptive_module_receipt)
    controller = _load_optional(
        root / CONTROLLER_SAFETY_RECEIPT,
        load_controller_safety_receipt,
    )
    deployment = _load_optional(root / SUIT_DEPLOYMENT_RECEIPT, load_suit_deployment_receipt)
    errors = _hash_errors(root, adaptive, controller, deployment)
    return AdaptiveSuitDagState(
        physical=physical,
        adaptive_module=adaptive.promotion if adaptive is not None else 0,
        controller_safety=controller.promotion if controller is not None else 0,
        deployment=deployment.promotion if deployment is not None else 0,
        first_blocker=_first_blocker(physical, adaptive, controller, deployment, errors),
        adaptive_module_receipt=adaptive,
        controller_safety_receipt=controller,
        deployment_receipt=deployment,
        hash_chain_errors=errors,
    )
