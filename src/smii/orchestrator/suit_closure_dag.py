"""Run-directory reader for the physical ROM-to-suit closure.

This module composes the existing digital receipt DAG with the post-pattern
physical evidence chain without changing the older reader's semantics.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from smii.seams.seam_derivation import FinishedSeamReceipt, load_finished_seam_receipt
from smii.seams.suit_closure_receipt import (
    AssemblyReceipt,
    PhysicalROMValidationReceipt,
    WearFitReceipt,
    load_assembly_receipt,
    load_physical_rom_validation_receipt,
    load_wear_fit_receipt,
)

from .receipt_dag import ReceiptDagState, read_receipt_dag

FINISHED_SEAM_RECEIPT = "finished_seam_receipt.json"
ASSEMBLY_RECEIPT = "assembly_receipt.json"
WEAR_FIT_RECEIPT = "wear_fit_receipt.json"
PHYSICAL_ROM_VALIDATION_RECEIPT = "physical_rom_validation_receipt.json"

__all__ = [
    "ASSEMBLY_RECEIPT",
    "FINISHED_SEAM_RECEIPT",
    "PHYSICAL_ROM_VALIDATION_RECEIPT",
    "WEAR_FIT_RECEIPT",
    "SuitClosureDagState",
    "read_suit_closure_dag",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_finished(root: Path) -> FinishedSeamReceipt | None:
    path = root / FINISHED_SEAM_RECEIPT
    return load_finished_seam_receipt(path) if path.exists() else None


def _load_assembly(root: Path) -> AssemblyReceipt | None:
    path = root / ASSEMBLY_RECEIPT
    return load_assembly_receipt(path) if path.exists() else None


def _load_wear_fit(root: Path) -> WearFitReceipt | None:
    path = root / WEAR_FIT_RECEIPT
    return load_wear_fit_receipt(path) if path.exists() else None


def _load_physical_rom(root: Path) -> PhysicalROMValidationReceipt | None:
    path = root / PHYSICAL_ROM_VALIDATION_RECEIPT
    return load_physical_rom_validation_receipt(path) if path.exists() else None


@dataclass(frozen=True, slots=True)
class SuitClosureDagState:
    """Promotion snapshot from digital garment through physical ROM validation."""

    digital: ReceiptDagState
    finished_seam: int
    assembly: int
    wear_fit: int
    physical_rom: int
    first_blocker: str | None
    finished_seam_receipt: FinishedSeamReceipt | None = None
    assembly_receipt: AssemblyReceipt | None = None
    wear_fit_receipt: WearFitReceipt | None = None
    physical_rom_receipt: PhysicalROMValidationReceipt | None = None
    hash_chain_errors: tuple[str, ...] = ()

    def can_assemble(self) -> bool:
        return (
            self.digital.can_manufacture()
            and self.digital.hash_chain_valid()
            and self.finished_seam == 1
            and not self.hash_chain_errors
        )

    def can_wear_test(self) -> bool:
        return self.can_assemble() and self.assembly == 1

    def can_validate_physical_rom(self) -> bool:
        return self.can_wear_test() and self.wear_fit == 1

    def can_deploy(self) -> bool:
        """Return only receipt eligibility; this is not a safety certification."""

        return self.can_validate_physical_rom() and self.physical_rom == 1

    def hash_chain_valid(self) -> bool:
        return self.digital.hash_chain_valid() and not self.hash_chain_errors


def _closure_hash_errors(
    root: Path,
    assembly: AssemblyReceipt | None,
    wear_fit: WearFitReceipt | None,
    physical_rom: PhysicalROMValidationReceipt | None,
) -> tuple[str, ...]:
    errors: list[str] = []

    finished_path = root / FINISHED_SEAM_RECEIPT
    assembly_path = root / ASSEMBLY_RECEIPT
    wear_fit_path = root / WEAR_FIT_RECEIPT

    if assembly is not None and finished_path.exists():
        actual = _sha256_file(finished_path)
        if assembly.finished_seam_receipt_hash != actual:
            errors.append(
                "assembly.finished_seam_receipt_hash mismatch: "
                f"expected {assembly.finished_seam_receipt_hash}, actual {actual}"
            )

    if wear_fit is not None and assembly_path.exists():
        actual = _sha256_file(assembly_path)
        if wear_fit.assembly_receipt_hash != actual:
            errors.append(
                "wear_fit.assembly_receipt_hash mismatch: "
                f"expected {wear_fit.assembly_receipt_hash}, actual {actual}"
            )

    if physical_rom is not None and wear_fit_path.exists():
        actual = _sha256_file(wear_fit_path)
        if physical_rom.wear_fit_receipt_hash != actual:
            errors.append(
                "physical_rom.wear_fit_receipt_hash mismatch: "
                f"expected {physical_rom.wear_fit_receipt_hash}, actual {actual}"
            )

    return tuple(errors)


def _first_blocker(
    digital: ReceiptDagState,
    finished: FinishedSeamReceipt | None,
    assembly: AssemblyReceipt | None,
    wear_fit: WearFitReceipt | None,
    physical_rom: PhysicalROMValidationReceipt | None,
    hash_errors: tuple[str, ...],
) -> str | None:
    if digital.first_blocker is not None:
        return digital.first_blocker
    if not digital.hash_chain_valid():
        return "digital_hash_chain"
    if not digital.can_manufacture():
        return "manufacture"
    if finished is None or finished.promotion != 1:
        return "finished_seam"
    if hash_errors:
        return "physical_hash_chain"
    if assembly is None or assembly.promotion != 1:
        return "assembly"
    if wear_fit is None or wear_fit.promotion != 1:
        return "wear_fit"
    if physical_rom is None or physical_rom.promotion != 1:
        return "physical_rom"
    return None


def read_suit_closure_dag(
    run_dir: str | Path,
    *,
    solve_domain: str | None = None,
) -> SuitClosureDagState:
    """Read the full digital-to-physical promotion chain from ``run_dir``."""

    root = Path(run_dir)
    digital = read_receipt_dag(root, solve_domain=solve_domain)
    finished = _load_finished(root)
    assembly = _load_assembly(root)
    wear_fit = _load_wear_fit(root)
    physical_rom = _load_physical_rom(root)
    hash_errors = _closure_hash_errors(root, assembly, wear_fit, physical_rom)

    return SuitClosureDagState(
        digital=digital,
        finished_seam=finished.promotion if finished is not None else 0,
        assembly=assembly.promotion if assembly is not None else 0,
        wear_fit=wear_fit.promotion if wear_fit is not None else 0,
        physical_rom=physical_rom.promotion if physical_rom is not None else 0,
        first_blocker=_first_blocker(
            digital,
            finished,
            assembly,
            wear_fit,
            physical_rom,
            hash_errors,
        ),
        finished_seam_receipt=finished,
        assembly_receipt=assembly,
        wear_fit_receipt=wear_fit,
        physical_rom_receipt=physical_rom,
        hash_chain_errors=hash_errors,
    )
