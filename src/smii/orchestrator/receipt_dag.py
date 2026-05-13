"""Minimal reader for receipt promotion state in a run directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from smii.meshing.body_carrier_receipt import (
    BodyCarrierReceipt,
    load_body_carrier_receipt,
)
from smii.meshing.correspondence_receipt import (
    CorrespondenceReceipt,
    load_correspondence_receipt,
)
from smii.rom.basis_receipt import BasisReceipt, load_basis_receipt
from smii.rom.rom_field_receipt import ROMFieldReceipt, load_rom_field_receipt
from smii.seams.seam_cost_receipt import SeamCostReceipt, load_seam_cost_receipt

Promotion = Literal[-1, 0, 1]

BODY_RECEIPT = "body_carrier_receipt.json"
CORRESPONDENCE_RECEIPT = "correspondence_receipt.json"
TRANSFORM_RECEIPT = "transform_receipt.json"
BASIS_RECEIPT = "basis_receipt.json"
ROM_FIELD_RECEIPT = "rom_field_receipt.json"
SEAM_COST_RECEIPT = "seam_cost_receipt.json"

FUTURE_GATES = (
    "solver",
    "panel",
    "manufacture",
)

_BLOCKER_ORDER = (
    "body",
    "basis",
    "rom_field",
    "correspondence",
    "seam_cost",
    "solver",
    "panel",
    "manufacture",
)

__all__ = [
    "BASIS_RECEIPT",
    "BODY_RECEIPT",
    "CORRESPONDENCE_RECEIPT",
    "FUTURE_GATES",
    "ReceiptDagState",
    "ROM_FIELD_RECEIPT",
    "SEAM_COST_RECEIPT",
    "TRANSFORM_RECEIPT",
    "read_receipt_dag",
]


@dataclass(frozen=True, slots=True)
class ReceiptDagState:
    """Promotion snapshot read from receipt files in a run directory."""

    run_dir: Path
    solve_domain: str | None
    body: Promotion
    correspondence: Promotion
    basis: Promotion
    rom_field: Promotion = 0
    seam_cost: Promotion = 0
    solver: Promotion = 0
    panel: Promotion = 0
    manufacture: Promotion = 0
    first_blocker: str | None = None
    body_receipt: BodyCarrierReceipt | None = None
    correspondence_receipt: CorrespondenceReceipt | None = None
    basis_receipt: BasisReceipt | None = None
    rom_field_receipt: ROMFieldReceipt | None = None
    seam_cost_receipt: SeamCostReceipt | None = None

    def is_solver_eligible(self) -> bool:
        """Return whether the state satisfies the solver promotion precondition."""

        return (
            self.body == 1
            and self.basis == 1
            and self.rom_field == 1
            and self.seam_cost == 1
            and (self.correspondence == 1 or self.solve_domain == "A_v3240")
        )


def read_receipt_dag(
    run_dir: str | Path,
    *,
    solve_domain: str | None = None,
    rom_field: Promotion = 0,
    seam_cost: Promotion = 0,
    solver: Promotion = 0,
    panel: Promotion = 0,
    manufacture: Promotion = 0,
) -> ReceiptDagState:
    """Read receipt promotions from a run directory without running tasks."""

    root = Path(run_dir)

    body_receipt = _load_body_receipt(root)
    correspondence_receipt = _load_correspondence_receipt(root)
    basis_receipt = _load_basis_receipt(root)
    rom_field_receipt = _load_rom_field_receipt(root)
    seam_cost_receipt = _load_seam_cost_receipt(root)

    promotions = {
        "body": _promotion_or_zero(body_receipt),
        "correspondence": _promotion_or_zero(correspondence_receipt),
        "basis": _promotion_or_zero(basis_receipt),
        "rom_field": _promotion_or_override(rom_field_receipt, rom_field),
        "seam_cost": _promotion_or_override(seam_cost_receipt, seam_cost),
        "solver": solver,
        "panel": panel,
        "manufacture": manufacture,
    }

    return ReceiptDagState(
        run_dir=root,
        solve_domain=solve_domain,
        body=promotions["body"],
        correspondence=promotions["correspondence"],
        basis=promotions["basis"],
        rom_field=promotions["rom_field"],
        seam_cost=promotions["seam_cost"],
        solver=promotions["solver"],
        panel=promotions["panel"],
        manufacture=promotions["manufacture"],
        first_blocker=_first_blocker(promotions, solve_domain),
        body_receipt=body_receipt,
        correspondence_receipt=correspondence_receipt,
        basis_receipt=basis_receipt,
        rom_field_receipt=rom_field_receipt,
        seam_cost_receipt=seam_cost_receipt,
    )


def _load_body_receipt(run_dir: Path) -> BodyCarrierReceipt | None:
    path = run_dir / BODY_RECEIPT
    if not path.exists():
        return None
    return load_body_carrier_receipt(path)


def _load_correspondence_receipt(run_dir: Path) -> CorrespondenceReceipt | None:
    path = run_dir / CORRESPONDENCE_RECEIPT
    if not path.exists():
        path = run_dir / TRANSFORM_RECEIPT
    if not path.exists():
        return None
    return load_correspondence_receipt(path)


def _load_basis_receipt(run_dir: Path) -> BasisReceipt | None:
    path = run_dir / BASIS_RECEIPT
    if not path.exists():
        return None
    return load_basis_receipt(path)


def _load_rom_field_receipt(run_dir: Path) -> ROMFieldReceipt | None:
    path = run_dir / ROM_FIELD_RECEIPT
    if not path.exists():
        return None
    return load_rom_field_receipt(path)


def _load_seam_cost_receipt(run_dir: Path) -> SeamCostReceipt | None:
    path = run_dir / SEAM_COST_RECEIPT
    if not path.exists():
        return None
    return load_seam_cost_receipt(path)


def _promotion_or_zero(
    receipt: BodyCarrierReceipt
    | CorrespondenceReceipt
    | BasisReceipt
    | ROMFieldReceipt
    | SeamCostReceipt
    | None,
) -> Promotion:
    if receipt is None:
        return 0
    return receipt.promotion


def _promotion_or_override(
    receipt: ROMFieldReceipt | SeamCostReceipt | None,
    override: Promotion,
) -> Promotion:
    if receipt is not None:
        return receipt.promotion
    return override


def _first_blocker(
    promotions: dict[str, Promotion],
    solve_domain: str | None,
) -> str | None:
    for gate in _BLOCKER_ORDER:
        if gate == "correspondence" and solve_domain == "A_v3240":
            continue
        if promotions[gate] != 1:
            return gate
    return None
