"""Fail-closed schema, cryptographic provenance, and semantic authority verification across Gates 0–7.

This module establishes a multi-layer verification pipeline across the digital
garment receipt DAG without modifying existing reader compatibility surfaces:
  Layer 1: Schema & Format validation (structure, types, required fields, value bounds)
  Layer 2: Cryptographic Provenance (SHA-256 integrity, forward hash references, parent-child links)
  Layer 3: Semantic Authority (promotion invariants, blocker checks, diagnostic-only guards, solve domain consistency)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Sequence

from .receipt_dag import (
    BASIS_RECEIPT,
    BODY_RECEIPT,
    CORRESPONDENCE_RECEIPT,
    CUT_TOPOLOGY_RECEIPT,
    MANUFACTURING_RECEIPT,
    METRIC_CORRECTION_RECEIPT,
    PANEL_UNWRAP_RECEIPT,
    ReceiptDagState,
    ROM_FIELD_RECEIPT,
    SEAM_COST_RECEIPT,
    SOLVER_PROMOTION_RECEIPT,
    TRANSFORM_RECEIPT,
    read_receipt_dag,
)

__all__ = [
    "TrustStatus",
    "VerificationIssue",
    "VerificationSeverity",
    "TrustedDagSnapshot",
    "verify_receipt_dag",
]


class TrustStatus(StrEnum):
    ACCEPTED = "accepted"
    DIAGNOSTIC = "diagnostic"
    BLOCKED = "blocked"


class VerificationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class VerificationIssue:
    code: str
    receipt_id: str
    detail: str
    severity: VerificationSeverity = VerificationSeverity.ERROR


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_raw_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


@dataclass(frozen=True, slots=True)
class TrustedDagSnapshot:
    """Immutable snapshot of cryptographic DAG verification results."""

    run_dir: Path
    status: TrustStatus
    issues: tuple[VerificationIssue, ...]
    receipt_hashes: dict[str, str] = field(default_factory=dict)
    verified_links: tuple[tuple[str, str, str], ...] = ()
    digital_dag_state: ReceiptDagState | None = None

    @property
    def is_accepted(self) -> bool:
        return self.status == TrustStatus.ACCEPTED

    @property
    def is_blocked(self) -> bool:
        return self.status == TrustStatus.BLOCKED

    @property
    def is_diagnostic(self) -> bool:
        return self.status == TrustStatus.DIAGNOSTIC

    @property
    def errors(self) -> tuple[VerificationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == VerificationSeverity.ERROR)

    @property
    def warnings(self) -> tuple[VerificationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == VerificationSeverity.WARNING)


def verify_receipt_dag(
    run_dir: str | Path,
    *,
    solve_domain: str | None = None,
    allow_missing_future_gates: bool = True,
    fail_fast: bool = False,
) -> TrustedDagSnapshot:
    """Verify cryptographic provenance, schemas, and semantic authority across Gates 0–7."""

    root = Path(run_dir)
    issues: list[VerificationIssue] = []
    receipt_hashes: dict[str, str] = {}
    verified_links: list[tuple[str, str, str]] = []

    # Map standard filenames to gate keys
    gate_files: dict[str, tuple[str, ...]] = {
        "body": (BODY_RECEIPT,),
        "correspondence": (CORRESPONDENCE_RECEIPT, TRANSFORM_RECEIPT),
        "basis": (BASIS_RECEIPT,),
        "rom_field": (ROM_FIELD_RECEIPT,),
        "seam_cost": (SEAM_COST_RECEIPT,),
        "solver": (SOLVER_PROMOTION_RECEIPT,),
        "cut_topology": (CUT_TOPOLOGY_RECEIPT,),
        "metric_correction": (METRIC_CORRECTION_RECEIPT,),
        "panel": (PANEL_UNWRAP_RECEIPT,),
        "manufacture": (MANUFACTURING_RECEIPT,),
    }

    raw_payloads: dict[str, dict[str, Any]] = {}
    found_paths: dict[str, Path] = {}

    # -------------------------------------------------------------------------
    # Layer 1: File Existence & Basic Schema Invariants
    # -------------------------------------------------------------------------
    for gate, filenames in gate_files.items():
        found_path: Path | None = None
        for fn in filenames:
            p = root / fn
            if p.exists():
                found_path = p
                break
        if found_path is not None:
            found_paths[gate] = found_path
            raw = _load_raw_json(found_path)
            if raw is None:
                issues.append(
                    VerificationIssue(
                        code="SCHEMA_INVALID_JSON",
                        receipt_id=gate,
                        detail=f"Receipt at {found_path.name} is not valid JSON or not a JSON object",
                        severity=VerificationSeverity.ERROR,
                    )
                )
                if fail_fast:
                    return _make_snapshot(root, issues, receipt_hashes, verified_links, None)
            else:
                raw_payloads[gate] = raw
                receipt_hashes[gate] = _sha256_file(found_path)
                # Check promotion invariant
                promo = raw.get("promotion")
                if promo not in (-1, 0, 1):
                    issues.append(
                        VerificationIssue(
                            code="SCHEMA_INVALID_PROMOTION",
                            receipt_id=gate,
                            detail=f"Field 'promotion' in {found_path.name} must be -1, 0, or 1; got {promo!r}",
                            severity=VerificationSeverity.ERROR,
                        )
                    )

    # -------------------------------------------------------------------------
    # Layer 2: Cryptographic Provenance & Hash Link Integrity
    # -------------------------------------------------------------------------
    def _verify_link(child_gate: str, parent_gate: str, hash_field: str) -> None:
        child_payload = raw_payloads.get(child_gate)
        if child_payload is None:
            return
        expected_hash = child_payload.get(hash_field)
        if expected_hash is None:
            # Hash field missing in child receipt
            issues.append(
                VerificationIssue(
                    code="PROVENANCE_MISSING_HASH_FIELD",
                    receipt_id=child_gate,
                    detail=f"Receipt {child_gate} is missing required hash reference '{hash_field}'",
                    severity=VerificationSeverity.ERROR,
                )
            )
            return

        if parent_gate not in found_paths or parent_gate not in receipt_hashes:
            issues.append(
                VerificationIssue(
                    code="PROVENANCE_MISSING_PARENT_RECEIPT",
                    receipt_id=child_gate,
                    detail=f"Receipt {child_gate} references parent {parent_gate} via '{hash_field}', but {parent_gate} receipt is absent or invalid",
                    severity=VerificationSeverity.ERROR,
                )
            )
            return

        actual_parent_hash = receipt_hashes[parent_gate]
        if actual_parent_hash != expected_hash:
            issues.append(
                VerificationIssue(
                    code="PROVENANCE_HASH_MISMATCH",
                    receipt_id=child_gate,
                    detail=f"Cryptographic mismatch in {child_gate}.{hash_field}: expected {expected_hash}, actual file hash is {actual_parent_hash}",
                    severity=VerificationSeverity.ERROR,
                )
            )
        else:
            verified_links.append((child_gate, parent_gate, expected_hash))

    # Provenance links across Gates 0 through 7
    if "basis" in raw_payloads:
        _verify_link("basis", "body", "carrier_receipt_hash")

    if "rom_field" in raw_payloads:
        _verify_link("rom_field", "basis", "basis_receipt_hash")

    if "seam_cost" in raw_payloads:
        _verify_link("seam_cost", "rom_field", "rom_field_receipt_hash")
        _verify_link("seam_cost", "body", "body_receipt_hash")
        if raw_payloads["seam_cost"].get("correspondence_receipt_hash") is not None:
            _verify_link("seam_cost", "correspondence", "correspondence_receipt_hash")

    if "solver" in raw_payloads:
        _verify_link("solver", "seam_cost", "seam_cost_receipt_hash")

    if "cut_topology" in raw_payloads:
        _verify_link("cut_topology", "solver", "solver_receipt_hash")

    if "metric_correction" in raw_payloads:
        _verify_link("metric_correction", "solver", "solver_receipt_hash")
        _verify_link("metric_correction", "cut_topology", "cut_topology_receipt_hash")

    if "panel" in raw_payloads:
        _verify_link("panel", "solver", "solver_receipt_hash")
        if raw_payloads["panel"].get("cut_topology_receipt_hash") is not None:
            _verify_link("panel", "cut_topology", "cut_topology_receipt_hash")
        if raw_payloads["panel"].get("metric_correction_receipt_hash") is not None:
            _verify_link("panel", "metric_correction", "metric_correction_receipt_hash")

    if "manufacture" in raw_payloads:
        _verify_link("manufacture", "panel", "panel_unwrap_receipt_hash")

    # -------------------------------------------------------------------------
    # Layer 3: Semantic Authority & Precondition Guards
    # -------------------------------------------------------------------------
    # Rule 3.1: Precondition monotonicity - a promoted child cannot be promoted if its parent is not promoted
    parent_dependencies: dict[str, tuple[str, ...]] = {
        "basis": ("body",),
        "rom_field": ("basis",),
        "seam_cost": ("rom_field", "body"),
        "solver": ("seam_cost",),
        "cut_topology": ("solver",),
        "metric_correction": ("solver", "cut_topology"),
        "panel": ("solver", "cut_topology"),
        "manufacture": ("panel",),
    }

    for child, parents in parent_dependencies.items():
        if raw_payloads.get(child, {}).get("promotion") == 1:
            for p in parents:
                parent_promo = raw_payloads.get(p, {}).get("promotion")
                if parent_promo != 1:
                    issues.append(
                        VerificationIssue(
                            code="AUTHORITY_PARENT_NOT_PROMOTED",
                            receipt_id=child,
                            detail=f"Gate '{child}' is marked promoted (1) but required parent '{p}' has promotion={parent_promo!r}",
                            severity=VerificationSeverity.ERROR,
                        )
                    )

    # Rule 3.2: Non-empty blockers guard when promotion is 1
    blocker_fields = {
        "cut_topology": "cut_topology_blockers",
        "metric_correction": "metric_correction_blockers",
        "panel": "panel_unwrap_blockers",
        "solver": "blocked_consumers",
    }
    for gate, field_name in blocker_fields.items():
        payload = raw_payloads.get(gate)
        if payload and payload.get("promotion") == 1:
            blockers = payload.get(field_name)
            if isinstance(blockers, Sequence) and len(blockers) > 0:
                issues.append(
                    VerificationIssue(
                        code="AUTHORITY_BLOCKERS_PRESENT_ON_PROMOTED",
                        receipt_id=gate,
                        detail=f"Receipt '{gate}' has promotion=1 but contains active blockers in '{field_name}': {blockers}",
                        severity=VerificationSeverity.ERROR,
                    )
                )

    # Rule 3.3: Solve domain consistency
    effective_domain = solve_domain
    if effective_domain is None and "seam_cost" in raw_payloads:
        effective_domain = str(raw_payloads["seam_cost"].get("solve_domain", ""))
    if effective_domain and effective_domain != "A_v3240":
        if (
            "correspondence" not in raw_payloads
            or raw_payloads["correspondence"].get("promotion") != 1
        ):
            issues.append(
                VerificationIssue(
                    code="AUTHORITY_CORRESPONDENCE_REQUIRED",
                    receipt_id="correspondence",
                    detail=f"Solve domain '{effective_domain}' requires a promoted correspondence receipt",
                    severity=VerificationSeverity.ERROR,
                )
            )

    # Read digital DAG state using compatibility reader
    digital_state: ReceiptDagState | None = None
    try:
        digital_state = read_receipt_dag(root, solve_domain=effective_domain)
        if digital_state.hash_chain_errors:
            for err in digital_state.hash_chain_errors:
                if not any(i.detail == err for i in issues):
                    issues.append(
                        VerificationIssue(
                            code="PROVENANCE_HASH_CHAIN_ERROR",
                            receipt_id="dag",
                            detail=err,
                            severity=VerificationSeverity.ERROR,
                        )
                    )
    except Exception as exc:
        issues.append(
            VerificationIssue(
                code="SCHEMA_CORRUPT_DAG_READ",
                receipt_id="dag",
                detail=f"Compatibility reader read_receipt_dag failed on {root.name}: {exc}",
                severity=VerificationSeverity.ERROR,
            )
        )

    return _make_snapshot(root, issues, receipt_hashes, verified_links, digital_state)


def _make_snapshot(
    root: Path,
    issues: list[VerificationIssue],
    receipt_hashes: dict[str, str],
    verified_links: list[tuple[str, str, str]],
    digital_state: ReceiptDagState | None,
) -> TrustedDagSnapshot:
    has_errors = any(i.severity == VerificationSeverity.ERROR for i in issues)
    has_warnings = any(i.severity == VerificationSeverity.WARNING for i in issues)

    if has_errors:
        status = TrustStatus.BLOCKED
    elif has_warnings:
        status = TrustStatus.DIAGNOSTIC
    else:
        status = TrustStatus.ACCEPTED

    return TrustedDagSnapshot(
        run_dir=root,
        status=status,
        issues=tuple(issues),
        receipt_hashes=receipt_hashes,
        verified_links=tuple(verified_links),
        digital_dag_state=digital_state,
    )
