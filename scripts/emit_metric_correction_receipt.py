#!/usr/bin/env python3
"""Emit a MetricCorrectionReceipt between cut topology and panel unwrap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from smii.seams import (
    CORRECTION_TYPES,
    CutTopologyReceipt,
    MetricCorrectionEntry,
    MetricCorrectionReceipt,
    load_cut_topology_receipt,
    load_solver_promotion_receipt,
)

UNWRAP_COMPATIBLE_STATES = {"correctionOk", "correctionDegraded", "correctionAbstained"}
UNSUPPORTED_RECEIPT_TYPES = {"variable_knit", "pleat", "bias_orientation"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def _selected_corrections(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    selected = payload.get("selected_corrections", [])
    if not isinstance(selected, list):
        return []
    return [entry for entry in selected if isinstance(entry, Mapping)]


def _energy(payload: Mapping[str, object]) -> Mapping[str, object]:
    value = payload.get("energy", {})
    return value if isinstance(value, Mapping) else {}


def _float_value(payload: Mapping[str, object], key: str, default: float = 0.0) -> float:
    try:
        value = float(payload.get(key, default))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return value if value >= 0.0 else default


def _entry_from_selected(entry: Mapping[str, object], *, residual_gate: float) -> MetricCorrectionEntry:
    family = str(entry.get("family", "abstain"))
    raw = _float_value(entry, "raw_residual")
    corrected = _float_value(entry, "corrected_residual", raw)
    blockers: list[str] = []
    if family in UNSUPPORTED_RECEIPT_TYPES or family not in CORRECTION_TYPES:
        blockers.append("unsupportedCorrectionType")
        family = "abstain"
    if corrected > raw:
        blockers.append("metricPropagationLawFailed")
    if corrected > residual_gate:
        result_state = "correctionDegraded"
    else:
        result_state = "correctionOk"
    energy_terms = {
        "correction_cost": _float_value(entry, "correction_cost"),
        "complexity_penalty": _float_value(entry, "complexity_penalty"),
        "manufacture_penalty": _float_value(entry, "manufacture_penalty"),
        "gain": _float_value(entry, "gain"),
    }
    return MetricCorrectionEntry(
        panel_label=int(entry.get("panel_label", 0)),
        correction_type=family,
        delta_metric_meaning=str(
            entry.get("reason", "local metric correction from panelization payload")
        ),
        raw_residual=raw,
        corrected_residual=corrected,
        energy_terms=energy_terms,
        result_state=result_state,
        blockers=blockers,
    )


def _typed_operator_count(receipt: CutTopologyReceipt) -> int:
    return int(
        receipt.typed_operator_count
        or (
            receipt.typed_dart_count
            + receipt.typed_gusset_count
            + receipt.typed_relief_cut_count
            + receipt.typed_ease_count
            + receipt.typed_stretch_zone_count
        )
    )


def emit_metric_correction_receipt(
    *,
    solver_receipt_path: Path,
    cut_topology_receipt_path: Path,
    seam_edges_path: Path,
    receipt_path: Path,
    corrections_path: Path | None = None,
    residual_gate: float = 0.05,
) -> MetricCorrectionReceipt:
    """Build and write a metric-correction receipt."""

    solver_receipt = load_solver_promotion_receipt(solver_receipt_path)
    if solver_receipt.promotion != 1:
        raise ValueError(
            f"SolverPromotionReceipt not promoted ({solver_receipt.promotion}). "
            f"Blocked: {solver_receipt.blocked_consumers}"
        )
    cut_topology_receipt = load_cut_topology_receipt(cut_topology_receipt_path)
    if cut_topology_receipt.promotion != 1:
        raise ValueError(
            f"CutTopologyReceipt not promoted ({cut_topology_receipt.promotion}). "
            f"Blocked: {cut_topology_receipt.blocked_consumers}"
        )
    if cut_topology_receipt.solver_receipt_hash != _sha256_file(solver_receipt_path):
        raise ValueError("CutTopologyReceipt.solver_receipt_hash does not match solver receipt.")
    seam_edges_hash = _sha256_file(seam_edges_path)
    if solver_receipt.seam_hash != seam_edges_hash:
        raise ValueError("Seam edges hash does not match SolverPromotionReceipt.seam_hash.")
    if cut_topology_receipt.seam_edges_hash != seam_edges_hash:
        raise ValueError("CutTopologyReceipt.seam_edges_hash does not match seam edges.")

    typed_count = _typed_operator_count(cut_topology_receipt)
    blockers: list[str] = []
    entries: list[MetricCorrectionEntry] = []
    correction_payload_hash: str | None = None
    raw_total = 0.0
    corrected_total = 0.0

    if corrections_path is not None:
        if not corrections_path.exists():
            blockers.append("missing_correction_payload")
        else:
            correction_payload_hash = _sha256_file(corrections_path)
            if (
                solver_receipt.correction_payload_hash is not None
                and solver_receipt.correction_payload_hash != correction_payload_hash
            ):
                blockers.append("correction_payload_hash_mismatch")
            payload = _load_json(corrections_path)
            selected = _selected_corrections(payload)
            entries = [
                _entry_from_selected(entry, residual_gate=float(residual_gate))
                for entry in selected
            ]
            energy = _energy(payload)
            raw_total = _float_value(energy, "raw_residual_total", sum(e.raw_residual for e in entries))
            corrected_total = _float_value(
                energy,
                "corrected_residual_total",
                sum(e.corrected_residual for e in entries),
            )
    elif typed_count > 0:
        blockers.extend(
            [
                "missingShapingIntentReceipt",
                "missingDeltaMetricMeaning",
                "missingPanelUnwrapCompatibility",
            ]
        )

    entry_blockers = [blocker for entry in entries for blocker in entry.blockers]
    blockers.extend(entry_blockers)
    if typed_count > 0 and len(entries) < typed_count:
        blockers.append("missingShapingIntentReceipt")
    if typed_count > 0 and not entries:
        blockers.append("missingDeltaMetricMeaning")
    if corrected_total > float(residual_gate):
        blockers.append("correctionResidualExceedsGate")
    if any(entry.result_state not in UNWRAP_COMPATIBLE_STATES for entry in entries):
        blockers.append("missingPanelUnwrapCompatibility")
    blockers = list(dict.fromkeys(blockers))

    promotion = 1 if not blockers else 0
    receipt = MetricCorrectionReceipt(
        solver_receipt_hash=_sha256_file(solver_receipt_path),
        cut_topology_receipt_hash=_sha256_file(cut_topology_receipt_path),
        seam_edges_hash=seam_edges_hash,
        panels_requiring_correction=sorted({entry.panel_label for entry in entries}),
        corrections=entries,
        raw_residual_total=raw_total,
        corrected_residual_total=corrected_total,
        residual_gate=float(residual_gate),
        promotion=promotion,
        blocked_consumers=[],
        metric_correction_blockers=blockers,
        correction_payload_hash=correction_payload_hash,
    )
    receipt.to_json(receipt_path)
    print(f"Wrote metric correction receipt to {receipt_path}")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-receipt", type=Path, required=True)
    parser.add_argument("--cut-topology-receipt", type=Path, required=True)
    parser.add_argument("--seam-edges", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, default=None)
    parser.add_argument("--residual-gate", type=float, default=0.05)
    parser.add_argument(
        "--out-metric-correction-receipt",
        type=Path,
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    emit_metric_correction_receipt(
        solver_receipt_path=args.solver_receipt,
        cut_topology_receipt_path=args.cut_topology_receipt,
        seam_edges_path=args.seam_edges,
        corrections_path=args.corrections,
        residual_gate=args.residual_gate,
        receipt_path=args.out_metric_correction_receipt,
    )


if __name__ == "__main__":
    main()
