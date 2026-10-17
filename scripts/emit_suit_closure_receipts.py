#!/usr/bin/env python3
"""Emit the post-pattern ROM-to-physical-suit closure receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from smii.seams import (
    TriadicTransportWitness,
    derive_assembly_receipt,
    derive_physical_rom_validation_receipt,
    derive_wear_fit_receipt,
    load_assembly_receipt,
    load_finished_seam_receipt,
    load_wear_fit_receipt,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def emit_assembly(args: argparse.Namespace) -> None:
    finished = load_finished_seam_receipt(args.finished_seam_receipt)
    receipt = derive_assembly_receipt(
        finished_seam_receipt_hash=_sha256_file(args.finished_seam_receipt),
        finished_seam_receipt=finished,
        manufactured_artifacts_hash=_sha256_file(args.manufactured_artifacts),
        assembled_geometry_hash=_sha256_file(args.assembled_geometry),
        assembly_evidence_hash=_sha256_file(args.assembly_evidence),
        piece_count=args.piece_count,
        realized_seam_count=args.realized_seam_count,
        max_geometry_deviation=args.max_geometry_deviation,
        geometry_deviation_threshold=args.geometry_deviation_threshold,
        material_lot_hashes=[_sha256_file(path) for path in args.material_lot],
    )
    receipt.to_json(args.out)
    print(f"Wrote AssemblyReceipt to {args.out} ({receipt.promotion})")


def emit_wear_fit(args: argparse.Namespace) -> None:
    assembly = load_assembly_receipt(args.assembly_receipt)
    residual_payload = _json_object(args.residuals)
    residuals = residual_payload.get("field_residuals", {})
    thresholds = residual_payload.get("residual_thresholds", {})
    if not isinstance(residuals, Mapping) or not isinstance(thresholds, Mapping):
        raise TypeError("residual JSON must contain object field_residuals and residual_thresholds")
    receipt = derive_wear_fit_receipt(
        assembly_receipt_hash=_sha256_file(args.assembly_receipt),
        assembly_receipt=assembly,
        rom_receipt_hash=_sha256_file(args.rom_receipt),
        task_profile_hash=_sha256_file(args.task_profile),
        predicted_field_hash=_sha256_file(args.predicted_field),
        measured_field_hash=_sha256_file(args.measured_field),
        measurement_evidence_hash=_sha256_file(args.measurement_evidence),
        sample_count=args.sample_count,
        task_coverage=args.task_coverage,
        field_residuals={str(k): float(v) for k, v in residuals.items()},
        residual_thresholds={str(k): float(v) for k, v in thresholds.items()},
        minimum_task_coverage=args.minimum_task_coverage,
    )
    receipt.to_json(args.out)
    print(f"Wrote WearFitReceipt to {args.out} ({receipt.promotion})")


def emit_physical_rom(args: argparse.Namespace) -> None:
    wear_fit = load_wear_fit_receipt(args.wear_fit_receipt)
    triadic_transport = None
    if args.triadic_transport is not None:
        triadic_transport = TriadicTransportWitness.from_mapping(
            _json_object(args.triadic_transport)
        )
    receipt = derive_physical_rom_validation_receipt(
        wear_fit_receipt_hash=_sha256_file(args.wear_fit_receipt),
        wear_fit_receipt=wear_fit,
        source_rom_receipt_hash=_sha256_file(args.source_rom_receipt),
        validation_evidence_hash=_sha256_file(args.validation_evidence),
        field_residual_hash=_sha256_file(args.field_residual),
        closure_iteration=args.closure_iteration,
        calibrated_rom_receipt_hash=(
            _sha256_file(args.calibrated_rom_receipt)
            if args.calibrated_rom_receipt is not None
            else None
        ),
        triadic_transport=triadic_transport,
        require_triadic_transport=not args.allow_nontriadic,
    )
    receipt.to_json(args.out)
    print(f"Wrote PhysicalROMValidationReceipt to {args.out} ({receipt.promotion})")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    assembly = subparsers.add_parser("assembly", help="emit AssemblyReceipt")
    assembly.add_argument("--finished-seam-receipt", type=Path, required=True)
    assembly.add_argument("--manufactured-artifacts", type=Path, required=True)
    assembly.add_argument("--assembled-geometry", type=Path, required=True)
    assembly.add_argument("--assembly-evidence", type=Path, required=True)
    assembly.add_argument("--material-lot", type=Path, action="append", default=[])
    assembly.add_argument("--piece-count", type=int, required=True)
    assembly.add_argument("--realized-seam-count", type=int, required=True)
    assembly.add_argument("--max-geometry-deviation", type=float, required=True)
    assembly.add_argument("--geometry-deviation-threshold", type=float, required=True)
    assembly.add_argument("--out", type=Path, required=True)
    assembly.set_defaults(func=emit_assembly)

    wear = subparsers.add_parser("wear-fit", help="emit WearFitReceipt")
    wear.add_argument("--assembly-receipt", type=Path, required=True)
    wear.add_argument("--rom-receipt", type=Path, required=True)
    wear.add_argument("--task-profile", type=Path, required=True)
    wear.add_argument("--predicted-field", type=Path, required=True)
    wear.add_argument("--measured-field", type=Path, required=True)
    wear.add_argument("--measurement-evidence", type=Path, required=True)
    wear.add_argument("--residuals", type=Path, required=True)
    wear.add_argument("--sample-count", type=int, required=True)
    wear.add_argument("--task-coverage", type=float, required=True)
    wear.add_argument("--minimum-task-coverage", type=float, default=0.95)
    wear.add_argument("--out", type=Path, required=True)
    wear.set_defaults(func=emit_wear_fit)

    physical = subparsers.add_parser("physical-rom", help="emit PhysicalROMValidationReceipt")
    physical.add_argument("--wear-fit-receipt", type=Path, required=True)
    physical.add_argument("--source-rom-receipt", type=Path, required=True)
    physical.add_argument("--validation-evidence", type=Path, required=True)
    physical.add_argument("--field-residual", type=Path, required=True)
    physical.add_argument("--closure-iteration", type=int, required=True)
    physical.add_argument("--calibrated-rom-receipt", type=Path, default=None)
    physical.add_argument("--triadic-transport", type=Path, default=None)
    physical.add_argument(
        "--allow-nontriadic",
        action="store_true",
        help="do not require a C3/BT369 witness for this declared validation slice",
    )
    physical.add_argument("--out", type=Path, required=True)
    physical.set_defaults(func=emit_physical_rom)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
