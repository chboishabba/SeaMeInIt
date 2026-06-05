#!/usr/bin/env python3
"""Run the SMII Afflec Gates 0-7 receipt chain.

The runner is intentionally thin: it invokes the existing gate CLIs in
dependency order, stops at the first non-promoted receipt, and records a run
manifest. It does not replace the individual lane scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

PROMOTED = 1
EXIT_PROMOTED = 0
EXIT_BLOCKED = 1
EXIT_HARD_FAILURE = 2


@dataclass(frozen=True, slots=True)
class GateStep:
    """One executable gate in the native Afflec demo path."""

    key: str
    label: str
    command: tuple[str, ...]
    receipt_path: Path
    notes: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_receipt(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Expected receipt was not emitted: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Receipt must contain a JSON object: {path}")
    return payload


def _promotion(payload: Mapping[str, object]) -> int:
    try:
        return int(payload.get("promotion", 0))
    except (TypeError, ValueError):
        return 0


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not existing else f"src{os.pathsep}{existing}"
    return env


def _run_command(command: Sequence[str], *, cwd: Path) -> int:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=_pythonpath_env(),
        check=False,
    )
    return int(result.returncode)


def _gate_record(
    *,
    gate: GateStep,
    run_dir: Path,
    receipt_payload: Mapping[str, object] | None,
) -> dict[str, object]:
    if receipt_payload is None:
        return {
            "promotion": 0,
            "receipt": None,
            "receipt_hash": None,
            "notes": gate.notes,
        }
    notes = gate.notes
    receipt_notes = receipt_payload.get("notes")
    if isinstance(receipt_notes, str) and receipt_notes:
        notes = receipt_notes if not notes else f"{notes}; {receipt_notes}"
    return {
        "promotion": _promotion(receipt_payload),
        "receipt": _relative(gate.receipt_path, run_dir),
        "receipt_hash": _sha256_file(gate.receipt_path),
        "notes": notes,
    }


def build_steps(
    *,
    run_dir: Path,
    python: str,
    allow_synthetic_promotion: bool,
    manufacturing_method: str,
    force: bool,
    detector: str,
    require_high_trust_detector: bool,
    images: Sequence[Path] = (),
) -> list[GateStep]:
    """Build the native `A_v3240` demo command sequence."""

    body_dir = run_dir / "body"
    basis_dir = run_dir / "basis"
    rom_dir = run_dir / "rom"
    seams_dir = run_dir / "seams"
    solver_dir = run_dir / "solver"
    topology_dir = run_dir / "topology"
    corrections_dir = run_dir / "corrections"
    panels_dir = run_dir / "panels"
    manufacturing_dir = run_dir / "manufacturing"

    body_receipt = body_dir / "body_carrier_receipt.json"
    basis_receipt = basis_dir / "basis_receipt.json"
    rom_receipt = rom_dir / "rom_field_receipt.json"
    seam_cost_receipt = seams_dir / "seam_cost_receipt.json"
    solver_receipt = solver_dir / "solver_promotion_receipt.json"
    cut_topology_receipt = topology_dir / "cut_topology_receipt.json"
    metric_correction_receipt = corrections_dir / "metric_correction_receipt.json"
    panel_receipt = panels_dir / "panel_unwrap_receipt.json"
    manufacturing_receipt = manufacturing_dir / "manufacturing_receipt.json"
    finished_seam_receipt = manufacturing_dir / "finished_seam_receipt.json"

    afflec_command = [
        python,
        "-m",
        "smii.app",
        "afflec-demo",
        "--output",
        str(body_dir),
        "--detector",
        detector,
    ]
    if require_high_trust_detector:
        afflec_command.append("--require-high-trust-detector")
    if images:
        afflec_command.append("--images")
        afflec_command.extend(str(path) for path in images)
    if force:
        afflec_command.append("--force")

    rom_command = [
        python,
        "examples/rom_aggregate_from_samples.py",
        "--samples",
        "examples/data/rom_samples_demo.json",
        "--basis",
        str(basis_dir / "canonical_basis.npz"),
        "--basis-receipt",
        str(basis_receipt),
        "--output-dir",
        str(rom_dir),
        "--out-rom-fields",
        str(rom_dir / "rom_fields.npz"),
        "--out-rom-field-receipt",
        str(rom_receipt),
        "--save-costs",
        str(rom_dir / "diagnostic_seam_costs.npz"),
    ]
    if allow_synthetic_promotion:
        rom_command.append("--allow-synthetic-promotion")

    return [
        GateStep(
            key="body",
            label="Gate 0: body fit",
            command=tuple(afflec_command),
            receipt_path=body_receipt,
        ),
        GateStep(
            key="basis",
            label="Gate 2: canonical basis",
            command=(
                python,
                "scripts/generate_canonical_basis.py",
                "--vertices",
                str(body_dir / "afflec_body.npz"),
                "--body-receipt",
                str(body_receipt),
                "--components",
                "4",
                "--output",
                str(basis_dir / "canonical_basis.npz"),
                "--receipt-output",
                str(basis_receipt),
            ),
            receipt_path=basis_receipt,
        ),
        GateStep(
            key="rom_field",
            label="Gate 3: ROM field aggregation",
            command=tuple(rom_command),
            receipt_path=rom_receipt,
            notes="synthetic promotion allowed" if allow_synthetic_promotion else "",
        ),
        GateStep(
            key="seam_cost",
            label="Gate 4: seam costs",
            command=(
                python,
                "scripts/compute_seam_costs.py",
                "--body-receipt",
                str(body_receipt),
                "--rom-field-receipt",
                str(rom_receipt),
                "--rom-fields",
                str(rom_dir / "rom_fields.npz"),
                "--mesh",
                str(body_dir / "afflec_body.npz"),
                "--out-costs",
                str(seams_dir / "seam_costs.npz"),
                "--out-seam-cost-receipt",
                str(seam_cost_receipt),
                "--solve-domain",
                "A_v3240",
            ),
            receipt_path=seam_cost_receipt,
        ),
        GateStep(
            key="solver",
            label="Gate 5: seam solver",
            command=(
                python,
                "scripts/solve_seams.py",
                "--seam-cost-receipt",
                str(seam_cost_receipt),
                "--costs",
                str(seams_dir / "seam_costs.npz"),
                "--mesh",
                str(body_dir / "afflec_body.npz"),
                "--out-dir",
                str(solver_dir),
                "--out-solver-receipt",
                str(solver_receipt),
                "--solver-mode",
                "metric_panelization",
                "--correction-families",
                "dart,relief_cut,ease,gusset,stretch_zone",
            ),
            receipt_path=solver_receipt,
        ),
        GateStep(
            key="cut_topology",
            label="Gate 5b: cut topology",
            command=(
                python,
                "scripts/validate_cut_topology.py",
                "--solver-receipt",
                str(solver_receipt),
                "--seam-edges",
                str(solver_dir / "seam_edges.npz"),
                "--mesh",
                str(body_dir / "afflec_body.npz"),
                "--corrections",
                str(solver_dir / "corrections.json"),
                "--out-cut-topology-receipt",
                str(cut_topology_receipt),
            ),
            receipt_path=cut_topology_receipt,
        ),
        GateStep(
            key="metric_correction",
            label="Gate 5c: metric correction",
            command=(
                python,
                "scripts/emit_metric_correction_receipt.py",
                "--solver-receipt",
                str(solver_receipt),
                "--cut-topology-receipt",
                str(cut_topology_receipt),
                "--seam-edges",
                str(solver_dir / "seam_edges.npz"),
                "--corrections",
                str(solver_dir / "corrections.json"),
                "--out-metric-correction-receipt",
                str(metric_correction_receipt),
            ),
            receipt_path=metric_correction_receipt,
        ),
        GateStep(
            key="panel_unwrap",
            label="Gate 6: panel unwrap",
            command=(
                python,
                "scripts/unwrap_panels.py",
                "--solver-receipt",
                str(solver_receipt),
                "--seam-edges",
                str(solver_dir / "seam_edges.npz"),
                "--mesh",
                str(body_dir / "afflec_body.npz"),
                "--out-dir",
                str(panels_dir),
                "--cut-topology-receipt",
                str(cut_topology_receipt),
                "--metric-correction-receipt",
                str(metric_correction_receipt),
                "--corrections",
                str(solver_dir / "corrections.json"),
                "--out-panel-receipt",
                str(panel_receipt),
            ),
            receipt_path=panel_receipt,
        ),
        GateStep(
            key="manufacture",
            label="Gate 7: manufacturing artifacts and finished seam receipt",
            command=(
                python,
                "scripts/generate_manufacturing_artifacts.py",
                "--panel-receipt",
                str(panel_receipt),
                "--panel-uvs",
                str(panels_dir / "panel_uvs.npz"),
                "--rom-fields",
                str(rom_dir / "rom_fields.npz"),
                "--out-dir",
                str(manufacturing_dir),
                "--out-manufacturing-receipt",
                str(manufacturing_receipt),
                "--manufacturing-method",
                manufacturing_method,
                "--out-finished-seam-receipt",
                str(finished_seam_receipt),
                "--body-receipt",
                str(body_receipt),
                "--rom-receipt",
                str(rom_receipt),
                "--fabric-receipt-hash",
                "inline-demo-fabric-profile",
                "--basis-receipt",
                str(basis_receipt),
                "--seam-cost-receipt",
                str(seam_cost_receipt),
                "--solver-receipt",
                str(solver_receipt),
                "--cut-topology-receipt",
                str(cut_topology_receipt),
                "--metric-correction-receipt",
                str(metric_correction_receipt),
            ),
            receipt_path=manufacturing_receipt,
        ),
    ]


def initial_gate_records() -> dict[str, dict[str, object]]:
    return {
        "body": {"promotion": 0, "receipt": None, "receipt_hash": None, "notes": ""},
        "correspondence": {
            "promotion": 0,
            "receipt": None,
            "receipt_hash": None,
            "notes": "skipped: native A_v3240 demo path does not make a transfer-backed claim",
        },
        "basis": {"promotion": 0, "receipt": None, "receipt_hash": None, "notes": ""},
        "rom_field": {"promotion": 0, "receipt": None, "receipt_hash": None, "notes": ""},
        "seam_cost": {"promotion": 0, "receipt": None, "receipt_hash": None, "notes": ""},
        "solver": {"promotion": 0, "receipt": None, "receipt_hash": None, "notes": ""},
        "cut_topology": {"promotion": 0, "receipt": None, "receipt_hash": None, "notes": ""},
        "metric_correction": {
            "promotion": 0,
            "receipt": None,
            "receipt_hash": None,
            "notes": "",
        },
        "panel_unwrap": {"promotion": 0, "receipt": None, "receipt_hash": None, "notes": ""},
        "manufacture": {"promotion": 0, "receipt": None, "receipt_hash": None, "notes": ""},
    }


def _can_manufacture(gates: Mapping[str, Mapping[str, object]]) -> bool:
    required = (
        "body",
        "basis",
        "rom_field",
        "seam_cost",
        "solver",
        "cut_topology",
        "metric_correction",
        "panel_unwrap",
        "manufacture",
    )
    return all(gates[name].get("promotion") == PROMOTED for name in required)


def write_manifest(
    *,
    run_dir: Path,
    run_id: str,
    started: str,
    gates: Mapping[str, Mapping[str, object]],
    first_blocker: str | None,
    exit_code: int,
) -> Path:
    manifest = {
        "run_id": run_id,
        "started": started,
        "completed": _utc_now(),
        "exit_code": int(exit_code),
        "gates": gates,
        "first_blocker": first_blocker,
        "can_manufacture": _can_manufacture(gates),
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def run_demo(
    *,
    run_dir: Path,
    python: str,
    allow_synthetic_promotion: bool,
    manufacturing_method: str,
    force: bool,
    detector: str,
    require_high_trust_detector: bool,
    dry_run: bool,
    images: Sequence[Path] = (),
    runner: Callable[[Sequence[str], Path], int] | None = None,
) -> int:
    started = _utc_now()
    run_id = f"afflec_receipted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    steps = build_steps(
        run_dir=run_dir,
        python=python,
        allow_synthetic_promotion=allow_synthetic_promotion,
        manufacturing_method=manufacturing_method,
        force=force,
        detector=detector,
        require_high_trust_detector=require_high_trust_detector,
        images=tuple(images),
    )

    if dry_run:
        print(f"[DRY-RUN] run_id={run_id}")
        print("[DRY-RUN] no directories, manifests, receipts, or artifacts will be written")
        for step in steps:
            print(f"\n[{step.label}]")
            print("  " + " ".join(step.command))
        return EXIT_PROMOTED

    gates = initial_gate_records()
    first_blocker: str | None = None
    command_runner = runner or (lambda command, cwd: _run_command(command, cwd=cwd))
    run_dir.mkdir(parents=True, exist_ok=True)

    for step in steps:
        print(f"\n[{step.label}]")
        print("  " + " ".join(step.command))
        return_code = command_runner(step.command, Path.cwd())
        if return_code != 0:
            gates[step.key]["notes"] = (
                f"{Path(step.command[1]).name if len(step.command) > 1 else step.key} exited {return_code}"
            )
            first_blocker = step.key
            manifest_path = write_manifest(
                run_dir=run_dir,
                run_id=run_id,
                started=started,
                gates=gates,
                first_blocker=first_blocker,
                exit_code=EXIT_HARD_FAILURE,
            )
            print(f"\nRun manifest: {manifest_path}")
            print(f"Hard gate failure at {step.key}: command exited {return_code}")
            return EXIT_HARD_FAILURE

        try:
            receipt_payload = _load_receipt(step.receipt_path)
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            gates[step.key]["notes"] = str(exc)
            first_blocker = step.key
            manifest_path = write_manifest(
                run_dir=run_dir,
                run_id=run_id,
                started=started,
                gates=gates,
                first_blocker=first_blocker,
                exit_code=EXIT_HARD_FAILURE,
            )
            print(f"\nRun manifest: {manifest_path}")
            print(f"Hard gate failure at {step.key}: {exc}")
            return EXIT_HARD_FAILURE

        gates[step.key] = _gate_record(
            gate=step,
            run_dir=run_dir,
            receipt_payload=receipt_payload,
        )
        promotion = gates[step.key]["promotion"]
        if promotion != PROMOTED:
            first_blocker = step.key
            manifest_path = write_manifest(
                run_dir=run_dir,
                run_id=run_id,
                started=started,
                gates=gates,
                first_blocker=first_blocker,
                exit_code=EXIT_BLOCKED,
            )
            print(f"\nRun manifest: {manifest_path}")
            print(f"Stopped at {step.key}: promotion={promotion}")
            return EXIT_BLOCKED
        print(f"  {step.key} promoted")

    manifest_path = write_manifest(
        run_dir=run_dir,
        run_id=run_id,
        started=started,
        gates=gates,
        first_blocker=None,
        exit_code=EXIT_PROMOTED,
    )
    print(f"\nRun manifest: {manifest_path}")
    print("All gates promoted; manufacturing artifacts are ready.")
    return EXIT_PROMOTED


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/demo/afflec_receipted"),
        help="Output run root (default: outputs/demo/afflec_receipted).",
    )
    parser.add_argument(
        "--allow-synthetic-promotion",
        action="store_true",
        help="Allow synthetic ROM samples to promote in Gate 3 for demo/test runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned command chain without writing artifacts.",
    )
    parser.add_argument(
        "--manufacturing-method",
        choices=[
            "home_sewing",
            "overlock",
            "flatlock",
            "bonded",
            "welded",
            "laser_cut",
            "3d_print",
            "eva_foam_cut",
        ],
        default="home_sewing",
        help="Manufacturing method carrier for Gate 7.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to afflec-demo so existing Gate 0 outputs are replaced.",
    )
    parser.add_argument(
        "--detector",
        choices=("mediapipe", "bbox"),
        default="mediapipe",
        help=(
            "Gate 0 body detector passed to afflec-demo "
            "(default: mediapipe; use bbox only for coarse diagnostics)."
        ),
    )
    parser.add_argument(
        "--require-high-trust-detector",
        action="store_true",
        help="Pass --require-high-trust-detector to afflec-demo.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        default=(),
        help="Optional Afflec image set passed through to Gate 0 afflec-demo.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for subcommands (default: current interpreter).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    return run_demo(
        run_dir=args.output,
        python=args.python,
        allow_synthetic_promotion=args.allow_synthetic_promotion,
        manufacturing_method=args.manufacturing_method,
        force=args.force,
        detector=args.detector,
        require_high_trust_detector=args.require_high_trust_detector,
        dry_run=args.dry_run,
        images=tuple(args.images),
    )


if __name__ == "__main__":
    raise SystemExit(main())
