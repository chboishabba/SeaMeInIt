#!/usr/bin/env python3
"""Run the P3 Afflec transfer check and native receipt chain until blocked."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_PROMOTED_ROOT = Path("outputs/p3_afflec_gate0_20260604/curated_refs_refined")
DEFAULT_FORWARD_OBJECT = Path("outputs/suits/afflec_body/base_layer.npz")
DEFAULT_ROM_SAMPLES = Path("examples/data/rom_samples_demo.json")
EXPECTED_FORWARD_HASH = "b122dc2cf8b075a5a5bcc0c124a075247268332203df7873c36de65e4027695c"
STAGE_ORDER = (
    "source validation",
    "back-transfer acceptance",
    "basis generation",
    "rom aggregation",
    "seam costs",
    "dart/relief candidates",
    "seam solve",
    "cut topology",
    "metric correction",
    "panel unwrap",
    "panel diagnostics",
)
STAGE_ESTIMATES_SECONDS = {
    "source validation": 2.0,
    "back-transfer acceptance": 5.0,
    "basis generation": 10.0,
    "rom aggregation": 15.0,
    "seam costs": 15.0,
    "dart/relief candidates": 10.0,
    "seam solve": 30.0,
    "cut topology": 5.0,
    "metric correction": 5.0,
    "panel unwrap": 20.0,
    "panel diagnostics": 15.0,
}
SCRIPT_STAGES = {
    "scripts/p3_back_transfer_acceptance.py": "back-transfer acceptance",
    "scripts/generate_canonical_basis.py": "basis generation",
    "examples/rom_aggregate_from_samples.py": "rom aggregation",
    "scripts/compute_seam_costs.py": "seam costs",
    "scripts/propose_dart_relief_cuts.py": "dart/relief candidates",
    "scripts/solve_seams.py": "seam solve",
    "scripts/validate_cut_topology.py": "cut topology",
    "scripts/emit_metric_correction_receipt.py": "metric correction",
    "scripts/unwrap_panels.py": "panel unwrap",
    "scripts/render_panel_patterns.py": "panel diagnostics",
}
RUN_STARTED_AT: float | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def _sample_component_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    if not entries:
        raise ValueError("ROM sample payload has no samples.")
    coeffs = entries[0]["coeffs"]
    first = next(iter(coeffs.values()))
    return int(len(first))


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.0f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    return f"{minutes}m {remainder:02d}s"


def _stage_name(command: list[str]) -> str:
    if len(command) > 1:
        return SCRIPT_STAGES.get(command[1], Path(command[1]).name)
    return "command"


def _stage_index(stage: str) -> int:
    try:
        return STAGE_ORDER.index(stage)
    except ValueError:
        return len(STAGE_ORDER) - 1


def _estimated_remaining(stage: str) -> float:
    index = _stage_index(stage)
    return sum(STAGE_ESTIMATES_SECONDS.get(name, 10.0) for name in STAGE_ORDER[index:])


def _progress(stage: str, message: str, *, started_at: float) -> None:
    index = _stage_index(stage)
    print(
        "[P3 progress] "
        f"{index + 1}/{len(STAGE_ORDER)} {stage}: {message} "
        f"(elapsed {_format_seconds(time.monotonic() - started_at)}, "
        f"est. remaining {_format_seconds(_estimated_remaining(stage))})",
        flush=True,
    )


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    stage = _stage_name(command)
    started_at = RUN_STARTED_AT if RUN_STARTED_AT is not None else time.monotonic()
    estimate = STAGE_ESTIMATES_SECONDS.get(stage, 10.0)
    _progress(stage, f"starting, estimate {_format_seconds(estimate)}", started_at=started_at)
    print("+ " + " ".join(command), flush=True)
    command_started_at = time.monotonic()
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            env=env,
        )
    except subprocess.CalledProcessError:
        print(
            "[P3 progress] "
            f"{_stage_index(stage) + 1}/{len(STAGE_ORDER)} {stage}: failed after "
            f"{_format_seconds(time.monotonic() - command_started_at)}",
            flush=True,
        )
        raise
    else:
        print(
            "[P3 progress] "
            f"{_stage_index(stage) + 1}/{len(STAGE_ORDER)} {stage}: finished in "
            f"{_format_seconds(time.monotonic() - command_started_at)}",
            flush=True,
        )
    return result


def _receipt_promoted(path: Path) -> tuple[bool, int | None, list[str]]:
    payload = _load_json(path)
    promotion = payload.get("promotion")
    blocked = payload.get("blocked_consumers", [])
    if not isinstance(blocked, list):
        blocked = [str(blocked)]
    return (
        promotion == 1,
        int(promotion) if isinstance(promotion, int) else None,
        [str(v) for v in blocked],
    )


def _typed_operator_count(path: Path) -> int:
    payload = _load_json(path)
    typed_count = payload.get("typed_operator_count")
    if isinstance(typed_count, int):
        return max(0, int(typed_count))
    total = 0
    for key in (
        "typed_dart_count",
        "typed_gusset_count",
        "typed_relief_cut_count",
        "typed_ease_count",
        "typed_stretch_zone_count",
    ):
        value = payload.get(key, 0)
        if isinstance(value, int):
            total += max(0, int(value))
    return total


def _record_blocker(
    *,
    summary_path: Path,
    stage: str,
    receipt_path: Path | None,
    error: str | None = None,
    extras: Mapping[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "status": "blocked",
        "blocked_stage": stage,
    }
    if receipt_path is not None and receipt_path.exists():
        promoted, promotion, blocked = _receipt_promoted(receipt_path)
        payload.update(
            {
                "receipt_path": str(receipt_path),
                "promotion": promotion,
                "blocked_consumers": blocked,
                "promoted": promoted,
            }
        )
    if error is not None:
        payload["error"] = error
    if extras is not None:
        payload.update(extras)
    _write_json(summary_path, payload)


def _render_panel_diagnostics(
    *,
    python: str,
    panel_receipt: Path,
    panel_uvs: Path,
    mesh: Path,
    seam_edges: Path,
    output_dir: Path,
) -> Path:
    _run(
        [
            python,
            "scripts/render_panel_patterns.py",
            "--panel-receipt",
            str(panel_receipt),
            "--panel-uvs",
            str(panel_uvs),
            "--out-dir",
            str(output_dir),
            "--mesh",
            str(mesh),
            "--seam-edges",
            str(seam_edges),
        ]
    )
    return output_dir / "diagnostic_pattern_summary.json"


def run_chain(
    *,
    output_root: Path,
    promoted_root: Path,
    forward_object: Path,
    rom_samples: Path,
    python: str,
) -> None:
    global RUN_STARTED_AT
    RUN_STARTED_AT = time.monotonic()
    _progress("source validation", "checking required local artifacts", started_at=RUN_STARTED_AT)
    body_receipt = promoted_root / "body_carrier_receipt.json"
    body_mesh = promoted_root / "afflec_body.npz"
    if not body_receipt.exists() or not body_mesh.exists():
        raise FileNotFoundError("Promoted curated body receipt and afflec_body.npz are required.")

    forward_hash = _sha256_file(forward_object)
    if forward_hash != EXPECTED_FORWARD_HASH:
        raise ValueError(
            "Historical B_v9438 forward object hash mismatch: "
            f"expected={EXPECTED_FORWARD_HASH}, actual={forward_hash}."
        )
    print(
        "[P3 progress] 1/11 source validation: finished in "
        f"{_format_seconds(time.monotonic() - RUN_STARTED_AT)}",
        flush=True,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    body_receipt_payload = _load_json(body_receipt)
    solve_domain = str(body_receipt_payload.get("topology_label", "A_v9384"))
    summary_path = output_root / "p3_chain_summary.json"
    lock_path = output_root / "promoted_afflec_body_source_lock.json"
    _write_json(
        lock_path,
        {
            "body_receipt_path": str(body_receipt),
            "body_receipt_hash": _sha256_file(body_receipt),
            "body_mesh_path": str(body_mesh),
            "body_mesh_hash": _sha256_file(body_mesh),
            "forward_object_path": str(forward_object),
            "forward_object_hash": forward_hash,
            "forward_object_topology_label": "B_v9438",
        },
    )

    transfer_receipt = output_root / "back_transfer_acceptance_receipt.json"
    transfer_map = output_root / "B_v9438_to_promoted_afflec_body_vertex_map.npz"
    _run(
        [
            python,
            "scripts/p3_back_transfer_acceptance.py",
            "--source-mesh",
            str(forward_object),
            "--target-mesh",
            str(body_mesh),
            "--target-body-receipt",
            str(body_receipt),
            "--out-receipt",
            str(transfer_receipt),
            "--out-map",
            str(transfer_map),
            "--source-topology-label",
            "B_v9438",
        ]
    )

    basis_path = output_root / "basis.npz"
    basis_receipt = output_root / "basis_receipt.json"
    components = _sample_component_count(rom_samples)
    _run(
        [
            python,
            "scripts/generate_canonical_basis.py",
            "--vertices",
            str(body_mesh),
            "--body-receipt",
            str(body_receipt),
            "--components",
            str(components),
            "--harmonics",
            "5",
            "--output",
            str(basis_path),
            "--receipt-output",
            str(basis_receipt),
            "--source-mesh",
            "promoted_afflec_curated_body",
        ]
    )
    promoted, promotion, blocked = _receipt_promoted(basis_receipt)
    if not promoted:
        _record_blocker(summary_path=summary_path, stage="basis", receipt_path=basis_receipt)
        print(f"Blocked at basis receipt: promotion={promotion}, blocked={blocked}")
        return

    rom_fields = output_root / "rom_fields.npz"
    rom_field_receipt = output_root / "rom_field_receipt.json"
    try:
        _run(
            [
                python,
                "examples/rom_aggregate_from_samples.py",
                "--samples",
                str(rom_samples),
                "--basis",
                str(basis_path),
                "--basis-receipt",
                str(basis_receipt),
                "--output-dir",
                str(output_root),
                "--out-rom-fields",
                str(rom_fields),
                "--out-rom-field-receipt",
                str(rom_field_receipt),
                "--save-costs",
                str(output_root / "diagnostic_rom_seam_costs.npz"),
                "--allow-synthetic-promotion",
            ]
        )
    except subprocess.CalledProcessError as exc:
        _record_blocker(
            summary_path=summary_path,
            stage="rom_field",
            receipt_path=rom_field_receipt if rom_field_receipt.exists() else None,
            error=exc.stderr or exc.stdout,
        )
        print("Blocked at ROM field aggregation.")
        return
    promoted, promotion, blocked = _receipt_promoted(rom_field_receipt)
    if not promoted:
        _record_blocker(
            summary_path=summary_path, stage="rom_field", receipt_path=rom_field_receipt
        )
        print(f"Blocked at ROM field receipt: promotion={promotion}, blocked={blocked}")
        return

    seam_costs = output_root / "seam_costs.npz"
    seam_cost_receipt = output_root / "seam_cost_receipt.json"
    _run(
        [
            python,
            "scripts/compute_seam_costs.py",
            "--body-receipt",
            str(body_receipt),
            "--rom-field-receipt",
            str(rom_field_receipt),
            "--rom-fields",
            str(rom_fields),
            "--mesh",
            str(body_mesh),
            "--out-costs",
            str(seam_costs),
            "--out-seam-cost-receipt",
            str(seam_cost_receipt),
            "--solve-domain",
            solve_domain,
        ]
    )
    promoted, promotion, blocked = _receipt_promoted(seam_cost_receipt)
    if not promoted:
        _record_blocker(
            summary_path=summary_path, stage="seam_cost", receipt_path=seam_cost_receipt
        )
        print(f"Blocked at seam cost receipt: promotion={promotion}, blocked={blocked}")
        return

    solver_dir = output_root / "solver"
    cut_candidates = solver_dir / "dart_relief_candidates.json"
    _run(
        [
            python,
            "scripts/propose_dart_relief_cuts.py",
            "--mesh",
            str(body_mesh),
            "--out-json",
            str(cut_candidates),
        ]
    )
    solver_receipt = solver_dir / "solver_promotion_receipt.json"
    _run(
        [
            python,
            "scripts/solve_seams.py",
            "--seam-cost-receipt",
            str(seam_cost_receipt),
            "--costs",
            str(seam_costs),
            "--mesh",
            str(body_mesh),
            "--out-dir",
            str(solver_dir),
            "--solver-mode",
            "cut_graph",
            "--target-panel-count",
            "4",
            "--dart-relief-candidates",
            str(cut_candidates),
        ]
    )
    promoted, promotion, blocked = _receipt_promoted(solver_receipt)
    if not promoted:
        _record_blocker(summary_path=summary_path, stage="solver", receipt_path=solver_receipt)
        print(f"Blocked at solver receipt: promotion={promotion}, blocked={blocked}")
        return

    cut_topology_receipt = solver_dir / "cut_topology_receipt.json"
    _run(
        [
            python,
            "scripts/validate_cut_topology.py",
            "--solver-receipt",
            str(solver_receipt),
            "--seam-edges",
            str(solver_dir / "seam_edges.npz"),
            "--mesh",
            str(body_mesh),
            "--out-cut-topology-receipt",
            str(cut_topology_receipt),
        ]
    )
    cut_topology_promoted, cut_topology_promotion, cut_topology_blocked = _receipt_promoted(
        cut_topology_receipt
    )
    metric_correction_receipt = solver_dir / "metric_correction_receipt.json"
    metric_correction_promoted = False
    typed_operator_count = _typed_operator_count(cut_topology_receipt)
    if cut_topology_promoted and typed_operator_count > 0:
        metric_command = [
            python,
            "scripts/emit_metric_correction_receipt.py",
            "--solver-receipt",
            str(solver_receipt),
            "--cut-topology-receipt",
            str(cut_topology_receipt),
            "--seam-edges",
            str(solver_dir / "seam_edges.npz"),
            "--out-metric-correction-receipt",
            str(metric_correction_receipt),
        ]
        corrections_path = solver_dir / "corrections.json"
        if corrections_path.exists():
            metric_command.extend(["--corrections", str(corrections_path)])
        _run(metric_command)
        metric_correction_promoted, metric_promotion, metric_blocked = _receipt_promoted(
            metric_correction_receipt
        )
        if not metric_correction_promoted:
            _record_blocker(
                summary_path=summary_path,
                stage="metric_correction",
                receipt_path=metric_correction_receipt,
                extras={
                    "cut_topology_receipt": str(cut_topology_receipt),
                    "typed_operator_count": typed_operator_count,
                    "dart_relief_candidates": str(cut_candidates),
                },
            )
            print(
                "Blocked at metric correction receipt: "
                f"promotion={metric_promotion}, blocked={metric_blocked}"
            )
            return
    unwrap_dir = output_root / "panel_unwrap"
    panel_receipt = unwrap_dir / "panel_unwrap_receipt.json"
    unwrap_command = [
        python,
        "scripts/unwrap_panels.py",
        "--solver-receipt",
        str(solver_receipt),
        "--seam-edges",
        str(solver_dir / "seam_edges.npz"),
        "--mesh",
        str(body_mesh),
        "--out-dir",
        str(unwrap_dir),
        "--solver",
        "lscm",
    ]
    if cut_topology_promoted:
        unwrap_command.extend(["--cut-topology-receipt", str(cut_topology_receipt)])
    corrections_path = solver_dir / "corrections.json"
    if corrections_path.exists():
        unwrap_command.extend(["--corrections", str(corrections_path)])
    if metric_correction_promoted:
        unwrap_command.extend(["--metric-correction-receipt", str(metric_correction_receipt)])
    try:
        _run(unwrap_command)
    except subprocess.CalledProcessError as exc:
        _record_blocker(
            summary_path=summary_path,
            stage="panel_unwrap",
            receipt_path=panel_receipt if panel_receipt.exists() else None,
            error=exc.stderr or exc.stdout,
        )
        print("Blocked at panel unwrap.")
        return
    promoted, promotion, blocked = _receipt_promoted(panel_receipt)
    diagnostic_summary = _render_panel_diagnostics(
        python=python,
        panel_receipt=panel_receipt,
        panel_uvs=unwrap_dir / "panel_uvs.npz",
        mesh=body_mesh,
        seam_edges=solver_dir / "seam_edges.npz",
        output_dir=unwrap_dir / "diagnostics",
    )
    if not cut_topology_promoted:
        _record_blocker(
            summary_path=summary_path,
            stage="cut_topology",
            receipt_path=cut_topology_receipt,
            extras={
                "cut_topology_promotion": cut_topology_promotion,
                "cut_topology_blocked_consumers": cut_topology_blocked,
                "dart_relief_candidates": str(cut_candidates),
                "panel_pattern_diagnostics": str(diagnostic_summary),
            },
        )
        print(
            "Blocked at cut topology receipt: "
            f"promotion={cut_topology_promotion}, blocked={cut_topology_blocked}"
        )
        return
    if not promoted:
        _record_blocker(
            summary_path=summary_path,
            stage="panel_unwrap",
            receipt_path=panel_receipt,
            extras={"panel_pattern_diagnostics": str(diagnostic_summary)},
        )
        print(f"Blocked at panel unwrap receipt: promotion={promotion}, blocked={blocked}")
        return

    _write_json(
        summary_path,
        {
            "status": "complete",
            "lock_path": str(lock_path),
            "back_transfer_receipt": str(transfer_receipt),
            "basis_receipt": str(basis_receipt),
            "rom_field_receipt": str(rom_field_receipt),
            "seam_cost_receipt": str(seam_cost_receipt),
            "solver_receipt": str(solver_receipt),
            "cut_topology_receipt": str(cut_topology_receipt),
            "metric_correction_receipt": str(metric_correction_receipt)
            if metric_correction_receipt.exists()
            else None,
            "dart_relief_candidates": str(cut_candidates),
            "panel_unwrap_receipt": str(panel_receipt),
            "panel_pattern_diagnostics": str(diagnostic_summary),
        },
    )
    print(f"P3 chain completed through panel unwrap: {summary_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/p3_afflec_transfer_chain_20260604"),
    )
    parser.add_argument("--promoted-root", type=Path, default=DEFAULT_PROMOTED_ROOT)
    parser.add_argument("--forward-object", type=Path, default=DEFAULT_FORWARD_OBJECT)
    parser.add_argument("--rom-samples", type=Path, default=DEFAULT_ROM_SAMPLES)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_chain(
        output_root=args.out_root,
        promoted_root=args.promoted_root,
        forward_object=args.forward_object,
        rom_samples=args.rom_samples,
        python=args.python,
    )


if __name__ == "__main__":
    main()
