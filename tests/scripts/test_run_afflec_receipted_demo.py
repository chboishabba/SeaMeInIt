from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _load_runner_module():
    path = Path("scripts/run_afflec_receipted_demo.py")
    spec = importlib.util.spec_from_file_location("run_afflec_receipted_demo", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _flag_value(command: Sequence[str], flag: str) -> Path:
    return Path(command[command.index(flag) + 1])


def _write_receipt(path: Path, *, promotion: int = 1, notes: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"promotion": promotion, "notes": notes}),
        encoding="utf-8",
    )


def _has_command(command: Sequence[str], name: str) -> bool:
    return any(part.endswith(name) for part in command)


def test_dry_run_prints_plan_without_writing_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "afflec_receipted"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_afflec_receipted_demo.py",
            "--output",
            str(out_dir),
            "--allow-synthetic-promotion",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "[DRY-RUN]" in result.stdout
    assert "diagnostic_seam_costs.npz" in result.stdout
    assert "run_manifest.json" not in result.stdout
    assert not out_dir.exists()


def test_runner_writes_manifest_for_promoted_chain(tmp_path: Path) -> None:
    runner_module = _load_runner_module()
    out_dir = tmp_path / "run"
    calls: list[Sequence[str]] = []

    def fake_runner(command: Sequence[str], cwd: Path) -> int:
        calls.append(command)
        if "afflec-demo" in command:
            _write_receipt(_flag_value(command, "--output") / "body_carrier_receipt.json")
        elif _has_command(command, "generate_canonical_basis.py"):
            _write_receipt(_flag_value(command, "--receipt-output"))
        elif _has_command(command, "rom_aggregate_from_samples.py"):
            _write_receipt(_flag_value(command, "--out-rom-field-receipt"))
        elif _has_command(command, "compute_seam_costs.py"):
            _write_receipt(_flag_value(command, "--out-seam-cost-receipt"))
        elif _has_command(command, "solve_seams.py"):
            _write_receipt(_flag_value(command, "--out-solver-receipt"))
        elif _has_command(command, "unwrap_panels.py"):
            _write_receipt(_flag_value(command, "--out-panel-receipt"))
        elif _has_command(command, "generate_manufacturing_artifacts.py"):
            _write_receipt(_flag_value(command, "--out-manufacturing-receipt"))
        else:  # pragma: no cover - defensive
            raise AssertionError(command)
        return 0

    exit_code = runner_module.run_demo(
        run_dir=out_dir,
        python=sys.executable,
        allow_synthetic_promotion=True,
        manufacturing_method="home_sewing",
        force=False,
        dry_run=False,
        runner=fake_runner,
    )

    assert exit_code == 0
    assert len(calls) == 7
    manifest = json.loads((out_dir / "run_manifest.json").read_text("utf-8"))
    assert manifest["exit_code"] == 0
    assert manifest["first_blocker"] is None
    assert manifest["can_manufacture"]
    assert manifest["gates"]["correspondence"]["promotion"] == 0
    assert "native A_v3240" in manifest["gates"]["correspondence"]["notes"]
    assert manifest["gates"]["manufacture"]["receipt"] == "manufacturing/manufacturing_receipt.json"
    assert manifest["gates"]["manufacture"]["receipt_hash"]


def test_runner_stops_at_first_non_promoted_receipt(tmp_path: Path) -> None:
    runner_module = _load_runner_module()
    out_dir = tmp_path / "run"

    def fake_runner(command: Sequence[str], cwd: Path) -> int:
        if "afflec-demo" in command:
            _write_receipt(_flag_value(command, "--output") / "body_carrier_receipt.json")
        elif _has_command(command, "generate_canonical_basis.py"):
            _write_receipt(_flag_value(command, "--receipt-output"))
        elif _has_command(command, "rom_aggregate_from_samples.py"):
            _write_receipt(
                _flag_value(command, "--out-rom-field-receipt"),
                promotion=0,
                notes="synthetic promotion not allowed",
            )
        else:  # pragma: no cover - runner must stop before later gates
            raise AssertionError(f"unexpected later command: {command}")
        return 0

    exit_code = runner_module.run_demo(
        run_dir=out_dir,
        python=sys.executable,
        allow_synthetic_promotion=False,
        manufacturing_method="home_sewing",
        force=False,
        dry_run=False,
        runner=fake_runner,
    )

    assert exit_code == 1
    manifest = json.loads((out_dir / "run_manifest.json").read_text("utf-8"))
    assert manifest["exit_code"] == 1
    assert manifest["first_blocker"] == "rom_field"
    assert not manifest["can_manufacture"]
    assert manifest["gates"]["rom_field"]["promotion"] == 0
    assert "synthetic promotion not allowed" in manifest["gates"]["rom_field"]["notes"]
    assert manifest["gates"]["seam_cost"]["receipt"] is None


def test_runner_treats_missing_receipt_as_hard_failure(tmp_path: Path) -> None:
    runner_module = _load_runner_module()
    out_dir = tmp_path / "run"

    def fake_runner(command: Sequence[str], cwd: Path) -> int:
        return 0

    exit_code = runner_module.run_demo(
        run_dir=out_dir,
        python=sys.executable,
        allow_synthetic_promotion=True,
        manufacturing_method="home_sewing",
        force=False,
        dry_run=False,
        runner=fake_runner,
    )

    assert exit_code == 2
    manifest = json.loads((out_dir / "run_manifest.json").read_text("utf-8"))
    assert manifest["exit_code"] == 2
    assert manifest["first_blocker"] == "body"
    assert "Expected receipt was not emitted" in manifest["gates"]["body"]["notes"]
