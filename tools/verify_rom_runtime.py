#!/usr/bin/env python3
"""Verify the runtime required by smii.rom.sampler_real."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


def _version(module: object) -> str:
    value = getattr(module, "__version__", None)
    return str(value) if value is not None else "unknown"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    versions: dict[str, str] = {}

    for name in ("torch", "smplx"):
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
        else:
            versions[name] = _version(module)

    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    command = [sys.executable, "-m", "smii.rom.sampler_real", "--help"]
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        failures.append(
            "sampler entrypoint: "
            f"exit={completed.returncode}; stderr={completed.stderr.strip()!r}"
        )

    payload = {
        "python": sys.executable,
        "torch": versions.get("torch"),
        "smplx": versions.get("smplx"),
        "sampler_help_exit": completed.returncode,
        "ok": not failures,
        "failures": failures,
    }
    print(json.dumps(payload, sort_keys=True))

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
