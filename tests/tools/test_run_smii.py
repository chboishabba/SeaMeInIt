"""Smoke checks for the run_smii.sh helper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "run_smii.sh"


def run_helper(*args: str) -> subprocess.CompletedProcess[str]:
    """Execute the helper with a controllable environment."""

    env = os.environ.copy()
    env["SMII_SKIP_BOOTSTRAP"] = "1"
    return subprocess.run(
        [str(SCRIPT), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_usage_documentation_mentions_sentinel():
    result = run_helper("--help")
    assert "--" in result.stdout


def test_dash_prefixed_command_executes():
    result = run_helper("--", "printf", "%s", "ok")
    assert result.stdout == "ok"
