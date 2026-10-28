from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "verify_rom_runtime.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_rom_runtime", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_rom_runtime_reports_success(monkeypatch, capsys):
    module = _load()

    fake_modules = {
        "torch": SimpleNamespace(__version__="2.test"),
        "smplx": SimpleNamespace(__version__="0.test"),
    }
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_modules[name])
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="usage", stderr=""),
    )

    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["torch"] == "2.test"
    assert payload["smplx"] == "0.test"
    assert payload["sampler_help_exit"] == 0


def test_verify_rom_runtime_fails_on_missing_smplx(monkeypatch, capsys):
    module = _load()

    def fake_import(name):
        if name == "smplx":
            raise ModuleNotFoundError("No module named 'smplx'")
        return SimpleNamespace(__version__="2.test")

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="missing"),
    )

    assert module.main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert any("smplx: ModuleNotFoundError" in item for item in payload["failures"])
    assert any("sampler entrypoint" in item for item in payload["failures"])
