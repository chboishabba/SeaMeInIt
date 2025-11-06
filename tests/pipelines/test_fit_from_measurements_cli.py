from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

import smii.pipelines.fit_from_measurements as cli


class RecordingSave:
    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.paths: list[Path] = []

    def __call__(self, result: cli.FitResult, output_path: Path) -> None:
        self.payloads.append(result.to_dict())
        self.paths.append(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict()), encoding="utf-8")


@pytest.mark.parametrize("argv", [["--images", "front", "side"], ["--images", "front"]])
def test_cli_routes_image_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: Sequence[str]):
    called = {}

    def fake_extract(paths):
        called["paths"] = tuple(paths)
        return {
            "height": 170.0,
            "chest_circumference": 96.5,
            "waist_circumference": 82.3,
        }

    def fake_fit(measurements, **kwargs):
        called["measurements"] = measurements
        return cli.FitResult(
            betas=np.zeros(10),
            scale=1.0,
            translation=np.zeros(3),
            residual=0.0,
            measurements_used=tuple(sorted(measurements)),
        )

    recorder = RecordingSave()

    monkeypatch.setattr(cli, "extract_measurements_from_afflec_images", fake_extract)
    monkeypatch.setattr(cli, "fit_smplx_from_measurements", fake_fit)
    monkeypatch.setattr(cli, "save_fit", recorder)

    output = tmp_path / "out.json"
    exit_code = cli.main([*argv, "--output", output])

    assert exit_code == 0
    assert called["paths"] == tuple(Path(arg) for arg in argv[1:])
    assert called["measurements"] == {
        "height": 170.0,
        "chest_circumference": 96.5,
        "waist_circumference": 82.3,
    }
    assert recorder.paths == [output]


def test_cli_requires_measurement_source(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(SystemExit):
        cli.main([])
