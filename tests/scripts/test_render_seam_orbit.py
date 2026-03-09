import importlib.util
from pathlib import Path

import numpy as np


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_seam_orbit_cleans_temporary_frames(tmp_path: Path, monkeypatch) -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "render_seam_orbit.py"
    module = _load_script_module(script_path, "render_seam_orbit")

    mesh_path = tmp_path / "body.npz"
    np.savez(mesh_path, vertices=np.zeros((3, 3), dtype=float), faces=np.array([[0, 1, 2]], dtype=int))

    def fake_render_frame(*args, output: Path, **kwargs) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"frame")

    def fake_encode_orbit(frames_dir: Path, gif_path: Path, webm_path: Path) -> None:
        gif_path.write_bytes(b"gif")
        webm_path.write_bytes(b"webm")

    monkeypatch.setattr(module, "_render_frame", fake_render_frame)
    monkeypatch.setattr(module, "_encode_orbit", fake_encode_orbit)

    out_dir = tmp_path / "renders"
    module.main(
        [
            "--mesh",
            str(mesh_path),
            "--out-dir",
            str(out_dir),
            "--stem",
            "demo",
            "--timestamp",
            "20260309_170000",
            "--frames",
            "2",
        ]
    )

    assert not (out_dir / "demo__orbit_frames__20260309_170000").exists()
    assert (out_dir / "demo__front_3q__20260309_170000.png").exists()
    assert (out_dir / "demo__orbit__20260309_170000.gif").exists()
    assert (out_dir / "demo__orbit__20260309_170000.webm").exists()
