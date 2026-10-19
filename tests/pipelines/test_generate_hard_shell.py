from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from smii.pipelines.generate_hard_shell import (
    generate_hard_shell,
    generate_hard_shell_panels,
    load_body_record,
)
from suit_hard import HardShellSegmentationOptions


def test_generate_hard_shell_from_npz(tmp_path: Path) -> None:
    body_path = tmp_path / "body.npz"
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=int)
    np.savez(body_path, vertices=vertices, faces=faces)

    shell_path = generate_hard_shell(
        body_path,
        output_dir=tmp_path / "shell",
        default_thickness=0.01,
        allow_non_watertight=True,
    )

    with np.load(shell_path) as payload:
        assert payload["vertices"].shape == vertices.shape
        np.testing.assert_array_equal(payload["faces"], faces)
        np.testing.assert_allclose(payload["thickness"], np.full(3, 0.01))
    metadata = json.loads(
        (shell_path.parent / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["thickness_profile_source"] == "uniform"


def test_generate_articulation_panels_from_joint_record(tmp_path: Path) -> None:
    body_path = tmp_path / "joints.json"
    body_path.write_text(
        json.dumps(
            {
                "joint_names": [
                    "left_shoulder",
                    "left_elbow",
                    "left_wrist",
                ],
                "joint_positions": [
                    [0.0, 0.0, 0.0],
                    [0.0, -0.4, 0.0],
                    [0.0, -0.8, 0.0],
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest_path = generate_hard_shell_panels(
        body_path,
        output_dir=tmp_path / "panels",
        options=HardShellSegmentationOptions(boundary_points=8),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["metadata"]["panel_count"] == 2
    assert {panel["name"] for panel in manifest["panels"]} == {
        "left_shoulder_panel",
        "left_elbow_panel",
    }
    assert (manifest_path.parent / "left_shoulder_panel.npz").exists()


def test_load_body_record_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "body.json"
    path.write_text("[]", encoding="utf-8")

    try:
        load_body_record(path)
    except TypeError as exc:
        assert "must be an object" in str(exc)
    else:  # pragma: no cover - assertion helper
        raise AssertionError("Expected non-object body JSON to be rejected")
