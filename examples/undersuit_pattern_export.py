"""Example configuration for exporting undersuit patterns."""

from __future__ import annotations

from pathlib import Path

from exporters.patterns import PatternExporter


def example_bodysuit_export(base_dir: Path) -> dict[str, Path]:
    """Generate a single-layer bodysuit pattern export."""

    mesh_payload = {
        "panels": [
            {
                "name": "front",
                "vertices": [
                    (-0.15, 1.4, 0.02),
                    (0.15, 1.4, 0.0),
                    (0.2, 0.8, 0.05),
                    (-0.2, 0.8, -0.03),
                ],
                "faces": [(0, 1, 2), (0, 2, 3)],
            },
            {
                "name": "back",
                "vertices": [
                    (-0.17, 1.3, -0.05),
                    (0.17, 1.3, -0.01),
                    (0.22, 0.7, -0.02),
                    (-0.22, 0.7, -0.07),
                ],
                "faces": [(0, 1, 2), (0, 2, 3)],
            },
        ]
    }

    seams = {
        "front": {"seam_allowance": 0.012},
        "back": {"seam_allowance": 0.01},
    }

    exporter = PatternExporter(scale=1.0, seam_allowance=0.01)
    return exporter.export(mesh_payload, seams, output_dir=base_dir / "bodysuit")


def example_layered_export(base_dir: Path) -> dict[str, Path]:
    """Generate a multi-layer pattern export showing outer shell and lining."""

    mesh_payload = {
        "panels": [
            {
                "name": "outer_shell",
                "vertices": [
                    (-0.3, 1.6, 0.03),
                    (0.3, 1.6, -0.02),
                    (0.35, 0.6, 0.0),
                    (-0.35, 0.6, 0.02),
                ],
                "faces": [(0, 1, 2), (0, 2, 3)],
            },
            {
                "name": "inner_lining",
                "vertices": [
                    (-0.28, 1.55, 0.01),
                    (0.28, 1.55, -0.03),
                    (0.31, 0.65, -0.02),
                    (-0.31, 0.65, 0.01),
                ],
                "faces": [(0, 1, 2), (0, 2, 3)],
            },
        ]
    }

    seams = {
        "outer_shell": {"seam_allowance": 0.015},
        "inner_lining": {"seam_allowance": 0.008},
    }

    exporter = PatternExporter(scale=0.98, seam_allowance=0.01)
    return exporter.export(
        mesh_payload,
        seams,
        output_dir=base_dir / "layered",
        metadata={"variant": "two_layer"},
    )


if __name__ == "__main__":  # pragma: no cover - example script
    base = Path("exports/patterns/examples")
    results = {
        "bodysuit": example_bodysuit_export(base),
        "layered": example_layered_export(base),
    }
    for name, files in results.items():
        print(name, "->", {fmt: str(path) for fmt, path in files.items()})

