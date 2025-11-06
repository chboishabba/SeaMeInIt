"""Command line demo for fitting SMPL-X parameters from manual measurements.

Measurement identifiers and required fields mirror the canonical definitions in
``schemas/body_unified.yaml``. Use ``schemas.validators.load_measurement_catalog``
to inspect the full list of supported measurements.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from smii.pipelines.fit_from_measurements import fit_smplx_from_measurements, save_fit
from smii.pipelines.fit_from_scan import create_parametric_mesh

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - optional dependency
    o3d = None  # type: ignore[assignment]

DEFAULT_MEASUREMENTS: dict[str, float] = {
    "height": 176.0,
    "chest_circumference": 98.0,
    "waist_circumference": 82.0,
    "hip_circumference": 101.0,
    "shoulder_width": 43.5,
    "arm_length": 61.0,
    "inseam_length": 80.0,
    "thigh_circumference": 57.0,
}


def _load_measurements(path: Path | None) -> Mapping[str, float]:
    if path is None:
        return DEFAULT_MEASUREMENTS
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive
        raise TypeError("Input file must contain a JSON object of measurement names and values.")
    return {name: float(value) for name, value in payload.items()}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurements",
        type=Path,
        help="Optional path to a JSON file containing manual measurements.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/meshes"),
        help="Directory in which to store fitted parameters and example mesh.",
    )
    args = parser.parse_args()

    measurements = _load_measurements(args.measurements)
    result = fit_smplx_from_measurements(measurements)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_path = args.output_dir / "manual_measurement_fit.json"
    save_fit(result, parameter_path)
    print(f"Saved fitted parameters to {parameter_path}")

    if o3d is not None:
        mesh = create_parametric_mesh(result.betas)
        mesh_path = args.output_dir / "manual_measurement_fit.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=True)
        print(f"Saved illustrative mesh to {mesh_path}")
    else:
        print("open3d is not installed; skipping mesh export.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
