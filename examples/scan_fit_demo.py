"""Command line demo for fitting a scan point cloud to the SMPL-X template."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from smii.pipelines.fit_from_measurements import fit_smplx_from_measurements, save_fit
from smii.pipelines.fit_from_scan import ICPSettings, RegistrationResult, fit_scan_to_smplx


def _load_betas(path: Path) -> Sequence[float]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, Mapping) and "betas" in payload:
        values = payload["betas"]
        if not isinstance(values, list):  # pragma: no cover - defensive
            raise TypeError("'betas' field must be a list of floats.")
        return [float(value) for value in values]
    if isinstance(payload, list):
        return [float(value) for value in payload]
    raise TypeError("Beta file must contain either a list of floats or an object with a 'betas' key.")


def _load_measurements(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):  # pragma: no cover - defensive
        raise TypeError("Measurement file must contain a JSON object.")
    return {key: float(value) for key, value in payload.items()}


def _fit_betas_from_measurements(measurement_path: Path | None, output_dir: Path) -> np.ndarray:
    if measurement_path is None:
        return np.zeros(10, dtype=float)
    result = fit_smplx_from_measurements(_load_measurements(measurement_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_fit(result, output_dir / "measurement_fit_parameters.json")
    return result.betas


def _resolve_betas(
    betas_path: Path | None,
    measurement_path: Path | None,
    output_dir: Path,
) -> Iterable[float]:
    if betas_path is not None:
        return _load_betas(betas_path)
    return _fit_betas_from_measurements(measurement_path, output_dir)


def _settings_from_args(args: "argparse.Namespace") -> ICPSettings:
    return ICPSettings(
        max_correspondence_distance=args.max_correspondence_distance,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
    )


def _save_registration(result: RegistrationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters_path = output_dir / "scan_fit_parameters.json"
    with parameters_path.open("w", encoding="utf-8") as stream:
        json.dump(result.to_dict(), stream, indent=2)
    print(f"Saved registration summary to {parameters_path}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("point_cloud", type=Path, help="Path to the scan point cloud (PLY/PCD/OBJ).")
    parser.add_argument(
        "--betas",
        type=Path,
        help="Optional path to a JSON file containing an array of SMPL-X betas.",
    )
    parser.add_argument(
        "--measurements",
        type=Path,
        help="Optional path to manual measurement JSON for estimating betas.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/meshes"),
        help="Directory in which to store fitted meshes and parameters.",
    )
    parser.add_argument(
        "--max-correspondence-distance",
        type=float,
        default=0.02,
        help="Maximum distance between corresponded points during ICP.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum number of ICP iterations.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Convergence tolerance for ICP fitness and RMSE deltas.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir

    if args.betas is None and args.measurements is None:
        print("No betas or measurements provided; defaulting to zero shape coefficients.")

    betas = _resolve_betas(args.betas, args.measurements, output_dir)
    settings = _settings_from_args(args)

    mesh_output = output_dir / "scan_fit_mesh.ply"
    result = fit_scan_to_smplx(
        args.point_cloud,
        betas=betas,
        settings=settings,
        output_mesh_path=mesh_output,
    )

    _save_registration(result, output_dir)
    print(f"Saved transformed mesh to {mesh_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
