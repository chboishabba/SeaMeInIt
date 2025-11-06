"""Fit SMPL-X parameters from manual body measurements."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "data" / "schemas" / "body_measurements.json"
SMPLX_NUM_BETAS = 10


@dataclass(frozen=True)
class MeasurementModel:
    """Describes how a single measurement influences SMPL-X shape coefficients."""

    name: str
    mean: float
    std: float
    weights: Sequence[float]

    def normalized(self, value: float) -> float:
        return (value - self.mean) / self.std


MEASUREMENT_MODELS: tuple[MeasurementModel, ...] = (
    MeasurementModel("height", 170.0, 7.5, (0.45, 0.05, 0.02, 0.0, 0.03, 0.0, 0.01, 0.0, 0.0, 0.0)),
    MeasurementModel("chest_circumference", 95.0, 8.0, (0.6, 0.2, 0.05, 0.03, 0.0, 0.02, 0.05, 0.01, 0.0, 0.0)),
    MeasurementModel("waist_circumference", 80.0, 7.0, (0.1, 0.55, 0.05, 0.05, 0.1, 0.02, 0.0, 0.04, 0.03, 0.0)),
    MeasurementModel("hip_circumference", 98.0, 8.5, (0.05, 0.05, 0.6, 0.08, 0.05, 0.02, 0.04, 0.02, 0.02, 0.0)),
    MeasurementModel("shoulder_width", 42.0, 2.5, (0.15, 0.05, 0.02, 0.4, 0.02, 0.1, 0.05, 0.1, 0.0, 0.0)),
    MeasurementModel("neck_circumference", 36.0, 2.5, (0.18, 0.02, 0.0, 0.05, 0.35, 0.05, 0.02, 0.0, 0.05, 0.0)),
    MeasurementModel("arm_length", 60.0, 4.5, (0.05, 0.05, 0.0, 0.12, 0.02, 0.3, 0.08, 0.12, 0.05, 0.0)),
    MeasurementModel("inseam_length", 78.0, 5.0, (0.05, 0.04, 0.08, 0.02, 0.0, 0.1, 0.4, 0.08, 0.08, 0.05)),
    MeasurementModel("thigh_circumference", 55.0, 5.0, (0.0, 0.05, 0.45, 0.02, 0.0, 0.1, 0.3, 0.05, 0.02, 0.01)),
    MeasurementModel("calf_circumference", 37.0, 3.5, (0.0, 0.02, 0.2, 0.01, 0.0, 0.05, 0.1, 0.4, 0.1, 0.02)),
    MeasurementModel("bicep_circumference", 30.0, 3.0, (0.1, 0.05, 0.02, 0.15, 0.02, 0.25, 0.05, 0.12, 0.1, 0.02)),
    MeasurementModel("wrist_circumference", 16.0, 1.0, (0.02, 0.01, 0.0, 0.06, 0.03, 0.15, 0.02, 0.2, 0.15, 0.1)),
)


@dataclass(frozen=True)
class FitResult:
    """Container for the fitted SMPL-X parameters."""

    betas: np.ndarray
    scale: float
    translation: np.ndarray
    residual: float
    measurements_used: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "betas": self.betas.tolist(),
            "scale": float(self.scale),
            "translation": self.translation.tolist(),
            "residual": float(self.residual),
            "measurements_used": list(self.measurements_used),
        }


def load_schema(path: Path | None = None) -> dict:
    schema_path = path or SCHEMA_PATH
    with schema_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def required_measurements(schema: Mapping[str, Iterable[Mapping[str, object]]]) -> set[str]:
    manual = schema.get("manual_measurements", [])
    return {
        item["name"]
        for item in manual
        if isinstance(item, Mapping) and item.get("required") is True and isinstance(item.get("name"), str)
    }


def validate_measurements(measurements: Mapping[str, float], schema: Mapping[str, object] | None = None) -> None:
    schema = schema or load_schema()
    missing = [name for name in required_measurements(schema) if name not in measurements]
    if missing:
        raise ValueError(f"Missing required measurements: {', '.join(sorted(missing))}")


def models_by_name(models: Sequence[MeasurementModel]) -> dict[str, MeasurementModel]:
    return {model.name: model for model in models}


def fit_shape_coefficients(
    measurements: Mapping[str, float],
    *,
    models: Sequence[MeasurementModel] = MEASUREMENT_MODELS,
    num_shape_coeffs: int = SMPLX_NUM_BETAS,
) -> tuple[np.ndarray, tuple[str, ...], float]:
    available = {m.name: m for m in models if m.name in measurements}
    if not available:
        raise ValueError("At least one measurement is required to fit SMPL-X shape coefficients.")

    used_names = tuple(sorted(available))
    A = np.vstack([available[name].weights[:num_shape_coeffs] for name in used_names])
    b = np.array([available[name].normalized(float(measurements[name])) for name in used_names])

    coeffs, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    if residuals.size:
        rms = float(np.sqrt(residuals[0] / len(used_names)))
    else:
        reconstruction = A @ coeffs
        rms = float(np.sqrt(np.mean((reconstruction - b) ** 2)))
    return coeffs, used_names, rms


def fit_smplx_from_measurements(
    measurements: Mapping[str, float],
    *,
    schema_path: Path | None = None,
    models: Sequence[MeasurementModel] = MEASUREMENT_MODELS,
    num_shape_coeffs: int = SMPLX_NUM_BETAS,
) -> FitResult:
    """Compute SMPL-X shape parameters from manual measurements."""

    schema = load_schema(schema_path)
    validate_measurements(measurements, schema)

    coeffs, used_names, rms = fit_shape_coefficients(
        measurements,
        models=models,
        num_shape_coeffs=num_shape_coeffs,
    )

    scale = float(measurements.get("height", models_by_name(models)["height"].mean)) / models_by_name(models)["height"].mean
    translation = np.zeros(3, dtype=float)

    return FitResult(
        betas=coeffs,
        scale=scale,
        translation=translation,
        residual=rms,
        measurements_used=used_names,
    )


def save_fit(result: FitResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result.to_dict(), stream, indent=2)


def _load_measurements_from_json(path: Path) -> Mapping[str, float]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("Measurement input must be a JSON object.")
    return {key: float(value) for key, value in payload.items()}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fit SMPL-X parameters from manual measurements.")
    parser.add_argument("input", type=Path, help="Path to a JSON file containing measurement values.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/meshes/manual_measurement_fit.json"),
        help="Where to store the fitted parameter dictionary.",
    )
    parser.add_argument(
        "--coeff-count",
        type=int,
        default=SMPLX_NUM_BETAS,
        help="Number of shape coefficients to estimate (default: 10).",
    )
    args = parser.parse_args(argv)

    measurements = _load_measurements_from_json(args.input)
    result = fit_smplx_from_measurements(measurements, num_shape_coeffs=args.coeff_count)
    save_fit(result, args.output)
    print(f"Saved SMPL-X parameters to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
