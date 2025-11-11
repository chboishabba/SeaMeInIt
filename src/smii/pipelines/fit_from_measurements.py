"""Fit body model parameters from manual body measurements."""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path

import numpy as np

from schemas.validators import load_measurement_catalog

from pipelines.measurement_inference import (
    GaussianMeasurementModel,
    MeasurementReport,
    load_default_model,
)

from .fit_from_images import extract_measurements_from_afflec_images

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "data" / "schemas" / "body_measurements.json"
MEASUREMENT_MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "measurement_models"
DEFAULT_BACKEND = "smplx"


@dataclass(frozen=True)
class MeasurementModel:
    """Describe how a single measurement influences body shape coefficients."""

    name: str
    mean: float
    std: float
    weights: Sequence[float]

    def normalized(self, value: float) -> float:
        return (value - self.mean) / self.std


@dataclass(frozen=True)
class BackendMeasurementConfig:
    """Configuration payload for fitting measurements to a body model backend."""

    backend: str
    num_betas: int
    scale_measurement: str
    models: tuple[MeasurementModel, ...]


_CONFIG_CACHE: dict[str, BackendMeasurementConfig] = {}


def _resolve_backend_path(backend: str) -> Path:
    normalized = backend.lower()
    candidates = [
        MEASUREMENT_MODEL_DIR / f"{normalized}.json",
        MEASUREMENT_MODEL_DIR / f"{normalized}.yaml",
        MEASUREMENT_MODEL_DIR / f"{normalized}.yml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No measurement model configuration found for backend '{backend}'. "
        f"Searched: {[str(path) for path in candidates]}"
    )


def _load_backend_payload(path: Path) -> Mapping[str, object]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        yaml_module = importlib.import_module("yaml")
        data = yaml_module.safe_load(text)
    else:
        raise ValueError(f"Unsupported configuration format for {path}")
    if not isinstance(data, Mapping):
        raise TypeError(
            "Measurement model configuration must decode to a mapping. "
            f"Received {type(data)!r} from {path}."
        )
    return data


def _parse_measurement_models(
    definitions: Iterable[Mapping[str, object]],
) -> tuple[MeasurementModel, ...]:
    models: list[MeasurementModel] = []
    for index, entry in enumerate(definitions):
        if not isinstance(entry, Mapping):
            raise TypeError(
                "Each measurement model definition must be a mapping. "
                f"Entry {index} has type {type(entry)!r}."
            )
        try:
            name = str(entry["name"])
            mean = float(entry["mean"])
            std = float(entry["std"])
            weights_raw = entry["weights"]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(
                f"Measurement model entry {index} is missing key {exc.args[0]!r}."
            ) from exc

        if std <= 0:
            raise ValueError(f"Standard deviation for measurement '{name}' must be positive.")

        if not isinstance(weights_raw, Iterable):
            raise TypeError(f"Weights for measurement '{name}' must be an iterable of numbers.")

        weights = tuple(float(value) for value in weights_raw)
        models.append(MeasurementModel(name=name, mean=mean, std=std, weights=weights))
    return tuple(models)


def load_backend_config(backend: str = DEFAULT_BACKEND) -> BackendMeasurementConfig:
    """Load the measurement model configuration for a backend."""

    normalized = backend.lower()
    if normalized in _CONFIG_CACHE:
        return _CONFIG_CACHE[normalized]

    path = _resolve_backend_path(normalized)
    payload = _load_backend_payload(path)

    backend_name = str(payload.get("backend", normalized))
    try:
        num_betas = int(payload["num_betas"])
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Configuration for backend '{backend}' is missing 'num_betas'.") from exc
    if num_betas <= 0:
        raise ValueError(
            f"Configuration for backend '{backend}' must define a positive 'num_betas'."
        )

    scale_measurement = str(payload.get("scale_measurement", "height"))
    model_definitions = payload.get("models", [])
    if not isinstance(model_definitions, Iterable):
        raise TypeError("'models' must be an iterable of measurement definitions.")

    models = _parse_measurement_models(model_definitions)
    if not models:
        raise ValueError(f"Backend '{backend}' must declare at least one measurement model.")

    config = BackendMeasurementConfig(
        backend=backend_name,
        num_betas=num_betas,
        scale_measurement=scale_measurement,
        models=models,
    )
    _CONFIG_CACHE[normalized] = config
    return config


def available_backends() -> tuple[str, ...]:
    """Return the set of backend identifiers with available configuration files."""

    if not MEASUREMENT_MODEL_DIR.exists():
        return (DEFAULT_BACKEND,)

    backends: set[str] = set()
    for path in MEASUREMENT_MODEL_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}:
            backends.add(path.stem.lower())

    if not backends:
        return (DEFAULT_BACKEND,)
    return tuple(sorted(backends))


DEFAULT_BACKEND_CONFIG = load_backend_config(DEFAULT_BACKEND)
MEASUREMENT_MODELS: tuple[MeasurementModel, ...] = DEFAULT_BACKEND_CONFIG.models
DEFAULT_NUM_BETAS = DEFAULT_BACKEND_CONFIG.num_betas
# Backwards compatibility with existing imports.
SMPLX_NUM_BETAS = DEFAULT_NUM_BETAS
MEASUREMENT_MODELS: tuple[MeasurementModel, ...] = (
    MeasurementModel("height", 170.0, 7.5, (0.45, 0.05, 0.02, 0.0, 0.03, 0.0, 0.01, 0.0, 0.0, 0.0)),
    MeasurementModel(
        "chest_circumference", 95.0, 8.0, (0.6, 0.2, 0.05, 0.03, 0.0, 0.02, 0.05, 0.01, 0.0, 0.0)
    ),
    MeasurementModel(
        "waist_circumference", 80.0, 7.0, (0.1, 0.55, 0.05, 0.05, 0.1, 0.02, 0.0, 0.04, 0.03, 0.0)
    ),
    MeasurementModel(
        "hip_circumference", 98.0, 8.5, (0.05, 0.05, 0.6, 0.08, 0.05, 0.02, 0.04, 0.02, 0.02, 0.0)
    ),
    MeasurementModel(
        "shoulder_width", 42.0, 2.5, (0.15, 0.05, 0.02, 0.4, 0.02, 0.1, 0.05, 0.1, 0.0, 0.0)
    ),
    MeasurementModel(
        "neck_circumference", 36.0, 2.5, (0.18, 0.02, 0.0, 0.05, 0.35, 0.05, 0.02, 0.0, 0.05, 0.0)
    ),
    MeasurementModel(
        "arm_length", 60.0, 4.5, (0.05, 0.05, 0.0, 0.12, 0.02, 0.3, 0.08, 0.12, 0.05, 0.0)
    ),
    MeasurementModel(
        "inseam_length", 78.0, 5.0, (0.05, 0.04, 0.08, 0.02, 0.0, 0.1, 0.4, 0.08, 0.08, 0.05)
    ),
    MeasurementModel(
        "thigh_circumference", 55.0, 5.0, (0.0, 0.05, 0.45, 0.02, 0.0, 0.1, 0.3, 0.05, 0.02, 0.01)
    ),
    MeasurementModel(
        "calf_circumference", 37.0, 3.5, (0.0, 0.02, 0.2, 0.01, 0.0, 0.05, 0.1, 0.4, 0.1, 0.02)
    ),
    MeasurementModel(
        "bicep_circumference", 30.0, 3.0, (0.1, 0.05, 0.02, 0.15, 0.02, 0.25, 0.05, 0.12, 0.1, 0.02)
    ),
    MeasurementModel(
        "wrist_circumference", 16.0, 1.0, (0.02, 0.01, 0.0, 0.06, 0.03, 0.15, 0.02, 0.2, 0.15, 0.1)
    ),
)


@dataclass(frozen=True)
class FitResult:
    """Container for the fitted body model parameters."""

    betas: np.ndarray
    scale: float
    translation: np.ndarray
    residual: float
    measurements_used: tuple[str, ...]
    measurement_report: MeasurementReport

    def to_dict(self) -> dict:
        return {
            "betas": self.betas.tolist(),
            "scale": float(self.scale),
            "translation": self.translation.tolist(),
            "residual": float(self.residual),
            "measurements_used": list(self.measurements_used),
            "measurement_report": {
                "coverage": float(self.measurement_report.coverage),
                "values": self.measurement_report.visualization_payload(),
            },
        }


def create_body_mesh(
    result: FitResult,
    *,
    model_path: Path | None = None,
    model_type: str = "smplx",
    gender: str = "neutral",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a body mesh from fitted parameters.

    Parameters
    ----------
    result:
        The fitted SMPL-X parameters to apply to the model.
    model_path:
        Optional override for the asset directory used by the provider.
        Defaults to ``assets/<model_type>`` relative to the project root.
    model_type:
        The registered provider name to instantiate (``smplx`` by default).
    gender:
        Gendered SMPL-X model variant to load.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the mesh vertices and triangular faces as numpy
        arrays.
    """

    assets = Path(model_path) if model_path is not None else Path("assets") / model_type

    from avatar_model import BodyModel

    betas = np.asarray(result.betas, dtype=float).reshape(1, -1)
    num_betas = int(betas.shape[1]) if betas.ndim > 1 else int(betas.size)
    if num_betas <= 0:
        num_betas = 1

    model = BodyModel(
        model_path=assets,
        model_type=model_type,
        gender=gender,
        batch_size=1,
        num_betas=num_betas,
    )
    model.set_shape(betas)

    vertices_tensor = model.vertices()
    vertices = vertices_tensor.detach().cpu().numpy()[0]
    vertices = vertices * float(result.scale)
    translation = np.asarray(result.translation, dtype=float).reshape(-1)
    if translation.size != 3:
        raise ValueError("FitResult.translation must contain three elements.")
    vertices = vertices + translation.reshape(1, 3)

    faces = np.asarray(getattr(model.model, "faces"), dtype=np.int32)
    return vertices.astype(np.float32), faces.astype(np.int32, copy=False)


def load_schema(path: Path | None = None) -> dict:
    if path is not None:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    return load_measurement_catalog(measurement_path=SCHEMA_PATH)


def required_measurements(schema: Mapping[str, Iterable[Mapping[str, object]]]) -> set[str]:
    manual = schema.get("manual_measurements", [])
    return {
        item["name"]
        for item in manual
        if isinstance(item, Mapping)
        and item.get("required") is True
        and isinstance(item.get("name"), str)
    }


def validate_measurements(
    measurements: Mapping[str, float], schema: Mapping[str, object] | None = None
) -> None:
    schema = schema or load_schema()
    missing = [name for name in required_measurements(schema) if name not in measurements]
    if missing:
        raise ValueError(f"Missing required measurements: {', '.join(sorted(missing))}")


def models_by_name(models: Sequence[MeasurementModel]) -> dict[str, MeasurementModel]:
    return {model.name: model for model in models}


def fit_shape_coefficients(
    measurements: Mapping[str, float],
    *,
    models: Sequence[MeasurementModel] | None = None,
    num_shape_coeffs: int | None = None,
) -> tuple[np.ndarray, tuple[str, ...], float]:
    models = tuple(models or MEASUREMENT_MODELS)
    num_shape_coeffs = int(num_shape_coeffs or DEFAULT_NUM_BETAS)

    available = {m.name: m for m in models if m.name in measurements}
    if not available:
        raise ValueError("At least one measurement is required to fit body shape coefficients.")

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
    backend: str = DEFAULT_BACKEND,
    schema_path: Path | None = None,
    models: Sequence[MeasurementModel] | None = None,
    num_shape_coeffs: int | None = None,
    inference_model: GaussianMeasurementModel | None = None,
) -> FitResult:
    """Compute SMPL-X shape parameters from manual measurements."""

    backend_config = load_backend_config(backend)
    models = tuple(models or backend_config.models)
    num_shape_coeffs = int(num_shape_coeffs or backend_config.num_betas)

    schema = load_schema(schema_path)
    model = inference_model or load_default_model()
    report = model.infer(measurements)
    completed_measurements = report.values()
    completed_measurements.update({name: float(value) for name, value in measurements.items()})

    validate_measurements(completed_measurements, schema)

    coeffs, used_names, rms = fit_shape_coefficients(
        completed_measurements,
        models=models,
        num_shape_coeffs=num_shape_coeffs,
    )

    scale = (
        float(completed_measurements.get("height", models_by_name(models)["height"].mean))
        / models_by_name(models)["height"].mean
    )
    translation = np.zeros(3, dtype=float)

    return FitResult(
        betas=coeffs,
        scale=scale,
        translation=translation,
        residual=rms,
        measurements_used=used_names,
        measurement_report=report,
    )


def save_fit(result: FitResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result.to_dict(), stream, indent=2)


def plot_measurement_report(result: FitResult, output_dir: Path) -> Path | None:
    """Visualise the measurement report if matplotlib is available."""

    if importlib_util.find_spec("matplotlib") is None:
        return None

    estimates = result.measurement_report.estimates
    if not estimates:
        return None

    from matplotlib import colors as mcolors
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch

    output_dir.mkdir(parents=True, exist_ok=True)
    values = [estimate.value for estimate in estimates]
    labels = [estimate.name for estimate in estimates]
    yerr = []
    for estimate in estimates:
        variance = getattr(estimate, "variance", 0.0) or 0.0
        if variance < 0:
            variance = 0.0
        yerr.append(float(np.sqrt(variance)))
    yerr_array = np.asarray(yerr, dtype=float)
    has_uncertainty = bool(np.any(yerr_array > 0))

    base_palette = {
        "measured": "#1f77b4",
        "inferred": "#ff7f0e",
    }
    color_lookup: dict[str, tuple[float, float, float, float]] = {}
    colors = []
    for estimate in estimates:
        base_color = base_palette.get(estimate.source, "#6c757d")
        rgba = list(mcolors.to_rgba(base_color))
        confidence = max(0.0, min(1.0, float(estimate.confidence)))
        rgba[3] = 0.35 + 0.65 * confidence
        colors.append(tuple(rgba))
        color_lookup.setdefault(estimate.source, tuple(rgba))

    fig_width = max(6.0, len(labels) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    indices = np.arange(len(labels))
    bars = ax.bar(
        indices,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        yerr=yerr_array,
        capsize=4,
    )

    ax.set_ylabel("Measurement value")
    if has_uncertainty:
        ax.set_title("Measurement report (bars show ±1σ)")
    else:
        ax.set_title("Measurement report")
    ax.set_xticks(indices, labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    legend_handles = [
        Patch(facecolor=color_lookup[source], label=source.title())
        for source in sorted(color_lookup)
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, title="Source", loc="best")

    for bar, estimate in zip(bars, estimates):
        ax.annotate(
            f"{estimate.value:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize="small",
        )

    fig.tight_layout()
    plot_path = output_dir / "measurement_report.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def _load_measurements_from_json(path: Path) -> Mapping[str, float]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("Measurement input must be a JSON object.")
    return {key: float(value) for key, value in payload.items()}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit body model parameters from manual measurements."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Path to a JSON file containing measurement values.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/meshes/manual_measurement_fit.json"),
        help="Where to store the fitted parameter dictionary.",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=available_backends(),
        help="Body model backend to use when fitting coefficients.",
    )
    parser.add_argument(
        "--coeff-count",
        type=int,
        default=None,
        help="Number of shape coefficients to estimate (defaults to backend configuration).",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        help="One or more Afflec preprocessed images containing measurement headers.",
    )
    args = parser.parse_args(argv)

    if args.images:
        measurements = extract_measurements_from_afflec_images(args.images)
    elif args.input is not None:
        measurements = _load_measurements_from_json(args.input)
    else:
        parser.error("Either a measurement JSON file or --images must be provided.")
    result = fit_smplx_from_measurements(
        measurements,
        backend=args.backend,
        num_shape_coeffs=args.coeff_count,
    )
    save_fit(result, args.output)
    print(f"Saved {args.backend} parameters to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
