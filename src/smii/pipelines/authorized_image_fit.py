"""Production image fitting with an explicit refinement authority boundary."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from pipelines.measurement_inference import MeasurementReport, load_default_model

from .fit_from_images import (
    SMPLXRegressionResult,
    finalize_regression_diagnostics,
    regress_smplx_from_images as _legacy_regress_smplx_from_images,
)
from .fit_from_measurements import (
    _load_backend_payload,
    _resolve_backend_path,
    available_backends,
    load_backend_config,
    load_schema,
    validate_measurements,
)
from .refinement_authority import RefinementReceipt, build_receipt, solve_bounded_refinement
from .refinement_policy import RefinementPolicy


@dataclass(frozen=True)
class GovernedFitResult:
    """FitResult-compatible selected state plus candidate authority evidence."""

    betas: np.ndarray
    scale: float
    translation: np.ndarray
    residual: float
    measurements_used: tuple[str, ...]
    measurement_report: MeasurementReport
    refinement_receipt: RefinementReceipt
    provenance: Mapping[str, Any] | None = None
    raw_measurements: Mapping[str, float] | None = None
    fit_mode: str | None = None
    trust_level: str | None = None
    consistency_status: str | None = None
    consistency_flags: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "betas": np.asarray(self.betas, dtype=float).reshape(-1).tolist(),
            "scale": float(self.scale),
            "translation": np.asarray(self.translation, dtype=float).reshape(-1).tolist(),
            "residual": float(self.residual),
            "measurements_used": list(self.measurements_used),
            "measurement_report": {
                "coverage": float(self.measurement_report.coverage),
                "values": self.measurement_report.visualization_payload(),
            },
            "refinement_receipt": self.refinement_receipt.to_dict(),
            "refinement_receipt_hash": self.refinement_receipt.receipt_hash,
        }
        if self.provenance is not None:
            payload["provenance"] = dict(self.provenance)
        if self.raw_measurements is not None:
            payload["raw_measurements"] = {
                name: float(value) for name, value in self.raw_measurements.items()
            }
        for key in ("fit_mode", "trust_level", "consistency_status"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.consistency_flags:
            payload["consistency_flags"] = list(self.consistency_flags)
        if self.diagnostics is not None:
            payload["diagnostics"] = dict(self.diagnostics)
        return payload


def _severity(status: str | None, flags: Sequence[str]) -> str:
    if str(status or "").upper() == "FAIL":
        return "fail"
    if str(status or "").upper() == "WARN" or flags:
        return "warn"
    return "pass"


def _effective_policy(backend: str) -> tuple[RefinementPolicy, tuple[object, ...]]:
    config = load_backend_config(backend)
    payload = _load_backend_payload(_resolve_backend_path(backend))
    settings = payload.get("refinement_policy", {})
    if not isinstance(settings, Mapping):
        raise TypeError("refinement_policy must be an object")
    policy = RefinementPolicy.from_effective_config(
        backend=config.backend,
        num_betas=config.num_betas,
        scale_measurement=config.scale_measurement,
        models=config.models,
        settings=settings,
    )
    return policy, tuple(config.models)


def _measurement_system(
    measurements: Mapping[str, float],
    models: Sequence[object],
    num_betas: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    available = {
        str(getattr(model, "name")): model
        for model in models
        if str(getattr(model, "name")) in measurements
    }
    if not available:
        raise ValueError("At least one configured measurement is required")
    names = tuple(sorted(available))
    matrix = np.vstack(
        [np.asarray(getattr(available[name], "weights"), dtype=float)[:num_betas] for name in names]
    )
    if matrix.shape[1] != num_betas:
        raise ValueError("Effective measurement matrix does not match policy beta count")
    target = np.asarray(
        [
            (float(measurements[name]) - float(getattr(available[name], "mean")))
            / float(getattr(available[name], "std"))
            for name in names
        ],
        dtype=float,
    )
    return matrix, target, names


def fit_anchored_measurement_candidate(
    measurements: Mapping[str, float],
    *,
    anchor_betas: Sequence[float] | np.ndarray,
    anchor_scale: float = 1.0,
    anchor_translation: Sequence[float] | np.ndarray | None = None,
    backend: str = "smplx",
    schema_path: Path | None = None,
    consistency_status: str | None = None,
    consistency_flags: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
    raw_measurements: Mapping[str, float] | None = None,
    fit_mode: str | None = None,
    trust_level: str | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> GovernedFitResult:
    """Generate a bounded candidate and select it only through its receipt decision."""

    policy, models = _effective_policy(backend)
    anchor = np.asarray(anchor_betas, dtype=float).reshape(-1)
    translation = (
        np.zeros(3, dtype=float)
        if anchor_translation is None
        else np.asarray(anchor_translation, dtype=float).reshape(-1)
    )
    if anchor.size != policy.num_betas:
        raise ValueError(f"anchor_betas must contain {policy.num_betas} values")
    if translation.size != 3 or not np.all(np.isfinite(translation)):
        raise ValueError("anchor_translation must contain three finite values")
    if not np.isfinite(anchor_scale) or anchor_scale <= 0:
        raise ValueError("anchor_scale must be finite and positive")

    report = load_default_model().infer(measurements)
    completed = report.values()
    completed.update({name: float(value) for name, value in measurements.items()})
    validate_measurements(completed, load_schema(schema_path))
    matrix, target, names = _measurement_system(completed, models, policy.num_betas)
    solution = solve_bounded_refinement(matrix, target, anchor, policy)

    model_by_name = {str(getattr(model, "name")): model for model in models}
    scale_model = model_by_name[policy.scale_measurement]
    scale_mean = float(getattr(scale_model, "mean"))
    if scale_mean == 0:
        raise ValueError("Configured scale measurement mean must be non-zero")
    candidate_scale = float(completed[policy.scale_measurement]) / scale_mean
    severity = _severity(consistency_status, consistency_flags)
    if not np.isfinite(candidate_scale) or candidate_scale <= 0:
        severity = "fail"

    receipt = build_receipt(
        policy=policy,
        measurements=completed,
        names=names,
        anchor_betas=anchor,
        solution=solution,
        warnings=consistency_flags,
        severity=severity,  # type: ignore[arg-type]
        input_context={"scale": float(anchor_scale), "translation": translation.tolist()},
        candidate_context={"scale": candidate_scale, "translation": translation.tolist()},
    )
    promoted = receipt.decision == "promote"
    return GovernedFitResult(
        betas=solution.betas if promoted else anchor.copy(),
        scale=candidate_scale if promoted else float(anchor_scale),
        translation=translation.copy(),
        residual=(
            solution.measurement_residual
            if promoted
            else solution.anchor_measurement_residual
        ),
        measurements_used=names,
        measurement_report=report,
        refinement_receipt=receipt,
        provenance=provenance,
        raw_measurements=raw_measurements,
        fit_mode=fit_mode,
        trust_level=trust_level,
        consistency_status=consistency_status,
        consistency_flags=tuple(str(item) for item in consistency_flags),
        diagnostics=diagnostics,
    )


def fit_smplx_from_images(
    image_paths: Iterable[Path],
    *,
    backend: str = "smplx",
    schema_path: Path | None = None,
    models: Sequence[object] | None = None,
    num_shape_coeffs: int | None = None,
    inference_model: object | None = None,
    detector: str = "mediapipe",
    fit_mode: str = "heuristic",
    model_path: Path | None = None,
    model_type: str = "smplx",
    gender: str = "neutral",
) -> GovernedFitResult:
    """Return the governed measurement-refinement selection for an image fit."""

    if models is not None or num_shape_coeffs is not None or inference_model is not None:
        raise ValueError(
            "The governed image-refinement path consumes the declared backend policy; "
            "custom model rows, coefficient counts, and inference models are not accepted."
        )
    paths = tuple(Path(path) for path in image_paths)
    regression = regress_smplx_from_images(
        paths,
        detector=detector,
        refine_with_measurements=True,
        fit_mode=fit_mode,
        model_path=model_path,
        model_type=model_type,
        gender=gender,
    )
    if regression.measurement_fit is None:  # pragma: no cover - defensive
        raise RuntimeError("Governed image fitting did not produce a refinement receipt")
    return regression.measurement_fit


def regress_smplx_from_images(
    image_paths: Iterable[Path],
    *,
    detector: str = "mediapipe",
    refine_with_measurements: bool = True,
    fit_mode: str = "heuristic",
    model_path: Path | None = None,
    model_type: str = "smplx",
    gender: str = "neutral",
) -> SMPLXRegressionResult:
    """Produce the raw image fit first, then govern measurement refinement."""

    raw = _legacy_regress_smplx_from_images(
        image_paths,
        detector=detector,
        refine_with_measurements=False,
        fit_mode=fit_mode,
        model_path=model_path,
        model_type=model_type,
        gender=gender,
    )
    if not refine_with_measurements or not raw.measurements:
        return raw

    backend = model_type if model_type in available_backends() else "smplx"
    refinement = fit_anchored_measurement_candidate(
        raw.measurements,
        anchor_betas=raw.betas,
        anchor_scale=1.0,
        anchor_translation=raw.transl,
        backend=backend,
        consistency_status=raw.consistency_status,
        consistency_flags=raw.consistency_flags,
        provenance={
            "images_used": [str(frame.image_path) for frame in raw.frames],
            "detector": raw.detector,
            "measurement_source": raw.measurement_source,
        },
        raw_measurements=raw.measurements,
        fit_mode=f"{raw.fit_mode}_plus_measurement_refinement",
        trust_level=raw.trust_level,
        diagnostics=raw.optimization_report,
    )
    governed = replace(
        raw,
        measurement_fit=refinement,
        fit_mode=f"{raw.fit_mode}_plus_measurement_refinement",
    )
    return finalize_regression_diagnostics(governed)


__all__ = [
    "GovernedFitResult",
    "fit_smplx_from_images",
    "fit_anchored_measurement_candidate",
    "regress_smplx_from_images",
]
