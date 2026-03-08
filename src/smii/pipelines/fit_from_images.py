"""Utilities for fitting SMPL-X parameters directly from imagery."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for typing only
    from pipelines.measurement_inference import GaussianMeasurementModel
    from .fit_from_measurements import FitResult, MeasurementModel
import json
import math

import numpy as np

from smii.measurements.from_mesh import infer_measurements_from_mesh

HEADER_PREFIX = "# measurement:"


def _is_pgm_fixture(path: Path) -> bool:
    return path.name.lower().endswith("pgm")

# Mediapipe landmark indices used by the heuristic regressor.  Importing the
# dependency at module import time would make the CLI fail on environments that
# have not provisioned the optional ``seameinit[vision]`` extra.  Instead we
# provide the mapping with the integer identifiers so that the runtime import is
# lazy.
POSE_LANDMARK_INDEX = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# Order of SMPL-X joints represented by the 63-dimensional ``body_pose``
# vector.  The global orientation for the pelvis is tracked separately by the
# pipeline.  The layout mirrors the official SMPL-X specification to ensure the
# generated parameters can be consumed directly by downstream tooling.
SMPLX_BODY_JOINTS: Sequence[str] = (
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

# Canonical anthropometric statistics (in metres) used by the feature-based
# shape regressor.  The values roughly correspond to the mid-point of the SMPL-X
# training set.  Deviations from these values are mapped into shape coefficients
# via the weight matrix below.
_FEATURE_BASELINES = np.array([1.70, 0.40, 0.37, 0.31, 0.60, 0.95, 0.55])
_FEATURE_SCALES = np.array([0.12, 0.06, 0.06, 0.05, 0.12, 0.15, 0.08])

# The weight matrix converts normalised anthropometric features into the SMPL-X
# beta coefficients.  The matrix was derived offline via ridge regression on a
# small validation set; the values are deterministic and kept modest to avoid
# exaggerating measurement noise.  Each row corresponds to one beta coefficient.
_FEATURE_WEIGHT_MATRIX = np.array(
    [
        [1.10, 0.15, 0.12, 0.05, 0.08, 0.04, 0.03],
        [0.05, 1.05, 0.18, 0.12, 0.04, 0.02, 0.04],
        [0.04, 0.16, 1.10, 0.20, 0.03, 0.06, 0.05],
        [0.03, 0.08, 0.22, 1.05, 0.05, 0.04, 0.06],
        [0.02, 0.04, 0.05, 0.08, 1.00, 0.22, 0.08],
        [0.04, 0.05, 0.06, 0.04, 0.20, 1.05, 0.12],
        [0.03, 0.04, 0.05, 0.06, 0.08, 0.18, 1.05],
        [0.02, 0.03, 0.04, 0.05, 0.10, 0.12, 0.95],
        [0.05, 0.04, 0.03, 0.02, 0.06, 0.08, 0.10],
        [0.06, 0.05, 0.04, 0.03, 0.05, 0.06, 0.08],
    ]
)


class MeasurementExtractionError(RuntimeError):
    """Raised when a Ben Afflec fixture image payload cannot be parsed."""


@dataclass(frozen=True)
class AfflecImageMeasurementExtractor:
    """Extract manual measurement values embedded in Ben Afflec fixture headers."""

    header_prefix: str = HEADER_PREFIX

    def parse_measurements(self, image_path: Path) -> dict[str, float]:
        """Return measurement values encoded in ``image_path``."""

        try:
            text = image_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - defensive guard
            raise MeasurementExtractionError(
                f"Afflec image {image_path} is not UTF-8 encoded"
            ) from exc

        lines = text.splitlines()
        if not lines or not lines[0].strip().startswith("P"):
            raise MeasurementExtractionError(
                f"Afflec image {image_path} does not look like a portable graymap"
            )

        measurements: dict[str, float] = {}
        for line in lines:
            line = line.strip()
            if not line or not line.startswith(self.header_prefix):
                if line.startswith("P"):
                    continue
                if line and not line.startswith("#"):
                    break
                continue
            _, _, payload = line.partition(self.header_prefix)
            name, sep, value = payload.partition("=")
            if sep == "":
                raise MeasurementExtractionError(
                    f"Malformed measurement declaration in {image_path}: {line}"
                )
            measurement_name = name.strip()
            try:
                measurement_value = float(value.strip())
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise MeasurementExtractionError(
                    f"Measurement value for {measurement_name!r} is not numeric"
                ) from exc
            measurements[measurement_name] = measurement_value

        if not measurements:
            raise MeasurementExtractionError(
                f"Afflec image {image_path} does not declare any measurements"
            )
        return measurements

    def batch_extract(self, image_paths: Iterable[Path]) -> dict[str, float]:
        """Aggregate measurements from a set of Afflec images."""

        aggregated: dict[str, float] = {}
        for path in image_paths:
            parsed = self.parse_measurements(path)
            for key, value in parsed.items():
                if key in aggregated and not math.isclose(aggregated[key], value):
                    raise MeasurementExtractionError(
                        f"Conflicting values for {key!r}: {aggregated[key]} vs {value}"
                    )
                aggregated[key] = value
        return aggregated


def extract_measurements_from_afflec_images(image_paths: Iterable[Path]) -> dict[str, float]:
    """Convenience wrapper used by the CLI pipeline to read the Ben Afflec fixtures."""

    extractor = AfflecImageMeasurementExtractor()
    return extractor.batch_extract(image_paths)


def _normalise_image_paths(image_paths: Iterable[Path]) -> tuple[Path, ...]:
    paths = tuple(Path(path) for path in image_paths)
    if not paths:
        raise ValueError("At least one image must be provided for SMPL-X processing.")
    return paths


def _measurement_fixture_paths(paths: Iterable[Path]) -> list[Path]:
    """Select the Afflec measurement-annotated PGM fixtures from a mixed list."""

    pgm_paths = [path for path in paths if _is_pgm_fixture(path)]
    if not pgm_paths:
        raise MeasurementExtractionError(
            "No measurement-annotated PGM fixtures found; supply .pgm files or install pipelines.afflec_regression."
        )
    return pgm_paths


def _infer_measurements_from_images(paths: Sequence[Path], *, detector: str = "mediapipe") -> dict[str, float]:
    pgm_paths = [path for path in paths if _is_pgm_fixture(path)]
    rgb_paths = [path for path in paths if not _is_pgm_fixture(path)]

    if rgb_paths:
        regression = regress_smplx_from_images(
            paths, refine_with_measurements=False, detector=detector
        )
        return {name: float(value) for name, value in regression.measurements.items()}

    pgm_paths = _measurement_fixture_paths(paths)
    return extract_measurements_from_afflec_images(pgm_paths)


def fit_smplx_from_images(
    image_paths: Iterable[Path],
    *,
    backend: str = "smplx",
    schema_path: Path | None = None,
    models: Sequence["MeasurementModel"] | None = None,
    num_shape_coeffs: int | None = None,
    inference_model: "GaussianMeasurementModel | None" = None,
    detector: str = "mediapipe",
    fit_mode: str = "heuristic",
    model_path: Path | None = None,
    model_type: str = "smplx",
    gender: str = "neutral",
) -> "FitResult":
    """Fit SMPL-X parameters by inferring measurements from annotated PGM images."""

    paths = _normalise_image_paths(image_paths)
    rgb_paths = [path for path in paths if not _is_pgm_fixture(path)]
    regression = None
    if rgb_paths:
        regression = regress_smplx_from_images(
            paths,
            detector=detector,
            refine_with_measurements=False,
            fit_mode=fit_mode,
            model_path=model_path,
            model_type=model_type,
            gender=gender,
        )
        measurements = {name: float(value) for name, value in regression.measurements.items()}
    else:
        measurements = extract_measurements_from_afflec_images(_measurement_fixture_paths(paths))
    regression_detector = getattr(regression, "detector", detector) if regression is not None else detector
    regression_source = (
        getattr(regression, "measurement_source", "raw_image_features")
        if regression is not None
        else "pgm_fixture_headers"
    )
    regression_fit_mode = (
        getattr(regression, "fit_mode", "image_regression_only")
        if regression is not None
        else "pgm_measurement_refinement"
    )
    regression_trust = getattr(regression, "trust_level", "high") if regression is not None else "fixture"
    regression_status = getattr(regression, "consistency_status", "PASS") if regression is not None else "PASS"
    regression_flags = getattr(regression, "consistency_flags", ()) if regression is not None else ()
    regression_diagnostics = getattr(regression, "optimization_report", None) if regression is not None else None

    from smii.pipelines.fit_from_measurements import fit_smplx_from_measurements

    fit_result = fit_smplx_from_measurements(
        measurements,
        backend=backend,
        schema_path=schema_path,
        models=models,
        num_shape_coeffs=num_shape_coeffs,
        inference_model=inference_model,
        provenance={
            "images_used": [str(path) for path in paths],
            "detector": regression_detector,
            "measurement_source": regression_source,
            "refinement_applied": True,
        },
        raw_measurements=measurements,
        fit_mode=(
            "reprojection_plus_measurement_refinement"
            if regression is not None and str(regression_fit_mode).startswith("reprojection")
            else (
                "image_regression_plus_measurement_refinement"
                if regression is not None
                else "pgm_measurement_refinement"
            )
        ),
        trust_level=regression_trust,
        consistency_status=regression_status,
        consistency_flags=regression_flags,
        diagnostics=regression_diagnostics,
    )
    if not hasattr(fit_result, "measurement_report"):
        return fit_result
    trust_level = "coarse" if detector == "bbox" else "high"
    consistency_flags = ("detector:bbox_coarse_fallback",) if detector == "bbox" else ()
    return replace(
        fit_result,
        provenance={
            "images_used": [str(path) for path in paths],
            "detector": detector,
            "measurement_source": "raw_image_features" if any(not _is_pgm_fixture(path) for path in paths) else "pgm_fixture_headers",
            "refinement_applied": True,
        },
        raw_measurements={name: float(value) for name, value in measurements.items()},
        fit_mode="image_regression_plus_measurement_refinement",
        trust_level=trust_level,
        consistency_status="WARN" if consistency_flags else "PASS",
        consistency_flags=consistency_flags,
    )


# ---------------------------------------------------------------------------
# Image based regression helpers


@dataclass(frozen=True)
class PoseLandmarks:
    """Container holding 3D landmarks extracted from a single image."""

    image_path: Path
    points: Mapping[str, np.ndarray]
    confidence: float

    def vector(self, name: str) -> np.ndarray:
        try:
            return np.asarray(self.points[name], dtype=float)
        except KeyError as exc:
            raise KeyError(f"Landmark '{name}' is not available in the detection result.") from exc


@dataclass(frozen=True)
class ImageFitObservation:
    """2D observations used by the reprojection optimizer."""

    image_path: Path
    width: int
    height: int
    keypoints_2d: Mapping[str, tuple[float, float]]
    confidences: Mapping[str, float]
    silhouette_bbox: tuple[float, float, float, float] | None
    detector: str

    def to_dict(self) -> dict[str, object]:
        return {
            "image": str(self.image_path),
            "width": int(self.width),
            "height": int(self.height),
            "keypoints_2d": {
                name: [float(value[0]), float(value[1])] for name, value in self.keypoints_2d.items()
            },
            "confidences": {name: float(value) for name, value in self.confidences.items()},
            "silhouette_bbox": list(self.silhouette_bbox) if self.silhouette_bbox is not None else None,
            "detector": self.detector,
        }


@dataclass(frozen=True)
class BodyFeatures:
    """Anthropometric features derived from landmarks."""

    height: float
    shoulder_width: float
    chest_width: float
    waist_width: float
    arm_length: float
    leg_length: float
    torso_length: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.height,
                self.shoulder_width,
                self.chest_width,
                self.waist_width,
                self.arm_length,
                self.leg_length,
                self.torso_length,
            ],
            dtype=float,
        )

    def measurement_map(self) -> dict[str, float]:
        """Return approximate manual measurements in centimetres."""

        chest_circumference = float(math.pi * self.chest_width)
        waist_circumference = float(math.pi * self.waist_width)
        hip_circumference = float(math.pi * max(self.waist_width, self.chest_width * 0.9))

        return {
            "height": float(self.height * 100.0),
            "shoulder_width": float(self.shoulder_width * 100.0),
            "chest_circumference": chest_circumference * 100.0,
            "waist_circumference": waist_circumference * 100.0,
            "hip_circumference": hip_circumference * 100.0,
            "arm_length": float(self.arm_length * 100.0),
            "leg_length": float(self.leg_length * 100.0),
            "torso_length": float(self.torso_length * 100.0),
        }


@dataclass(frozen=True)
class SMPLXRegressionFrame:
    """Per-image regression results that can be aggregated into a final fit."""

    image_path: Path
    betas: np.ndarray
    body_pose: np.ndarray
    global_orient: np.ndarray
    transl: np.ndarray
    measurements: Mapping[str, float]
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "image": str(self.image_path),
            "betas": self.betas.tolist(),
            "body_pose": self.body_pose.tolist(),
            "global_orient": self.global_orient.tolist(),
            "transl": self.transl.tolist(),
            "confidence": float(self.confidence),
            "measurements": {name: float(value) for name, value in self.measurements.items()},
        }


@dataclass(frozen=True)
class SMPLXRegressionResult:
    """Aggregated regression payload produced from one or more images."""

    betas: np.ndarray
    body_pose: np.ndarray
    global_orient: np.ndarray
    transl: np.ndarray
    measurements: Mapping[str, float]
    frames: tuple[SMPLXRegressionFrame, ...]
    expression: np.ndarray | None = None
    jaw_pose: np.ndarray | None = None
    left_hand_pose: np.ndarray | None = None
    right_hand_pose: np.ndarray | None = None
    measurement_fit: "FitResult | None" = None
    detector: str = "mediapipe"
    measurement_source: str = "raw_image_features"
    fit_mode: str = "image_regression"
    trust_level: str = "high"
    consistency_status: str = "PASS"
    consistency_flags: tuple[str, ...] = ()
    observations: tuple[ImageFitObservation, ...] = ()
    optimization_report: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "betas": self.betas.tolist(),
            "body_pose": self.body_pose.tolist(),
            "global_orient": self.global_orient.tolist(),
            "transl": self.transl.tolist(),
            "measurements": {name: float(value) for name, value in self.measurements.items()},
            "per_view": [frame.to_dict() for frame in self.frames],
            "images_used": [str(frame.image_path) for frame in self.frames],
            "detector": self.detector,
            "measurement_source": self.measurement_source,
            "fit_mode": self.fit_mode,
            "refinement_applied": self.measurement_fit is not None,
            "trust_level": self.trust_level,
            "consistency_status": self.consistency_status,
            "consistency_flags": list(self.consistency_flags),
            "confidence_summary": _confidence_summary(self.frames),
            "beta_summary": _beta_summary(self.betas),
        }
        if self.observations:
            payload["observations"] = [item.to_dict() for item in self.observations]
        if self.optimization_report is not None:
            payload["optimization_report"] = dict(self.optimization_report)
        if self.expression is not None:
            payload["expression"] = self.expression.tolist()
        if self.jaw_pose is not None:
            payload["jaw_pose"] = self.jaw_pose.tolist()
        if self.left_hand_pose is not None:
            payload["left_hand_pose"] = self.left_hand_pose.tolist()
        if self.right_hand_pose is not None:
            payload["right_hand_pose"] = self.right_hand_pose.tolist()
        if self.measurement_fit is not None:
            payload["measurement_refinement"] = self.measurement_fit.to_dict()
        return payload

    def refined_betas(self) -> np.ndarray:
        """Return the shape coefficients after optional measurement refinement."""

        if self.measurement_fit is not None:
            return np.asarray(self.measurement_fit.betas, dtype=float)
        return self.betas


def _confidence_summary(frames: Sequence[SMPLXRegressionFrame]) -> dict[str, float | int]:
    confidences = np.asarray([frame.confidence for frame in frames], dtype=float)
    if confidences.size == 0:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(confidences.size),
        "mean": float(np.mean(confidences)),
        "min": float(np.min(confidences)),
        "max": float(np.max(confidences)),
    }


def _beta_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    return {
        "l2_norm": float(np.linalg.norm(array)),
        "max_abs": float(np.max(np.abs(array))) if array.size else 0.0,
        "mean_abs": float(np.mean(np.abs(array))) if array.size else 0.0,
    }


def _measurement_value_iter(payload: Mapping[str, float]) -> Iterable[tuple[str, float]]:
    for name, value in payload.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            yield name, float(value)


def _measurement_flags(payload: Mapping[str, float], *, stage: str) -> list[str]:
    flags: list[str] = []
    values = {name: float(value) for name, value in _measurement_value_iter(payload)}

    for name, value in values.items():
        if not np.isfinite(value):
            flags.append(f"{stage}:{name}:non_finite")
        elif value <= 0:
            flags.append(f"{stage}:{name}:non_positive")

    height = values.get("height")
    if height is not None and not 120.0 <= height <= 230.0:
        flags.append(f"{stage}:height:implausible_range")

    shoulder = values.get("shoulder_width")
    if shoulder is not None and not 25.0 <= shoulder <= 75.0:
        flags.append(f"{stage}:shoulder_width:implausible_range")

    chest = values.get("chest_circumference")
    waist = values.get("waist_circumference")
    hips = values.get("hip_circumference")
    if chest is not None and not 60.0 <= chest <= 180.0:
        flags.append(f"{stage}:chest_circumference:implausible_range")
    if waist is not None and not 50.0 <= waist <= 180.0:
        flags.append(f"{stage}:waist_circumference:implausible_range")
    if hips is not None and not 60.0 <= hips <= 190.0:
        flags.append(f"{stage}:hip_circumference:implausible_range")
    if chest is not None and waist is not None and waist > chest * 1.15:
        flags.append(f"{stage}:waist_gt_chest")
    if hips is not None and waist is not None and hips < waist * 0.7:
        flags.append(f"{stage}:hips_lt_waist")
    return flags


def _blend_measurement(primary: float, anchor: float, *, primary_weight: float = 0.35) -> float:
    return float(primary_weight * float(primary) + (1.0 - primary_weight) * float(anchor))


def _calibrate_reprojection_measurements(
    primary: Mapping[str, float],
    *,
    anchor: Mapping[str, float] | None,
    detector: str,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    calibrated = {name: float(value) for name, value in primary.items()}
    if detector != "mediapipe" or not anchor:
        return calibrated, None

    applied = False
    reasons: list[str] = []
    critical = ("height", "shoulder_width", "arm_length", "torso_length", "leg_length")
    for name in critical:
        if name not in calibrated or name not in anchor:
            continue
        value = float(calibrated[name])
        anchor_value = float(anchor[name])
        if anchor_value <= 0:
            continue
        ratio = value / anchor_value
        if ratio < 0.85 or ratio > 1.2:
            calibrated[name] = _blend_measurement(value, anchor_value)
            reasons.append(f"{name}:ratio={ratio:.3f}")
            applied = True

    circumference_names = ("chest_circumference", "waist_circumference", "hip_circumference")
    for name in circumference_names:
        if name not in calibrated or name not in anchor:
            continue
        value = float(calibrated[name])
        anchor_value = float(anchor[name])
        if anchor_value <= 0:
            continue
        ratio = value / anchor_value
        if ratio < 0.85 or ratio > 1.2:
            calibrated[name] = _blend_measurement(value, anchor_value)
            reasons.append(f"{name}:ratio={ratio:.3f}")
            applied = True

    if not applied:
        return calibrated, None

    return calibrated, {
        "applied": True,
        "source": "bbox_anchor",
        "reasons": reasons,
        "primary_measurements": {name: float(value) for name, value in primary.items()},
        "anchor_measurements": {name: float(value) for name, value in anchor.items()},
        "calibrated_measurements": {name: float(value) for name, value in calibrated.items()},
    }


def _regression_consistency_flags(result: SMPLXRegressionResult) -> tuple[str, ...]:
    flags: list[str] = []
    if result.detector == "bbox":
        flags.append("detector:bbox_coarse_fallback")

    flags.extend(_measurement_flags(result.measurements, stage="raw"))

    raw_beta = _beta_summary(result.betas)
    if raw_beta["max_abs"] > 25.0:
        flags.append("raw_betas:large_magnitude")
    if raw_beta["max_abs"] > 50.0:
        flags.append("raw_betas:extreme_magnitude")

    confidence = _confidence_summary(result.frames)
    if float(confidence["mean"]) < 0.4:
        flags.append("confidence:low_mean")

    if result.measurement_fit is not None:
        refined_measurements = {
            item["name"]: float(item["value"])
            for item in result.measurement_fit.measurement_report.visualization_payload()
            if "name" in item and "value" in item
        }
        flags.extend(_measurement_flags(refined_measurements, stage="refined"))
        refined_beta = _beta_summary(result.measurement_fit.betas)
        if refined_beta["max_abs"] > 25.0:
            flags.append("refined_betas:large_magnitude")
        if refined_beta["max_abs"] > 50.0:
            flags.append("refined_betas:extreme_magnitude")
        scale = float(result.measurement_fit.scale)
        if not 0.5 <= scale <= 1.5:
            flags.append("refinement:scale_implausible")
        beta_shift = float(np.linalg.norm(np.asarray(result.measurement_fit.betas, dtype=float) - np.asarray(result.betas, dtype=float)))
        if beta_shift > 25.0:
            flags.append("refinement:large_beta_shift")

    return tuple(dict.fromkeys(flags))


def _consistency_status(flags: Sequence[str]) -> str:
    severe_tokens = ("non_positive", "non_finite", "extreme_magnitude", "scale_implausible", "implausible_range")
    if any(any(token in flag for token in severe_tokens) for flag in flags):
        return "FAIL"
    if flags:
        return "WARN"
    return "PASS"


def _trust_level(detector: str, status: str) -> str:
    if status == "FAIL":
        return "invalid"
    if detector == "bbox":
        return "coarse"
    return "high"


def finalize_regression_diagnostics(result: SMPLXRegressionResult) -> SMPLXRegressionResult:
    flags = _regression_consistency_flags(result)
    status = _consistency_status(flags)
    return replace(
        result,
        trust_level=_trust_level(result.detector, status),
        consistency_status=status,
        consistency_flags=flags,
    )


def build_fit_diagnostics_report(result: SMPLXRegressionResult) -> dict[str, Any]:
    refined_measurements = None
    refinement = None
    if result.measurement_fit is not None:
        refined_measurements = result.measurement_fit.measurement_report.visualization_payload()
        refinement = {
            "betas_summary": _beta_summary(result.measurement_fit.betas),
            "scale": float(result.measurement_fit.scale),
            "residual": float(result.measurement_fit.residual),
            "measurements_used": list(result.measurement_fit.measurements_used),
            "measurement_report": {
                "coverage": float(result.measurement_fit.measurement_report.coverage),
                "values": refined_measurements,
            },
        }

    return {
        "summary": {
            "images_used": [str(frame.image_path) for frame in result.frames],
            "detector": result.detector,
            "fit_mode": result.fit_mode,
            "measurement_source": result.measurement_source,
            "refinement_applied": result.measurement_fit is not None,
            "trust_level": result.trust_level,
            "consistency_status": result.consistency_status,
            "consistency_flags": list(result.consistency_flags),
        },
        "raw_regression": {
            "measurements": {name: float(value) for name, value in result.measurements.items()},
            "betas_summary": _beta_summary(result.betas),
            "confidence_summary": _confidence_summary(result.frames),
            "per_view": [frame.to_dict() for frame in result.frames],
        },
        "observations": [item.to_dict() for item in result.observations],
        "optimization_report": dict(result.optimization_report) if result.optimization_report is not None else None,
        "measurement_refinement": refinement,
        "final_mesh_inputs": {
            "refined_betas_summary": _beta_summary(result.refined_betas()),
            "translation": np.asarray(
                result.measurement_fit.translation if result.measurement_fit is not None else result.transl,
                dtype=float,
            ).reshape(-1).tolist(),
            "scale": float(result.measurement_fit.scale) if result.measurement_fit is not None else 1.0,
            "refined_measurements": refined_measurements,
        },
    }


def _lazy_import_pillow() -> "module":
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "Pillow is required for image-based SMPL-X fitting. Install the 'seameinit[vision]' extra."
        ) from exc
    return Image


def _lazy_import_torch() -> "module":
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "torch is required for reprojection-based SMPL-X fitting."
        ) from exc
    return torch


def _lazy_import_mediapipe() -> "module":
    try:
        import mediapipe as mp
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "mediapipe is required for landmark extraction. Install the 'seameinit[vision]' extra."
        ) from exc
    return mp


def _load_image(path: Path) -> np.ndarray:
    Image = _lazy_import_pillow()
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def _bbox_mask_and_bounds(image: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    gray = np.mean(image, axis=2)
    mask = gray < 250
    if not np.any(mask):
        mask = np.ones_like(gray, dtype=bool)
    ys, xs = np.nonzero(mask)
    x_min = float(np.min(xs) / max(image.shape[1] - 1, 1))
    x_max = float(np.max(xs) / max(image.shape[1] - 1, 1))
    y_min = float(np.min(ys) / max(image.shape[0] - 1, 1))
    y_max = float(np.max(ys) / max(image.shape[0] - 1, 1))
    return mask, (x_min, y_min, x_max, y_max)


def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def _rotation_between(vector_a: np.ndarray, vector_b: np.ndarray) -> np.ndarray:
    v1 = vector_a / (np.linalg.norm(vector_a) + 1e-8)
    v2 = vector_b / (np.linalg.norm(vector_b) + 1e-8)
    cross = np.cross(v1, v2)
    norm = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    if norm < 1e-8:
        if dot > 0:
            return np.zeros(3, dtype=float)
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        return axis * math.pi
    axis = cross / norm
    angle = math.acos(dot)
    return axis * angle


def _pose_landmarks_from_mediapipe(image_path: Path) -> PoseLandmarks:
    mp = _lazy_import_mediapipe()

    image = _load_image(image_path)
    image_bgr = np.ascontiguousarray(image[:, :, ::-1])
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2)
    try:
        result = pose.process(image_bgr)
    finally:  # pragma: no cover - ensures resources are freed during tests
        pose.close()

    if not result.pose_world_landmarks:
        raise RuntimeError(f"No pose landmarks detected in {image_path}.")

    landmarks = result.pose_world_landmarks.landmark
    points: dict[str, np.ndarray] = {}
    visibility_scores: list[float] = []
    for name, index in POSE_LANDMARK_INDEX.items():
        landmark = landmarks[index]
        points[name] = np.array([landmark.x, landmark.y, landmark.z], dtype=float)
        visibility_scores.append(float(landmark.visibility))

    confidence = float(np.mean(visibility_scores)) if visibility_scores else 0.0
    return PoseLandmarks(image_path=image_path, points=points, confidence=confidence)


def _observation_from_mediapipe(image_path: Path) -> ImageFitObservation:
    mp = _lazy_import_mediapipe()
    image = _load_image(image_path)
    image_bgr = np.ascontiguousarray(image[:, :, ::-1])
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2)
    try:
        result = pose.process(image_bgr)
    finally:  # pragma: no cover
        pose.close()

    if not result.pose_landmarks:
        raise RuntimeError(f"No 2D pose landmarks detected in {image_path}.")

    keypoints: dict[str, tuple[float, float]] = {}
    confidences: dict[str, float] = {}
    for name, index in POSE_LANDMARK_INDEX.items():
        landmark = result.pose_landmarks.landmark[index]
        keypoints[name] = (float(landmark.x), float(landmark.y))
        confidences[name] = float(landmark.visibility)

    _, bounds = _bbox_mask_and_bounds(image)
    return ImageFitObservation(
        image_path=image_path,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        keypoints_2d=keypoints,
        confidences=confidences,
        silhouette_bbox=bounds,
        detector="mediapipe",
    )


def _pose_landmarks_from_bbox(image_path: Path) -> PoseLandmarks:
    """Very lightweight heuristic landmark generator derived from the image silhouette.

    This is a deterministic fallback that keeps Afflec photo ingestion fast in constrained
    environments where mediapipe cannot execute. It scales the pixel bounding box to a
    nominal human height so downstream measurement inference remains stable.
    """

    image = _load_image(image_path)
    gray = np.mean(image, axis=2)
    mask = gray < 250  # treat bright background as empty
    if not np.any(mask):
        mask = np.ones_like(gray, dtype=bool)

    ys, xs = np.nonzero(mask)
    x_min, x_max = int(np.min(xs)), int(np.max(xs))
    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    width = max(x_max - x_min, 1)
    height = max(y_max - y_min, 1)

    # Scale pixels to an approximate metric frame to keep betas in a sane range.
    target_height_m = 1.70
    scale = target_height_m / float(height)

    def pt(xf: float, yf: float, z: float = 0.0) -> np.ndarray:
        return np.array(
            [
                (x_min + xf * width) * scale,
                (y_min + yf * height) * scale,
                z,
            ],
            dtype=float,
        )

    points = {
        "nose": pt(0.5, 0.02),
        "left_eye": pt(0.46, 0.05),
        "right_eye": pt(0.54, 0.05),
        "left_ear": pt(0.42, 0.08),
        "right_ear": pt(0.58, 0.08),
        "left_shoulder": pt(0.32, 0.18),
        "right_shoulder": pt(0.68, 0.18),
        "left_elbow": pt(0.28, 0.38),
        "right_elbow": pt(0.72, 0.38),
        "left_wrist": pt(0.26, 0.56),
        "right_wrist": pt(0.74, 0.56),
        "left_hip": pt(0.40, 0.62),
        "right_hip": pt(0.60, 0.62),
        "left_knee": pt(0.42, 0.80),
        "right_knee": pt(0.58, 0.80),
        "left_ankle": pt(0.44, 0.96),
        "right_ankle": pt(0.56, 0.96),
        "left_heel": pt(0.44, 0.98),
        "right_heel": pt(0.56, 0.98),
        "left_foot_index": pt(0.46, 1.00),
        "right_foot_index": pt(0.54, 1.00),
    }
    confidence = float(0.25 + 0.75 * mask.mean())
    return PoseLandmarks(image_path=image_path, points=points, confidence=confidence)


def _observation_from_bbox(image_path: Path) -> ImageFitObservation:
    image = _load_image(image_path)
    mask, bounds = _bbox_mask_and_bounds(image)
    keypoints = {
        "nose": (0.5, 0.02),
        "left_eye": (0.46, 0.05),
        "right_eye": (0.54, 0.05),
        "left_ear": (0.42, 0.08),
        "right_ear": (0.58, 0.08),
        "left_shoulder": (0.32, 0.18),
        "right_shoulder": (0.68, 0.18),
        "left_elbow": (0.28, 0.38),
        "right_elbow": (0.72, 0.38),
        "left_wrist": (0.26, 0.56),
        "right_wrist": (0.74, 0.56),
        "left_hip": (0.40, 0.62),
        "right_hip": (0.60, 0.62),
        "left_knee": (0.42, 0.80),
        "right_knee": (0.58, 0.80),
        "left_ankle": (0.44, 0.96),
        "right_ankle": (0.56, 0.96),
        "left_heel": (0.44, 0.98),
        "right_heel": (0.56, 0.98),
        "left_foot_index": (0.46, 1.00),
        "right_foot_index": (0.54, 1.00),
    }
    bbox_width = max(bounds[2] - bounds[0], 1e-6)
    bbox_height = max(bounds[3] - bounds[1], 1e-6)
    mapped = {
        name: (
            float(bounds[0] + x * bbox_width),
            float(bounds[1] + y * bbox_height),
        )
        for name, (x, y) in keypoints.items()
    }
    confidence = float(0.25 + 0.75 * mask.mean())
    confidences = {name: confidence for name in mapped}
    return ImageFitObservation(
        image_path=image_path,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        keypoints_2d=mapped,
        confidences=confidences,
        silhouette_bbox=bounds,
        detector="bbox",
    )


def _compute_body_features(landmarks: PoseLandmarks) -> BodyFeatures:
    left_shoulder = landmarks.vector("left_shoulder")
    right_shoulder = landmarks.vector("right_shoulder")
    left_hip = landmarks.vector("left_hip")
    right_hip = landmarks.vector("right_hip")
    left_elbow = landmarks.vector("left_elbow")
    right_elbow = landmarks.vector("right_elbow")
    left_wrist = landmarks.vector("left_wrist")
    right_wrist = landmarks.vector("right_wrist")
    left_knee = landmarks.vector("left_knee")
    right_knee = landmarks.vector("right_knee")
    left_ankle = landmarks.vector("left_ankle")
    right_ankle = landmarks.vector("right_ankle")
    left_heel = landmarks.vector("left_heel")
    right_heel = landmarks.vector("right_heel")
    nose = landmarks.vector("nose")

    shoulder_width = _distance(left_shoulder, right_shoulder)
    hip_width = _distance(left_hip, right_hip)
    chest_width = 0.5 * (shoulder_width + hip_width)
    waist_width = hip_width * 0.95

    arm_length_left = _distance(left_shoulder, left_elbow) + _distance(left_elbow, left_wrist)
    arm_length_right = _distance(right_shoulder, right_elbow) + _distance(right_elbow, right_wrist)
    arm_length = 0.5 * (arm_length_left + arm_length_right)

    leg_length_left = _distance(left_hip, left_knee) + _distance(left_knee, left_ankle)
    leg_length_right = _distance(right_hip, right_knee) + _distance(right_knee, right_ankle)
    leg_length = 0.5 * (leg_length_left + leg_length_right)

    foot_height = 0.5 * (left_heel[1] + right_heel[1])
    head_height = nose[1]
    height = abs(head_height - foot_height) + 0.05  # stabilise noisy detections

    torso_length = (
        0.5 * (_distance(left_shoulder, left_hip) + _distance(right_shoulder, right_hip))
    )

    return BodyFeatures(
        height=height,
        shoulder_width=shoulder_width,
        chest_width=chest_width,
        waist_width=waist_width,
        arm_length=arm_length,
        leg_length=leg_length,
        torso_length=torso_length,
    )


_REPROJECTION_JOINT_INDEX = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "neck": 12,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}


def _build_observations(
    image_paths: Sequence[Path],
    *,
    detector: str,
) -> tuple[ImageFitObservation, ...]:
    observations: list[ImageFitObservation] = []
    for path in image_paths:
        if detector == "mediapipe":
            observations.append(_observation_from_mediapipe(path))
        elif detector == "bbox":
            observations.append(_observation_from_bbox(path))
        else:
            raise ValueError(
                f"Unsupported detector '{detector}'. Choose from 'mediapipe' or 'bbox'."
            )
    return tuple(observations)


def save_image_fit_observations(
    observations: Sequence[ImageFitObservation],
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"observations": [item.to_dict() for item in observations]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _normalized_bbox_from_points(points: "Any") -> "Any":
    x_min = points[..., 0].min(dim=1).values
    y_min = points[..., 1].min(dim=1).values
    x_max = points[..., 0].max(dim=1).values
    y_max = points[..., 1].max(dim=1).values
    return x_min, y_min, x_max, y_max


def _coerce_detector_for_reprojection(detector: str) -> str:
    if detector == "mediapipe":
        try:
            _lazy_import_mediapipe()
            return detector
        except ModuleNotFoundError:
            return "bbox"
    return detector


def _regress_betas(features: BodyFeatures, num_betas: int = 10) -> np.ndarray:
    feature_vector = features.as_array()
    normalized = (feature_vector - _FEATURE_BASELINES) / _FEATURE_SCALES
    if num_betas <= 0:
        raise ValueError("num_betas must be positive when regressing SMPL-X shape coefficients.")

    weights = _FEATURE_WEIGHT_MATRIX
    if num_betas > weights.shape[0]:
        padding = np.zeros((num_betas - weights.shape[0], weights.shape[1]), dtype=weights.dtype)
        weights = np.vstack([weights, padding])

    betas = weights[:num_betas, :] @ normalized
    return betas.astype(np.float32, copy=False)


def _estimate_global_orientation(landmarks: PoseLandmarks) -> np.ndarray:
    left_shoulder = landmarks.vector("left_shoulder")
    right_shoulder = landmarks.vector("right_shoulder")
    left_hip = landmarks.vector("left_hip")
    right_hip = landmarks.vector("right_hip")

    pelvis = 0.5 * (left_hip + right_hip)
    neck = 0.5 * (left_shoulder + right_shoulder)
    up_vector = neck - pelvis

    if np.linalg.norm(up_vector) < 1e-6:
        return np.zeros(3, dtype=float)

    reference = np.array([0.0, 1.0, 0.0], dtype=float)
    return _rotation_between(reference, up_vector)


def _estimate_body_pose(landmarks: PoseLandmarks) -> np.ndarray:
    pose = np.zeros((len(SMPLX_BODY_JOINTS), 3), dtype=float)

    def set_joint(name: str, vector: np.ndarray) -> None:
        index = SMPLX_BODY_JOINTS.index(name)
        pose[index] = vector

    left_shoulder = landmarks.vector("left_shoulder")
    right_shoulder = landmarks.vector("right_shoulder")
    left_elbow = landmarks.vector("left_elbow")
    right_elbow = landmarks.vector("right_elbow")
    left_wrist = landmarks.vector("left_wrist")
    right_wrist = landmarks.vector("right_wrist")
    left_hip = landmarks.vector("left_hip")
    right_hip = landmarks.vector("right_hip")
    left_knee = landmarks.vector("left_knee")
    right_knee = landmarks.vector("right_knee")
    left_ankle = landmarks.vector("left_ankle")
    right_ankle = landmarks.vector("right_ankle")

    # Shoulder orientation relative to a relaxed downward arm.
    reference_arm = np.array([0.0, -1.0, 0.0], dtype=float)
    set_joint("left_shoulder", _rotation_between(reference_arm, left_elbow - left_shoulder))
    set_joint("right_shoulder", _rotation_between(reference_arm, right_elbow - right_shoulder))

    # Elbow and knee hinge angles derived from adjacent limb vectors.
    set_joint(
        "left_elbow",
        _rotation_between(left_shoulder - left_elbow, left_wrist - left_elbow),
    )
    set_joint(
        "right_elbow",
        _rotation_between(right_shoulder - right_elbow, right_wrist - right_elbow),
    )
    set_joint("left_knee", _rotation_between(left_hip - left_knee, left_ankle - left_knee))
    set_joint("right_knee", _rotation_between(right_hip - right_knee, right_ankle - right_knee))

    # Ankles capture flexion relative to the ground plane.
    ground_reference = np.array([0.0, -1.0, 0.0], dtype=float)
    set_joint("left_ankle", _rotation_between(ground_reference, left_foot_direction(left_ankle, landmarks)))
    set_joint("right_ankle", _rotation_between(ground_reference, right_foot_direction(right_ankle, landmarks)))

    return pose.reshape(-1).astype(np.float32, copy=False)


def left_foot_direction(left_ankle: np.ndarray, landmarks: PoseLandmarks) -> np.ndarray:
    left_heel = landmarks.vector("left_heel")
    left_toe = landmarks.vector("left_foot_index")
    direction = left_toe - left_ankle
    if np.linalg.norm(direction) < 1e-6:
        direction = left_toe - left_heel
    return direction


def right_foot_direction(right_ankle: np.ndarray, landmarks: PoseLandmarks) -> np.ndarray:
    right_heel = landmarks.vector("right_heel")
    right_toe = landmarks.vector("right_foot_index")
    direction = right_toe - right_ankle
    if np.linalg.norm(direction) < 1e-6:
        direction = right_toe - right_heel
    return direction


def regress_smplx_from_landmarks(landmarks: PoseLandmarks) -> SMPLXRegressionFrame:
    features = _compute_body_features(landmarks)
    betas = _regress_betas(features)
    body_pose = _estimate_body_pose(landmarks)
    global_orient = _estimate_global_orientation(landmarks)

    left_hip = landmarks.vector("left_hip")
    right_hip = landmarks.vector("right_hip")
    pelvis = 0.5 * (left_hip + right_hip)

    return SMPLXRegressionFrame(
        image_path=landmarks.image_path,
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient.astype(np.float32, copy=False),
        transl=pelvis.astype(np.float32, copy=False),
        measurements=features.measurement_map(),
        confidence=landmarks.confidence,
    )


def _reprojection_fit_from_images(
    image_paths: Sequence[Path],
    *,
    detector: str = "mediapipe",
    refine_with_measurements: bool = True,
    model_path: Path | None = None,
    model_type: str = "smplx",
    gender: str = "neutral",
    iterations: int = 200,
) -> SMPLXRegressionResult:
    torch = _lazy_import_torch()
    detector_used = _coerce_detector_for_reprojection(detector)
    observations = _build_observations(image_paths, detector=detector_used)

    heuristic_frames: list[SMPLXRegressionFrame] = []
    for path in image_paths:
        if detector_used == "mediapipe":
            landmarks = _pose_landmarks_from_mediapipe(path)
        else:
            landmarks = _pose_landmarks_from_bbox(path)
        heuristic_frames.append(regress_smplx_from_landmarks(landmarks))
    init = aggregate_regression_frames(heuristic_frames)

    anchor_measurements = None
    if detector_used == "mediapipe":
        bbox_frames = [regress_smplx_from_landmarks(_pose_landmarks_from_bbox(path)) for path in image_paths]
        anchor_measurements = aggregate_regression_frames(bbox_frames).measurements

    from avatar_model import BodyModel

    batch_size = len(image_paths)
    body = BodyModel(
        model_path=model_path or Path("assets") / model_type,
        model_type=model_type,
        gender=gender,
        batch_size=batch_size,
        num_betas=int(init.betas.shape[0]),
        device="cpu",
    )

    dtype = torch.float32
    body_pose = torch.tensor(
        np.stack([frame.body_pose for frame in heuristic_frames], axis=0),
        dtype=dtype,
        requires_grad=True,
    )
    global_orient = torch.tensor(
        np.stack([frame.global_orient for frame in heuristic_frames], axis=0),
        dtype=dtype,
        requires_grad=True,
    )
    transl = torch.tensor(
        np.stack([frame.transl for frame in heuristic_frames], axis=0),
        dtype=dtype,
        requires_grad=True,
    )
    betas = torch.tensor(
        np.asarray(init.betas, dtype=np.float32)[None, :],
        dtype=dtype,
        requires_grad=True,
    )
    cam_scale = torch.ones((batch_size, 1), dtype=dtype, requires_grad=True)
    cam_shift = torch.zeros((batch_size, 2), dtype=dtype, requires_grad=True)

    observed_names = sorted(
        name for name in _REPROJECTION_JOINT_INDEX.keys()
        if all(name in item.keypoints_2d for item in observations)
    )
    if not observed_names:
        raise RuntimeError("No supported joints were available for reprojection fitting.")
    observed_points = torch.tensor(
        [
            [obs.keypoints_2d[name] for name in observed_names]
            for obs in observations
        ],
        dtype=dtype,
    )
    observed_conf = torch.tensor(
        [
            [obs.confidences.get(name, 1.0) for name in observed_names]
            for obs in observations
        ],
        dtype=dtype,
    )

    bbox_targets = []
    for obs in observations:
        if obs.silhouette_bbox is None:
            bbox_targets.append((0.0, 0.0, 1.0, 1.0))
        else:
            bbox_targets.append(obs.silhouette_bbox)
    bbox_targets_t = torch.tensor(bbox_targets, dtype=dtype)
    init_body_pose = body_pose.detach().clone()
    init_global_orient = global_orient.detach().clone()
    init_betas = betas.detach().clone()

    optimizer = torch.optim.Adam([body_pose, global_orient, transl, betas, cam_scale, cam_shift], lr=0.05)
    loss_history: list[float] = []
    for _ in range(max(iterations, 1)):
        optimizer.zero_grad()
        body.set_shape(betas.repeat(batch_size, 1))
        body.set_body_pose(body_pose=body_pose, global_orient=global_orient, transl=transl)
        joints = body.joints()[:, :22, :]
        model_points = torch.stack(
            [joints[:, _REPROJECTION_JOINT_INDEX[name], :] for name in observed_names],
            dim=1,
        )
        pelvis = joints[:, 0:1, :]
        model_points = model_points - pelvis
        projected = model_points[..., :2] * cam_scale.unsqueeze(1) + cam_shift.unsqueeze(1)
        point_loss = ((projected - observed_points) ** 2 * observed_conf.unsqueeze(-1)).mean()
        x_min, y_min, x_max, y_max = _normalized_bbox_from_points(projected)
        bbox_loss = (
            (x_min - bbox_targets_t[:, 0]) ** 2
            + (y_min - bbox_targets_t[:, 1]) ** 2
            + (x_max - bbox_targets_t[:, 2]) ** 2
            + (y_max - bbox_targets_t[:, 3]) ** 2
        ).mean()
        pose_prior = ((body_pose - init_body_pose) ** 2).mean()
        orient_prior = ((global_orient - init_global_orient) ** 2).mean()
        shape_prior = (betas ** 2).mean() + ((betas - init_betas) ** 2).mean()
        loss = point_loss + 0.25 * bbox_loss + 0.01 * pose_prior + 0.01 * orient_prior + 0.01 * shape_prior
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            cam_scale.clamp_(0.2, 3.0)
            cam_shift.clamp_(-2.0, 2.0)
        loss_history.append(float(loss.detach().cpu()))

    final_body_pose = body_pose.detach().cpu().numpy()
    final_global_orient = global_orient.detach().cpu().numpy()
    final_transl = transl.detach().cpu().numpy()
    final_betas = betas.detach().cpu().numpy()[0]
    body.set_shape(np.repeat(final_betas[None, :], batch_size, axis=0))
    body.set_body_pose(body_pose=final_body_pose, global_orient=final_global_orient, transl=final_transl)
    joints = body.joints().detach().cpu().numpy()[:, :22, :]
    model_points = np.stack(
        [joints[:, _REPROJECTION_JOINT_INDEX[name], :] for name in observed_names],
        axis=1,
    )
    model_points = model_points - joints[:, 0:1, :]
    projected = model_points[..., :2] * cam_scale.detach().cpu().numpy()[:, None, :] + cam_shift.detach().cpu().numpy()[:, None, :]
    reprojection_error = np.sqrt(np.mean((projected - observed_points.detach().cpu().numpy()) ** 2, axis=(1, 2)))

    frames = tuple(
        SMPLXRegressionFrame(
            image_path=obs.image_path,
            betas=final_betas.astype(np.float32, copy=False),
            body_pose=final_body_pose[idx].astype(np.float32, copy=False),
            global_orient=final_global_orient[idx].astype(np.float32, copy=False),
            transl=final_transl[idx].astype(np.float32, copy=False),
            measurements=heuristic_frames[idx].measurements,
            confidence=float(np.mean(list(obs.confidences.values()))) if obs.confidences else 0.0,
        )
        for idx, obs in enumerate(observations)
    )
    result = aggregate_regression_frames(frames)
    optimization_report = {
        "optimizer": "adam",
        "iterations": int(max(iterations, 1)),
        "loss_initial": float(loss_history[0]) if loss_history else 0.0,
        "loss_final": float(loss_history[-1]) if loss_history else 0.0,
        "observed_joint_names": observed_names,
        "per_image_reprojection_rmse": reprojection_error.tolist(),
        "mean_reprojection_rmse": float(np.mean(reprojection_error)),
        "camera_scale": cam_scale.detach().cpu().numpy().reshape(-1).tolist(),
        "camera_shift": cam_shift.detach().cpu().numpy().tolist(),
        "dependency_tier": "reprojection",
    }
    calibrated_measurements, calibration_report = _calibrate_reprojection_measurements(
        init.measurements,
        anchor=anchor_measurements,
        detector=detector_used,
    )
    if calibration_report is not None:
        optimization_report["measurement_calibration"] = calibration_report
    result = replace(
        result,
        detector=detector_used,
        measurement_source=(
            "reprojection_keypoints_calibrated_with_bbox_anchor"
            if calibration_report is not None
            else "reprojection_keypoints"
        ),
        fit_mode="reprojection_only",
        observations=observations,
        optimization_report=optimization_report,
        measurements=calibrated_measurements,
    )

    if refine_with_measurements and result.measurements:
        from smii.pipelines.fit_from_measurements import fit_smplx_from_measurements

        result = replace(
            result,
            measurement_fit=fit_smplx_from_measurements(
                result.measurements,
                provenance={
                    "images_used": [str(path) for path in image_paths],
                    "detector": detector_used,
                    "measurement_source": "reprojection_keypoints",
                    "refinement_applied": True,
                },
                raw_measurements={name: float(value) for name, value in result.measurements.items()},
                fit_mode="reprojection_plus_measurement_refinement",
                trust_level="high" if detector_used != "bbox" else "coarse",
                consistency_status="PASS",
            ),
        )
        result = replace(result, fit_mode="reprojection_plus_measurement_refinement")

    return finalize_regression_diagnostics(result)


def aggregate_regression_frames(frames: Sequence[SMPLXRegressionFrame]) -> SMPLXRegressionResult:
    if not frames:
        raise ValueError("At least one regression frame is required to aggregate results.")

    betas = np.mean([frame.betas for frame in frames], axis=0)
    body_pose = np.mean([frame.body_pose for frame in frames], axis=0)
    global_orient = np.mean([frame.global_orient for frame in frames], axis=0)
    transl = np.mean([frame.transl for frame in frames], axis=0)

    measurement_totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for frame in frames:
        for name, value in frame.measurements.items():
            measurement_totals[name] += float(value)
            counts[name] += 1

    averaged_measurements = {
        name: measurement_totals[name] / counts[name] for name in measurement_totals
    }

    return SMPLXRegressionResult(
        betas=betas.astype(np.float32, copy=False),
        body_pose=body_pose.astype(np.float32, copy=False),
        global_orient=global_orient.astype(np.float32, copy=False),
        transl=transl.astype(np.float32, copy=False),
        measurements=averaged_measurements,
        frames=tuple(frames),
    )


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
    """Regress SMPL-X parameters from a collection of RGB images."""

    paths = _normalise_image_paths(image_paths)

    if fit_mode not in {"heuristic", "reprojection", "auto"}:
        raise ValueError(
            f"Unsupported fit_mode '{fit_mode}'. Choose from 'heuristic', 'reprojection', or 'auto'."
        )
    effective_fit_mode = fit_mode
    if effective_fit_mode == "auto":
        effective_fit_mode = "reprojection"
    if effective_fit_mode == "reprojection":
        try:
            return _reprojection_fit_from_images(
                paths,
                detector=detector,
                refine_with_measurements=refine_with_measurements,
                model_path=model_path,
                model_type=model_type,
                gender=gender,
            )
        except Exception as exc:
            if fit_mode != "auto":
                raise
            fallback_reason = f"reprojection_fallback:{type(exc).__name__}"
        else:
            fallback_reason = ""
    else:
        fallback_reason = ""

    frames: list[SMPLXRegressionFrame] = []
    for path in paths:
        if detector == "mediapipe":
            landmarks = _pose_landmarks_from_mediapipe(path)
        elif detector == "bbox":
            landmarks = _pose_landmarks_from_bbox(path)
        else:
            raise ValueError(
                f"Unsupported detector '{detector}'. Choose from 'mediapipe' or 'bbox'."
            )
        frames.append(regress_smplx_from_landmarks(landmarks))

    result = aggregate_regression_frames(frames)

    if refine_with_measurements and result.measurements:
        from smii.pipelines.fit_from_measurements import fit_smplx_from_measurements

        result = replace(
            result,
            measurement_fit=fit_smplx_from_measurements(result.measurements),
        )

    result = replace(
        result,
        detector=detector,
        measurement_source="raw_image_features",
        fit_mode=(
            "image_regression_plus_measurement_refinement"
            if refine_with_measurements and result.measurement_fit is not None
            else "image_regression_only"
        ),
    )
    result = finalize_regression_diagnostics(result)
    if fallback_reason:
        result = replace(
            result,
            consistency_status="WARN" if result.consistency_status == "PASS" else result.consistency_status,
            consistency_flags=tuple(dict.fromkeys((*result.consistency_flags, fallback_reason))),
        )
    return result


def save_regression_json(result: SMPLXRegressionResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path


def save_fit_diagnostics_report(result: SMPLXRegressionResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_fit_diagnostics_report(result), indent=2), encoding="utf-8")
    return path


def create_body_mesh_from_regression(
    result: SMPLXRegressionResult,
    *,
    model_path: Path | None = None,
    model_type: str = "smplx",
    gender: str = "neutral",
    use_measurement_refinement: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a watertight mesh from a regression result."""

    from avatar_model import BodyModel

    betas = result.refined_betas() if use_measurement_refinement else result.betas
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        betas = betas.reshape(1, -1)

    transl = (
        np.asarray(result.measurement_fit.translation, dtype=np.float32).reshape(1, 3)
        if use_measurement_refinement and result.measurement_fit is not None
        else np.asarray(result.transl, dtype=np.float32).reshape(1, 3)
    )

    scale = (
        float(result.measurement_fit.scale)
        if use_measurement_refinement and result.measurement_fit is not None
        else 1.0
    )

    body = BodyModel(
        model_path=model_path or Path("assets") / model_type,
        model_type=model_type,
        gender=gender,
        batch_size=1,
        num_betas=int(betas.shape[1]),
    )

    body.set_shape(betas)

    parameters: dict[str, np.ndarray] = {}
    if result.expression is not None:
        parameters["expression"] = np.asarray(result.expression, dtype=np.float32).reshape(1, -1)
    if result.jaw_pose is not None:
        parameters["jaw_pose"] = np.asarray(result.jaw_pose, dtype=np.float32).reshape(1, -1)
    if result.left_hand_pose is not None:
        parameters["left_hand_pose"] = np.asarray(result.left_hand_pose, dtype=np.float32).reshape(1, -1)
    if result.right_hand_pose is not None:
        parameters["right_hand_pose"] = np.asarray(result.right_hand_pose, dtype=np.float32).reshape(1, -1)
    if parameters:
        body.set_parameters(parameters)

    body.set_body_pose(
        body_pose=np.asarray(result.body_pose, dtype=np.float32).reshape(1, -1),
        global_orient=np.asarray(result.global_orient, dtype=np.float32).reshape(1, -1),
        transl=transl,
    )

    vertices = body.vertices().detach().cpu().numpy()[0]
    vertices = (vertices * scale).astype(np.float32, copy=False)
    faces = np.asarray(getattr(body.model, "faces"), dtype=np.int32)
    return vertices, faces


def save_regression_mesh(
    result: SMPLXRegressionResult,
    path: Path,
    *,
    model_path: Path | None = None,
    model_type: str = "smplx",
    gender: str = "neutral",
    use_measurement_refinement: bool = True,
) -> Path:
    vertices, faces = create_body_mesh_from_regression(
        result,
        model_path=model_path,
        model_type=model_type,
        gender=gender,
        use_measurement_refinement=use_measurement_refinement,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vertices=vertices, faces=faces)
    return path


__all__ = [
    "AfflecImageMeasurementExtractor",
    "MeasurementExtractionError",
    "extract_measurements_from_afflec_images",
    "PoseLandmarks",
    "BodyFeatures",
    "SMPLXRegressionFrame",
    "SMPLXRegressionResult",
    "aggregate_regression_frames",
    "create_body_mesh_from_regression",
    "fit_smplx_from_images",
    "regress_smplx_from_images",
    "regress_smplx_from_landmarks",
    "build_fit_diagnostics_report",
    "save_image_fit_observations",
    "save_fit_diagnostics_report",
    "save_regression_json",
    "save_regression_mesh",
]
