"""Utilities for fitting SMPL-X parameters directly from imagery."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TYPE_CHECKING

import warnings

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for typing only
    from pipelines.measurement_inference import GaussianMeasurementModel
    from .fit_from_measurements import FitResult, MeasurementModel
import json
import math

import numpy as np

HEADER_PREFIX = "# measurement:"

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


def _infer_measurements_from_images(paths: Sequence[Path]) -> dict[str, float]:
    try:
        from pipelines.afflec_regression import regress_measurements_from_images
    except ModuleNotFoundError:  # pragma: no cover - exercised in integration usage
        warnings.warn(
            "pipelines.afflec_regression is not available (and is not shipped here); using the Ben Afflec fixture metadata instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return extract_measurements_from_afflec_images(paths)
    return regress_measurements_from_images(paths)


def fit_smplx_from_images(
    image_paths: Iterable[Path],
    *,
    backend: str = "smplx",
    schema_path: Path | None = None,
    models: Sequence["MeasurementModel"] | None = None,
    num_shape_coeffs: int | None = None,
    inference_model: "GaussianMeasurementModel | None" = None,
) -> "FitResult":
    """Fit SMPL-X parameters by inferring measurements from annotated PGM images."""

    paths = _normalise_image_paths(image_paths)
    measurements = _infer_measurements_from_images(paths)

    from smii.pipelines.fit_from_measurements import fit_smplx_from_measurements

    return fit_smplx_from_measurements(
        measurements,
        backend=backend,
        schema_path=schema_path,
        models=models,
        num_shape_coeffs=num_shape_coeffs,
        inference_model=inference_model,
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

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "betas": self.betas.tolist(),
            "body_pose": self.body_pose.tolist(),
            "global_orient": self.global_orient.tolist(),
            "transl": self.transl.tolist(),
            "measurements": {name: float(value) for name, value in self.measurements.items()},
            "per_view": [frame.to_dict() for frame in self.frames],
        }
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


def _lazy_import_pillow() -> "module":
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "Pillow is required for image-based SMPL-X fitting. Install the 'seameinit[vision]' extra."
        ) from exc
    return Image


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
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2)
    try:
        result = pose.process(image[:, :, ::-1])
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
) -> SMPLXRegressionResult:
    """Regress SMPL-X parameters from a collection of RGB images."""

    paths = _normalise_image_paths(image_paths)

    frames: list[SMPLXRegressionFrame] = []
    for path in paths:
        if detector != "mediapipe":
            raise ValueError(f"Unsupported detector '{detector}'. Only 'mediapipe' is currently available.")
        landmarks = _pose_landmarks_from_mediapipe(path)
        frames.append(regress_smplx_from_landmarks(landmarks))

    result = aggregate_regression_frames(frames)

    if refine_with_measurements and result.measurements:
        from smii.pipelines.fit_from_measurements import fit_smplx_from_measurements

        result = replace(
            result,
            measurement_fit=fit_smplx_from_measurements(result.measurements),
        )

    return result


def save_regression_json(result: SMPLXRegressionResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
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
    "save_regression_json",
    "save_regression_mesh",
]
