import dataclasses
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

import numpy as np

from pipelines.measurement_inference import MeasurementEstimate, MeasurementReport

from smii.pipelines.fit_from_images import (
    AfflecImageMeasurementExtractor,
    ImageFitObservation,
    MeasurementExtractionError,
    SMPLXRegressionFrame,
    _calibrate_reprojection_measurements,
    aggregate_regression_frames,
    extract_measurements_from_afflec_images,
    fit_smplx_from_images,
    regress_smplx_from_images,
)
from smii import app

if "jsonschema" not in sys.modules:
    jsonschema_stub = ModuleType("jsonschema")

    class _ValidationError(Exception):
        """Placeholder validation error for jsonschema-free testing."""

    jsonschema_stub.Draft202012Validator = object
    jsonschema_stub.ValidationError = _ValidationError
    sys.modules["jsonschema"] = jsonschema_stub

@dataclasses.dataclass(frozen=True)
class DummyFitResult:
    betas: np.ndarray
    scale: float
    translation: np.ndarray
    residual: float
    measurements_used: tuple[str, ...]
    measurement_report: MeasurementReport

    def to_dict(self) -> dict[str, object]:
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

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "afflec"
AFFLEC_PHOTOS = [
    FIXTURE_DIR / "afflec1.jpg",
    FIXTURE_DIR / "afflec2.jpg",
    FIXTURE_DIR / "afflec3.avif",
]
PGM_FRONT = next((p for p in FIXTURE_DIR.glob("*front*pgm")), None)
PGM_SIDE = next((p for p in FIXTURE_DIR.glob("*side*pgm")), None)


def test_expand_image_inputs_covers_afflec_photos_when_present():
    missing = [p for p in AFFLEC_PHOTOS if not p.exists()]
    if missing:
        pytest.skip(f"Afflec photo fixtures missing: {', '.join(str(m) for m in missing)}")

    expanded = app._expand_image_inputs([FIXTURE_DIR])
    for photo in AFFLEC_PHOTOS:
        assert photo in expanded
    # Legacy PGM fixtures are intentionally ignored by the default expander.
    assert all(not p.name.lower().endswith("pgm") for p in expanded)


def test_measurement_record_exists_for_afflec_fixture():
    record = FIXTURE_DIR / "measurements.yaml"
    assert record.exists()


def test_extract_measurements_from_single_image():
    extractor = AfflecImageMeasurementExtractor()
    if PGM_FRONT is None:
        pytest.skip("Afflec PGM front fixture missing")
    path = PGM_FRONT
    measurements = extractor.parse_measurements(path)
    assert measurements == {
        "height": 170.0,
        "chest_circumference": 96.5,
        "shoulder_width": 42.2,
    }


def test_batch_extract_merges_measurements():
    if PGM_FRONT is None or PGM_SIDE is None:
        pytest.skip("Afflec PGM fixtures missing")
    measurements = extract_measurements_from_afflec_images([PGM_FRONT, PGM_SIDE])
    assert measurements == {
        "height": 170.0,
        "chest_circumference": 96.5,
        "shoulder_width": 42.2,
        "waist_circumference": 82.3,
        "hip_circumference": 98.1,
    }


def test_batch_extract_raises_on_conflicts(tmp_path: Path):
    conflicting = tmp_path / "conflict.pgm"
    conflicting.write_text(
        """P2\n# measurement:height=165.0\n2 2\n255\n0 0\n0 0\n""",
        encoding="utf-8",
    )
    if PGM_FRONT is None:
        pytest.skip("Afflec PGM fixtures missing")
    with pytest.raises(MeasurementExtractionError):
        extract_measurements_from_afflec_images([PGM_FRONT, conflicting])


def test_parse_measurements_requires_metadata(tmp_path: Path):
    missing = tmp_path / "missing.pgm"
    missing.write_text("P2\n2 2\n255\n0 0\n0 0\n", encoding="utf-8")
    extractor = AfflecImageMeasurementExtractor()
    with pytest.raises(MeasurementExtractionError):
        extractor.parse_measurements(missing)


def test_fit_from_images_uses_embedded_metadata_when_regressor_missing(monkeypatch: pytest.MonkeyPatch):
    called: dict[str, object] = {}
    if PGM_FRONT is None or PGM_SIDE is None:
        pytest.skip("Afflec PGM fixtures missing")

    def fake_fit_from_measurements(measurements, **kwargs):
        called["measurements"] = measurements
        called["kwargs"] = kwargs
        return "fit-result"

    monkeypatch.setattr(
        "smii.pipelines.fit_from_measurements.fit_smplx_from_measurements",
        fake_fit_from_measurements,
    )
    monkeypatch.delitem(sys.modules, "pipelines.afflec_regression", raising=False)
    monkeypatch.setattr(
        "smii.pipelines.fit_from_images.regress_smplx_from_images",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("regression should not be called")),
    )

    result = fit_smplx_from_images([PGM_FRONT, PGM_SIDE])

    assert result == "fit-result"
    assert called["measurements"] == {
        "height": 170.0,
        "chest_circumference": 96.5,
        "shoulder_width": 42.2,
        "waist_circumference": 82.3,
        "hip_circumference": 98.1,
    }
    assert called["kwargs"]["backend"] == "smplx"
    assert called["kwargs"]["schema_path"] is None
    assert called["kwargs"]["models"] is None
    assert called["kwargs"]["num_shape_coeffs"] is None
    assert called["kwargs"]["inference_model"] is None
    assert called["kwargs"]["fit_mode"] == "pgm_measurement_refinement"


def test_fit_from_images_prefers_regressor_when_available(monkeypatch: pytest.MonkeyPatch):
    called: dict[str, object] = {}

    def fake_fit_from_measurements(measurements, **kwargs):
        called["measurements"] = measurements
        called["kwargs"] = kwargs
        return "regressed-fit"

    def fake_regress(paths, detector="mediapipe", refine_with_measurements=False, **kwargs):
        called["regress_paths"] = tuple(paths)
        called["detector"] = detector
        called["regress_refine"] = refine_with_measurements
        called["regress_kwargs"] = kwargs
        return SimpleNamespace(measurements={"height": 180.0, "waist_circumference": 84.0})

    monkeypatch.setattr(
        "smii.pipelines.fit_from_images.regress_smplx_from_images",
        fake_regress,
    )
    monkeypatch.setattr(
        "smii.pipelines.fit_from_measurements.fit_smplx_from_measurements",
        fake_fit_from_measurements,
    )
    missing = [p for p in AFFLEC_PHOTOS if not p.exists()]
    if missing or PGM_FRONT is None:
        pytest.skip("Afflec fixtures missing")
    paths = AFFLEC_PHOTOS + [PGM_FRONT]
    result = fit_smplx_from_images(paths)

    assert result == "regressed-fit"
    assert called["regress_paths"] == tuple(paths)
    assert called["regress_refine"] is False
    assert called["regress_kwargs"]["fit_mode"] == "heuristic"
    assert called["measurements"] == {
        "height": 180.0,
        "waist_circumference": 84.0,
    }
    assert called["kwargs"]["backend"] == "smplx"
    assert called["kwargs"]["fit_mode"] == "image_regression_plus_measurement_refinement"


def test_fit_from_images_runs_regression_when_photos_present(monkeypatch: pytest.MonkeyPatch):
    missing = [p for p in AFFLEC_PHOTOS if not p.exists()]
    if missing:
        pytest.skip(f"Afflec photo fixtures missing: {', '.join(str(m) for m in missing)}")

    called: dict[str, object] = {}

    def fake_regress(paths, detector="mediapipe", refine_with_measurements=False, **kwargs):
        called["regress_paths"] = tuple(paths)
        called["regress_refine"] = refine_with_measurements
        called["regress_kwargs"] = kwargs
        return SimpleNamespace(measurements={"height": 172.0})

    def fake_fit_from_measurements(measurements, **kwargs):
        called["measurements"] = measurements
        called["kwargs"] = kwargs
        return "photo-regression-fit"

    monkeypatch.delitem(sys.modules, "pipelines.afflec_regression", raising=False)
    monkeypatch.setattr(
        "smii.pipelines.fit_from_images.regress_smplx_from_images",
        fake_regress,
    )
    monkeypatch.setattr(
        "smii.pipelines.fit_from_images.create_body_mesh_from_regression",
        lambda *args, **kwargs: (
            np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]]),
            np.zeros((0, 3), dtype=np.int32),
        ),
    )
    monkeypatch.setattr(
        "smii.pipelines.fit_from_measurements.fit_smplx_from_measurements",
        fake_fit_from_measurements,
    )

    # Mix real photos with the PGM fixtures; regression should still drive measurement inference.
    if PGM_FRONT is None or PGM_SIDE is None:
        pytest.skip("Afflec PGM fixtures missing")
    paths = AFFLEC_PHOTOS + [PGM_FRONT, PGM_SIDE]

    result = fit_smplx_from_images(paths)

    assert result == "photo-regression-fit"
    assert called["regress_paths"] == tuple(paths)
    assert called["regress_refine"] is False
    assert called["regress_kwargs"]["fit_mode"] == "heuristic"
    assert called["measurements"] == {
        "height": 172.0,
    }
    assert called["kwargs"]["backend"] == "smplx"
    assert called["kwargs"]["fit_mode"] == "image_regression_plus_measurement_refinement"


def test_fit_from_images_derives_measurements_from_photos_when_no_pgm(monkeypatch: pytest.MonkeyPatch):
    missing = [p for p in AFFLEC_PHOTOS if not p.exists()]
    if missing:
        pytest.skip(f"Afflec photo fixtures missing: {', '.join(str(m) for m in missing)}")

    called: dict[str, object] = {}

    monkeypatch.delitem(sys.modules, "pipelines.afflec_regression", raising=False)
    def fake_regress(paths, detector="mediapipe", refine_with_measurements=False, **kwargs):
        called["regress_paths"] = tuple(paths)
        called["regress_kwargs"] = kwargs
        return SimpleNamespace(measurements={"height": 171.5, "waist_circumference": 90.0})

    def fake_fit_from_measurements(measurements, **kwargs):
        called["measurements"] = measurements
        called["kwargs"] = kwargs
        return "photo-derived-fit"

    monkeypatch.setattr(
        "smii.pipelines.fit_from_images.regress_smplx_from_images",
        fake_regress,
    )
    monkeypatch.setattr(
        "smii.pipelines.fit_from_images.create_body_mesh_from_regression",
        lambda *args, **kwargs: (
            np.array([[0.0, 0.0, 0.0], [0.0, 1.8, 0.0], [0.5, 0.0, 0.0]]),
            np.zeros((0, 3), dtype=np.int32),
        ),
    )
    monkeypatch.setattr(
        "smii.pipelines.fit_from_measurements.fit_smplx_from_measurements",
        fake_fit_from_measurements,
    )

    result = fit_smplx_from_images(AFFLEC_PHOTOS)

    assert result == "photo-derived-fit"
    assert called["regress_paths"] == tuple(AFFLEC_PHOTOS)
    assert called["regress_kwargs"]["fit_mode"] == "heuristic"
    assert called["measurements"] == {
        "height": 171.5,
        "waist_circumference": 90.0,
    }
    assert called["kwargs"]["backend"] == "smplx"
    assert called["kwargs"]["fit_mode"] == "image_regression_plus_measurement_refinement"
def _regression_frame(seed: float) -> SMPLXRegressionFrame:
    return SMPLXRegressionFrame(
        image_path=Path(f"frame_{seed:.0f}.jpg"),
        betas=np.full(10, seed, dtype=float),
        body_pose=np.full(63, seed, dtype=float),
        global_orient=np.array([seed, seed + 1.0, seed + 2.0], dtype=float),
        transl=np.array([seed, seed * 2.0, seed * 3.0], dtype=float),
        measurements={"height": 170.0 + seed, "shoulder_width": 40.0 + seed},
        confidence=0.9,
    )


def test_aggregate_regression_frames_averages_parameters():
    frame_a = _regression_frame(1.0)
    frame_b = _regression_frame(3.0)
    result = aggregate_regression_frames([frame_a, frame_b])

    assert np.allclose(result.betas, np.full(10, 2.0))
    assert np.allclose(result.body_pose, np.full(63, 2.0))
    assert np.allclose(result.global_orient, np.array([2.0, 3.0, 4.0]))
    assert np.allclose(result.transl, np.array([2.0, 4.0, 6.0]))
    assert result.measurements["height"] == pytest.approx(172.0)
    assert result.measurements["shoulder_width"] == pytest.approx(42.0)


def test_regression_result_includes_measurement_refinement():
    frame = _regression_frame(0.0)
    measurement = MeasurementEstimate(
        name="height",
        value=170.0,
        source="measured",
        confidence=1.0,
        variance=0.0,
    )
    report = MeasurementReport(estimates=(measurement,), coverage=1.0)
    fit_result = DummyFitResult(
        betas=np.ones(10),
        scale=1.0,
        translation=np.zeros(3),
        residual=0.0,
        measurements_used=("height",),
        measurement_report=report,
    )

    result = aggregate_regression_frames([frame])
    result = dataclasses.replace(result, measurement_fit=fit_result)
    payload = result.to_dict()
    assert "measurement_refinement" in payload


def test_reprojection_fit_emits_optimization_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    torch = pytest.importorskip("torch")

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"stub")

    observation = ImageFitObservation(
        image_path=image_path,
        width=100,
        height=200,
        keypoints_2d={
            "left_shoulder": (0.35, 0.2),
            "right_shoulder": (0.65, 0.2),
            "left_elbow": (0.3, 0.4),
            "right_elbow": (0.7, 0.4),
            "left_wrist": (0.28, 0.55),
            "right_wrist": (0.72, 0.55),
            "left_hip": (0.42, 0.62),
            "right_hip": (0.58, 0.62),
            "left_knee": (0.44, 0.8),
            "right_knee": (0.56, 0.8),
            "left_ankle": (0.45, 0.96),
            "right_ankle": (0.55, 0.96),
            "neck": (0.5, 0.16),
            "head": (0.5, 0.06),
            "pelvis": (0.5, 0.62),
        },
        confidences={name: 1.0 for name in {
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
            "right_knee", "left_ankle", "right_ankle", "neck", "head", "pelvis"
        }},
        silhouette_bbox=(0.2, 0.05, 0.8, 0.98),
        detector="bbox",
    )

    def fake_build_observations(paths, *, detector):
        return (observation,)

    def fake_pose_landmarks(path):
        return SimpleNamespace(
            image_path=path,
            confidence=1.0,
            vector=lambda name: {
                "nose": np.array([0.5, 0.05, 0.0]),
                "left_shoulder": np.array([0.35, 0.2, 0.0]),
                "right_shoulder": np.array([0.65, 0.2, 0.0]),
                "left_elbow": np.array([0.3, 0.4, 0.0]),
                "right_elbow": np.array([0.7, 0.4, 0.0]),
                "left_wrist": np.array([0.28, 0.55, 0.0]),
                "right_wrist": np.array([0.72, 0.55, 0.0]),
                "left_hip": np.array([0.42, 0.62, 0.0]),
                "right_hip": np.array([0.58, 0.62, 0.0]),
                "left_knee": np.array([0.44, 0.8, 0.0]),
                "right_knee": np.array([0.56, 0.8, 0.0]),
                "left_ankle": np.array([0.45, 0.96, 0.0]),
                "right_ankle": np.array([0.55, 0.96, 0.0]),
                "left_heel": np.array([0.45, 0.98, 0.0]),
                "right_heel": np.array([0.55, 0.98, 0.0]),
                "left_foot_index": np.array([0.46, 1.0, 0.0]),
                "right_foot_index": np.array([0.54, 1.0, 0.0]),
            }[name],
        )

    class DummyBodyModel:
        def __init__(self, **kwargs):
            self.batch_size = int(kwargs.get("batch_size", 1))
            self._betas = torch.zeros((self.batch_size, 10), dtype=torch.float32)
            self._body_pose = torch.zeros((self.batch_size, 63), dtype=torch.float32)
            self._global_orient = torch.zeros((self.batch_size, 3), dtype=torch.float32)
            self._transl = torch.zeros((self.batch_size, 3), dtype=torch.float32)

        def set_shape(self, betas):
            self._betas = torch.as_tensor(betas, dtype=torch.float32)

        def set_body_pose(self, body_pose=None, global_orient=None, transl=None):
            if body_pose is not None:
                self._body_pose = torch.as_tensor(body_pose, dtype=torch.float32)
            if global_orient is not None:
                self._global_orient = torch.as_tensor(global_orient, dtype=torch.float32)
            if transl is not None:
                self._transl = torch.as_tensor(transl, dtype=torch.float32)

        def joints(self):
            batch = self._betas.shape[0]
            joints = torch.zeros((batch, 22, 3), dtype=torch.float32)
            joints[:, 0, :2] = torch.tensor([0.5, 0.62])
            joints[:, 1, :2] = torch.tensor([0.42, 0.62])
            joints[:, 2, :2] = torch.tensor([0.58, 0.62])
            joints[:, 4, :2] = torch.tensor([0.44, 0.80])
            joints[:, 5, :2] = torch.tensor([0.56, 0.80])
            joints[:, 7, :2] = torch.tensor([0.45, 0.96])
            joints[:, 8, :2] = torch.tensor([0.55, 0.96])
            joints[:, 12, :2] = torch.tensor([0.50, 0.16])
            joints[:, 15, :2] = torch.tensor([0.50, 0.06])
            joints[:, 16, :2] = torch.tensor([0.35, 0.20])
            joints[:, 17, :2] = torch.tensor([0.65, 0.20])
            joints[:, 18, :2] = torch.tensor([0.30, 0.40])
            joints[:, 19, :2] = torch.tensor([0.70, 0.40])
            joints[:, 20, :2] = torch.tensor([0.28, 0.55])
            joints[:, 21, :2] = torch.tensor([0.72, 0.55])
            joints = joints + self._transl[:, None, :]
            return joints

    dummy_avatar_module = ModuleType("avatar_model")
    dummy_avatar_module.BodyModel = DummyBodyModel
    monkeypatch.setitem(sys.modules, "avatar_model", dummy_avatar_module)
    monkeypatch.setattr("smii.pipelines.fit_from_images._build_observations", fake_build_observations)
    monkeypatch.setattr("smii.pipelines.fit_from_images._pose_landmarks_from_bbox", fake_pose_landmarks)
    monkeypatch.setattr(
        "smii.pipelines.fit_from_measurements.fit_smplx_from_measurements",
        lambda measurements, **kwargs: DummyFitResult(
            betas=np.zeros(10),
            scale=1.0,
            translation=np.zeros(3),
            residual=0.0,
            measurements_used=tuple(sorted(measurements)),
            measurement_report=MeasurementReport(estimates=(), coverage=0.0),
        ),
    )

    result = regress_smplx_from_images(
        [image_path],
        detector="bbox",
        refine_with_measurements=False,
        fit_mode="reprojection",
        model_path=tmp_path,
    )

    assert result.fit_mode == "reprojection_only"
    assert result.observations
    assert result.optimization_report is not None
    assert result.optimization_report["optimizer"] == "adam"
    assert result.optimization_report["mean_reprojection_rmse"] >= 0.0


def test_reprojection_measurements_can_be_calibrated_against_bbox_anchor():
    primary = {
        "height": 135.0,
        "shoulder_width": 32.0,
        "chest_circumference": 83.0,
        "waist_circumference": 62.0,
        "hip_circumference": 75.0,
        "arm_length": 44.0,
        "leg_length": 79.0,
        "torso_length": 49.0,
    }
    anchor = {
        "height": 168.0,
        "shoulder_width": 47.5,
        "chest_circumference": 116.0,
        "waist_circumference": 79.0,
        "hip_circumference": 104.0,
        "arm_length": 65.0,
        "leg_length": 58.0,
        "torso_length": 75.0,
    }

    calibrated, report = _calibrate_reprojection_measurements(
        primary,
        anchor=anchor,
        detector="mediapipe",
    )

    assert report is not None
    assert report["applied"] is True
    assert calibrated["height"] > primary["height"]
    assert calibrated["shoulder_width"] > primary["shoulder_width"]
    assert calibrated["chest_circumference"] > primary["chest_circumference"]
