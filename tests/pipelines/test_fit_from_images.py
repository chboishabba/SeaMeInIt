import dataclasses
from pathlib import Path

import pytest

import numpy as np

from pipelines.measurement_inference import MeasurementEstimate, MeasurementReport

from smii.pipelines.fit_from_images import (
    AfflecImageMeasurementExtractor,
    MeasurementExtractionError,
    SMPLXRegressionFrame,
    aggregate_regression_frames,
    extract_measurements_from_afflec_images,
)


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


def test_extract_measurements_from_single_image():
    extractor = AfflecImageMeasurementExtractor()
    path = FIXTURE_DIR / "sample_front.pgm"
    measurements = extractor.parse_measurements(path)
    assert measurements == {
        "height": 170.0,
        "chest_circumference": 96.5,
        "shoulder_width": 42.2,
    }


def test_batch_extract_merges_measurements():
    measurements = extract_measurements_from_afflec_images(
        [FIXTURE_DIR / "sample_front.pgm", FIXTURE_DIR / "sample_side.pgm"]
    )
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
    with pytest.raises(MeasurementExtractionError):
        extract_measurements_from_afflec_images(
            [FIXTURE_DIR / "sample_front.pgm", conflicting]
        )


def test_parse_measurements_requires_metadata(tmp_path: Path):
    missing = tmp_path / "missing.pgm"
    missing.write_text("P2\n2 2\n255\n0 0\n0 0\n", encoding="utf-8")
    extractor = AfflecImageMeasurementExtractor()
    with pytest.raises(MeasurementExtractionError):
        extractor.parse_measurements(missing)


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
