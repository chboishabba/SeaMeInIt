from pathlib import Path

import pytest

from smii.pipelines.fit_from_images import (
    AfflecImageMeasurementExtractor,
    MeasurementExtractionError,
    extract_measurements_from_afflec_images,
)

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
