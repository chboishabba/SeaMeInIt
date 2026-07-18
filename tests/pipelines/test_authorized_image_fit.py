from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from smii.pipelines import authorized_image_fit


class MeasurementReportStub:
    coverage = 1.0

    def values(self) -> dict[str, float]:
        return {
            "height": 170.0,
            "chest_circumference": 95.0,
            "waist_circumference": 80.0,
            "hip_circumference": 98.0,
            "shoulder_width": 42.0,
            "neck_circumference": 36.0,
            "arm_length": 60.0,
            "inseam_length": 78.0,
            "thigh_circumference": 55.0,
            "calf_circumference": 37.0,
            "bicep_circumference": 30.0,
            "wrist_circumference": 16.0,
        }

    def visualization_payload(self) -> list[object]:
        return []


class InferenceModelStub:
    def infer(self, values: object) -> MeasurementReportStub:
        del values
        return MeasurementReportStub()


@dataclass(frozen=True)
class RawRegressionStub:
    measurements: dict[str, float]
    betas: np.ndarray
    transl: np.ndarray
    frames: tuple[object, ...]
    detector: str
    measurement_source: str
    fit_mode: str
    trust_level: str
    consistency_status: str
    consistency_flags: tuple[str, ...]
    optimization_report: dict[str, float]
    measurement_fit: object | None


def raw_regression(flags: tuple[str, ...] = ()) -> RawRegressionStub:
    return RawRegressionStub(
        measurements={"height": 170.0},
        betas=np.zeros(10),
        transl=np.zeros(3),
        frames=(SimpleNamespace(image_path=Path("x.png")),),
        detector="mediapipe",
        measurement_source="raw",
        fit_mode="image_regression_only",
        trust_level="high",
        consistency_status="WARN" if flags else "PASS",
        consistency_flags=flags,
        optimization_report={},
        measurement_fit=None,
    )


def test_guarded_candidate_promotes_when_supported(monkeypatch: object) -> None:
    monkeypatch.setattr(authorized_image_fit, "load_default_model", lambda: InferenceModelStub())

    result = authorized_image_fit.fit_anchored_measurement_candidate(
        {"height": 170.0},
        anchor_betas=np.zeros(10),
        consistency_status="PASS",
    )

    assert result.refinement_receipt.decision == "promote"
    np.testing.assert_allclose(result.betas, np.zeros(10))


def test_reference_warning_abstains_and_keeps_raw_betas(monkeypatch: object) -> None:
    monkeypatch.setattr(authorized_image_fit, "load_default_model", lambda: InferenceModelStub())
    anchor = np.linspace(-0.5, 0.5, 10)

    result = authorized_image_fit.fit_anchored_measurement_candidate(
        {"height": 170.0},
        anchor_betas=anchor,
        anchor_scale=1.2,
        consistency_status="WARN",
        consistency_flags=("WARN:low_view_diversity",),
    )

    assert result.refinement_receipt.decision == "abstain"
    np.testing.assert_allclose(result.betas, anchor)
    assert result.scale == 1.2


def test_image_wrapper_disables_legacy_refinement(monkeypatch: object) -> None:
    source = raw_regression(("WARN:low_view_diversity",))
    calls: dict[str, object] = {}

    def legacy(paths: object, **kwargs: object) -> RawRegressionStub:
        del paths
        calls.update(kwargs)
        return source

    monkeypatch.setattr(authorized_image_fit, "_legacy_regress_smplx_from_images", legacy)
    monkeypatch.setattr(authorized_image_fit, "load_default_model", lambda: InferenceModelStub())
    monkeypatch.setattr(authorized_image_fit, "finalize_regression_diagnostics", lambda value: value)

    result = authorized_image_fit.regress_smplx_from_images([Path("x.png")])

    assert calls["refine_with_measurements"] is False
    assert result.measurement_fit.refinement_receipt.decision == "abstain"
    np.testing.assert_allclose(result.measurement_fit.betas, source.betas)
