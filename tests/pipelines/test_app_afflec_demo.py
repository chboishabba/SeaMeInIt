from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import json
import numpy as np
import pytest

from pipelines.measurement_inference import MeasurementEstimate, MeasurementReport

if "jsonschema" not in sys.modules:
    jsonschema_stub = ModuleType("jsonschema")

    class _ValidationError(Exception):
        """Placeholder validation error for jsonschema-free testing."""

    jsonschema_stub.Draft202012Validator = object
    jsonschema_stub.ValidationError = _ValidationError
    sys.modules["jsonschema"] = jsonschema_stub

from smii.pipelines.fit_from_measurements import FitResult


def _fake_result(measurements: dict[str, float]) -> FitResult:
    estimates = tuple(
        MeasurementEstimate(
            name=name,
            value=float(value),
            source="measured",
            confidence=1.0,
            variance=0.0,
        )
        for name, value in measurements.items()
    )
    return FitResult(
        betas=np.zeros(10, dtype=float),
        scale=1.0,
        translation=np.array([0.1, 0.2, 0.3], dtype=float),
        residual=0.0,
        measurements_used=tuple(sorted(measurements)),
        measurement_report=MeasurementReport(estimates=estimates, coverage=1.0),
    )


def test_afflec_demo_announces_plot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from smii import app

    called: dict[str, object] = {}

    monkeypatch.setattr(app.importlib_util, "find_spec", lambda name: object())

    measurements = {"height": 170.0, "chest_circumference": 95.5}

    def fake_regress(images, **kwargs):
        called["images"] = tuple(images)
        called["fit_kwargs"] = kwargs
        fit_result = _fake_result(measurements)
        return SimpleNamespace(
            measurement_fit=fit_result,
            trust_level="coarse",
            consistency_status="WARN",
            consistency_flags=("detector:bbox_coarse_fallback",),
            fit_mode="image_regression_plus_measurement_refinement",
            measurement_source="raw_image_features",
            measurements=measurements,
            to_dict=lambda: {"betas": [0.0] * 10},
        )

    def fake_save(result: FitResult, output_path: Path) -> None:
        called["save_path"] = output_path
        called["saved_result"] = result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict()), encoding="utf-8")

    def fake_save_regression_json(result, output_path: Path) -> None:
        called["raw_path"] = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")

    def fake_save_diagnostics(result, output_path: Path) -> None:
        called["diagnostics_path"] = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")

    def fake_build_diagnostics(result) -> dict[str, object]:
        return {"summary": {"consistency_status": "WARN"}}

    def fake_plot(result: FitResult, output_dir: Path) -> Path:
        called["plot_dir"] = output_dir
        plot_path = output_dir / "measurement_report.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_text("plot", encoding="utf-8")
        return plot_path

    class _DummyTensor:
        def __init__(self, array: np.ndarray) -> None:
            self._array = np.asarray(array)

        def detach(self) -> "_DummyTensor":
            return self

        def cpu(self) -> "_DummyTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._array

    class DummyBodyModel:
        def __init__(self, **kwargs: object) -> None:
            called.setdefault("body_model_kwargs", []).append(kwargs)
            self.model = SimpleNamespace(faces=np.array([[0, 1, 2]], dtype=np.int32))
            self._base_vertices = np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            )
            dims = {
                "betas": int(kwargs.get("num_betas", 10)),
                "expression": int(kwargs.get("num_expression_coeffs", 10)),
                "global_orient": 3,
                "body_pose": 63,
                "left_hand_pose": 45,
                "right_hand_pose": 45,
                "jaw_pose": 3,
                "leye_pose": 3,
                "reye_pose": 3,
                "transl": 3,
            }
            self._parameters: dict[str, np.ndarray] = {
                name: np.zeros((1, dim), dtype=np.float32) for name, dim in dims.items()
            }

        def set_shape(self, betas: np.ndarray) -> None:
            array = np.asarray(betas, dtype=np.float32)
            self._parameters["betas"] = array.reshape(1, -1)
            called["betas"] = self._parameters["betas"]

        def set_body_pose(
            self,
            body_pose: np.ndarray | None = None,
            global_orient: np.ndarray | None = None,
            transl: np.ndarray | None = None,
        ) -> None:
            if transl is not None:
                self._parameters["transl"] = np.asarray(transl, dtype=np.float32).reshape(1, -1)
                called["transl"] = self._parameters["transl"]

        def set_parameters(self, updates: dict[str, np.ndarray]) -> None:
            for name, value in updates.items():
                self._parameters[name] = np.asarray(value, dtype=np.float32).reshape(1, -1)
            called.setdefault("set_parameters", []).append({key: val.copy() for key, val in self._parameters.items()})

        def parameters(self, as_numpy: bool = False) -> dict[str, np.ndarray]:
            return {name: value.copy() for name, value in self._parameters.items()}

        def vertices(self) -> _DummyTensor:
            called["vertices_called"] = called.get("vertices_called", 0) + 1
            transl = self._parameters.get("transl", np.zeros((1, 3), dtype=np.float32))
            vertices = self._base_vertices + transl.reshape(1, 3)
            return _DummyTensor(vertices[np.newaxis, ...])

    monkeypatch.setattr("smii.pipelines.regress_smplx_from_images", fake_regress)
    monkeypatch.setattr("smii.pipelines.save_regression_json", fake_save_regression_json)
    monkeypatch.setattr("smii.pipelines.save_fit_diagnostics_report", fake_save_diagnostics)
    monkeypatch.setattr("smii.pipelines.build_fit_diagnostics_report", fake_build_diagnostics)
    monkeypatch.setattr("smii.pipelines.fit_from_measurements.save_fit", fake_save)
    monkeypatch.setattr("smii.pipelines.fit_from_measurements.plot_measurement_report", fake_plot)

    dummy_avatar_module = ModuleType("avatar_model")
    dummy_avatar_module.BodyModel = DummyBodyModel
    monkeypatch.setitem(sys.modules, "avatar_model", dummy_avatar_module)

    class DummyTrimesh:
        def __init__(self, *, vertices, faces, process: bool) -> None:
            called["trimesh_vertices"] = np.asarray(vertices)
            called["trimesh_faces"] = np.asarray(faces)
            called["trimesh_process"] = process
            self.is_watertight = True

    dummy_trimesh_module = ModuleType("trimesh")
    dummy_trimesh_module.Trimesh = DummyTrimesh
    monkeypatch.setitem(sys.modules, "trimesh", dummy_trimesh_module)

    images = [tmp_path / "front.jpg"]
    # The demo expands and validates image inputs; create a stub file so the
    # mocked fitter is exercised instead of failing early.
    images[0].write_bytes(b"stub")
    output_dir = tmp_path / "afflec"

    result_path = app.run_afflec_fixture_demo(images=images, output_dir=output_dir)

    assert result_path == output_dir / "afflec_measurement_fit.json"
    assert called["images"] == tuple(images)
    assert called["fit_kwargs"] == {"detector": "bbox", "refine_with_measurements": True}
    assert called["save_path"] == result_path
    assert called["raw_path"] == output_dir / "afflec_raw_regression.json"
    assert called["diagnostics_path"] == output_dir / "afflec_fit_diagnostics.json"
    assert called["plot_dir"] == output_dir
    assert (output_dir / "measurement_report.png").exists()
    np.testing.assert_array_equal(called["betas"], np.zeros((1, 10), dtype=np.float32))
    assert len(called["body_model_kwargs"]) == 2
    assert called["body_model_kwargs"][0]["num_betas"] == 10
    assert called["body_model_kwargs"][1]["num_betas"] == 10
    assert called["vertices_called"] == 2

    mesh_path = output_dir / "afflec_body.npz"
    params_path = output_dir / "afflec_smplx_params.json"
    assert params_path.exists()
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    assert payload["model_type"] == "smplx"
    assert payload["gender"] == "neutral"
    assert payload["parameters"]
    assert pytest.approx(payload["scale"], rel=1e-6) == 1.0
    assert payload["transl"] == pytest.approx([0.1, 0.2, 0.3])
    assert payload["images_used"] == [str(images[0])]
    assert payload["detector"] == "bbox"
    assert payload["fit_mode"] == "image_regression_plus_measurement_refinement"
    assert payload["consistency_status"] == "WARN"
    assert mesh_path.exists()
    archive = np.load(mesh_path)
    try:
        np.testing.assert_allclose(
            archive["vertices"],
            np.array(
                [[0.1, 0.2, 0.3], [1.1, 0.2, 0.3], [0.1, 1.2, 0.3]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(archive["faces"], np.array([[0, 1, 2]], dtype=np.int32))
    finally:
        archive.close()

    np.testing.assert_allclose(
        called["trimesh_vertices"],
        np.array(
            [[0.1, 0.2, 0.3], [1.1, 0.2, 0.3], [0.1, 1.2, 0.3]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(called["trimesh_faces"], np.array([[0, 1, 2]], dtype=np.int32))
    assert called["trimesh_process"] is False

    saved_payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved_payload["provenance"]["images_used"] == [str(images[0])]
    assert saved_payload["provenance"]["detector"] == "bbox"
    assert saved_payload["raw_measurements"]["height"] == pytest.approx(170.0)
    assert saved_payload["raw_measurements"]["chest_circumference"] == pytest.approx(95.5)
    assert saved_payload["consistency_status"] == "WARN"

    out = capsys.readouterr().out
    assert f"Generated measurement report plot at {output_dir / 'measurement_report.png'}" in out
    assert f"Saved SMPL-X parameter payload to {params_path}" in out
    assert f"Saved fitted body mesh to {mesh_path}" in out
