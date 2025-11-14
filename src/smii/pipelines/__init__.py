"""Pipeline utilities for fitting SMPL-X models."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    "AfflecImageMeasurementExtractor",
    "FitResult",
    "ICPSettings",
    "MeasurementExtractionError",
    "RegistrationResult",
    "UndersuitGenerationResult",
    "UndersuitMesh",
    "UndersuitPipeline",
    "create_body_mesh",
    "create_parametric_mesh",
    "extract_measurements_from_afflec_images",
    "fit_smplx_from_images",
    "fit_scan_to_smplx",
    "fit_smplx_from_measurements",
    "export_hard_layer_main",
    "export_patterns_main",
    "generate_hard_shell",
    "generate_undersuit",
    "load_body_record",
    "run_clearance",
    "load_mesh",
    "load_transforms",
]

_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    "AfflecImageMeasurementExtractor": (
        "smii.pipelines.fit_from_images",
        "AfflecImageMeasurementExtractor",
    ),
    "MeasurementExtractionError": ("smii.pipelines.fit_from_images", "MeasurementExtractionError"),
    "extract_measurements_from_afflec_images": (
        "smii.pipelines.fit_from_images",
        "extract_measurements_from_afflec_images",
    ),
    "fit_smplx_from_images": (
        "smii.pipelines.fit_from_images",
        "fit_smplx_from_images",
    ),
    "create_body_mesh": ("smii.pipelines.fit_from_measurements", "create_body_mesh"),
    "FitResult": ("smii.pipelines.fit_from_measurements", "FitResult"),
    "fit_smplx_from_measurements": (
        "smii.pipelines.fit_from_measurements",
        "fit_smplx_from_measurements",
    ),
    "ICPSettings": ("smii.pipelines.fit_from_scan", "ICPSettings"),
    "RegistrationResult": ("smii.pipelines.fit_from_scan", "RegistrationResult"),
    "create_parametric_mesh": ("smii.pipelines.fit_from_scan", "create_parametric_mesh"),
    "fit_scan_to_smplx": ("smii.pipelines.fit_from_scan", "fit_scan_to_smplx"),
    "UndersuitGenerationResult": ("smii.pipelines.undersuit", "UndersuitGenerationResult"),
    "UndersuitMesh": ("smii.pipelines.undersuit", "UndersuitMesh"),
    "UndersuitPipeline": ("smii.pipelines.undersuit", "UndersuitPipeline"),
    "export_patterns_main": ("smii.pipelines.export_patterns", "main"),
    "generate_undersuit": ("smii.pipelines.generate_undersuit", "generate_undersuit"),
    "load_body_record": ("smii.pipelines.generate_undersuit", "load_body_record"),
    "run_clearance": ("smii.pipelines.analyze_clearance", "run_clearance"),
    "load_mesh": ("smii.pipelines.analyze_clearance", "load_mesh"),
    "load_transforms": ("smii.pipelines.analyze_clearance", "load_transforms"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover - simple delegation
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError as exc:  # pragma: no cover - attribute errors fall through
        raise AttributeError(f"module 'smii.pipelines' has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - interactive helper
    return sorted(__all__)
