"""High-level command helpers for running SeaMeInIt demos."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Iterable, Sequence

import json
import numpy as np
import shutil

from smii.meshing import repair_body_mesh_for_export

__all__ = [
    "InteractiveSession",
    "launch_interactive_session",
    "run_afflec_fixture_demo",
    "run_image_fitting_pipeline",
]


def _pillow_supports_avif() -> bool:
    try:
        from PIL import features  # type: ignore
    except Exception:
        return False
    try:
        return bool(features.check("avif"))
    except Exception:
        return False


_SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}
if _pillow_supports_avif():
    _SUPPORTED_IMAGE_SUFFIXES.add(".avif")


@dataclass(frozen=True)
class InteractiveSession:
    """Container describing a user-driven clearance simulation run."""

    shell: Path
    target: Path
    poses: Path | None
    output_dir: Path | None
    samples_per_segment: int


def _prompt_path(prompt: str) -> Path:
    response = input(prompt).strip()
    return Path(response)


def _prompt_optional_path(prompt: str) -> Path | None:
    response = input(prompt).strip()
    return Path(response) if response else None


def _prompt_int(prompt: str, default: int) -> int:
    response = input(f"{prompt} [{default}]: ").strip()
    if not response:
        return default
    return int(response)


def _collect_interactive_inputs(
    *,
    shell: Path | None = None,
    target: Path | None = None,
    poses: Path | None = None,
    output: Path | None = None,
    samples: int | None = None,
) -> InteractiveSession:
    print("=== Hard-shell clearance interactive run ===")
    shell = shell or _prompt_path("Path to shell mesh (JSON/NPZ): ")
    target = target or _prompt_path("Path to target mesh (JSON/NPZ): ")
    poses = (
        poses
        if poses is not None
        else _prompt_optional_path("Optional pose JSON (leave blank for identity): ")
    )
    output = (
        output
        if output is not None
        else _prompt_optional_path("Output directory (leave blank for auto-generated): ")
    )
    samples = (
        samples
        if samples is not None
        else _prompt_int("Interpolated samples per keyframe segment", 0)
    )
    return InteractiveSession(
        shell=shell,
        target=target,
        poses=poses,
        output_dir=output,
        samples_per_segment=samples,
    )


def launch_interactive_session(
    *,
    shell: Path | None = None,
    target: Path | None = None,
    poses: Path | None = None,
    output: Path | None = None,
    samples: int | None = None,
) -> Path:
    """Run the clearance pipeline with interactive fallbacks for missing inputs."""

    from smii.pipelines import run_clearance

    session = _collect_interactive_inputs(
        shell=shell,
        target=target,
        poses=poses,
        output=output,
        samples=samples,
    )
    print("Running clearance analysis...")
    result_dir = run_clearance(
        session.shell,
        session.target,
        pose_path=session.poses,
        output_dir=session.output_dir,
        samples_per_segment=session.samples_per_segment,
    )
    print(f"Clearance reports written to {result_dir}")
    return result_dir


def _default_afflec_images() -> list[Path]:
    """Return the bundled Ben Afflec fixture captures used for smoke tests."""

    fixture_dir = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "afflec"
    return sorted(
        path
        for path in fixture_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES
    )


def _expand_image_inputs(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for candidate in paths:
        path = Path(candidate)
        if not path.exists():
            raise FileNotFoundError(f"Image input {path} does not exist.")
        if path.is_dir():
            files = [
                item
                for item in sorted(path.rglob("*"))
                if item.is_file()
                and item.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES
            ]
            expanded.extend(files)
        else:
            suffix = path.suffix.lower()
            if suffix == ".avif" and ".avif" not in _SUPPORTED_IMAGE_SUFFIXES:
                print(f"Skipping {path}: AVIF support not available in this environment.")
                continue
            if suffix in _SUPPORTED_IMAGE_SUFFIXES:
                expanded.append(path)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in expanded:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _infer_subject_id(images: Sequence[Path]) -> str:
    for path in images:
        if path.is_file() and path.stem:
            return path.stem
        if path.parent.stem:
            return path.parent.stem
    return "image_fit"


def _fit_payload_metadata(
    *,
    images: Sequence[Path],
    detector: str,
    fit_mode: str,
    measurement_source: str,
    refinement_applied: bool,
    trust_level: str,
    consistency_status: str,
    consistency_flags: Sequence[str],
    diagnostics_path: Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "images_used": [str(path) for path in images],
        "detector": detector,
        "fit_mode": fit_mode,
        "measurement_source": measurement_source,
        "refinement_applied": bool(refinement_applied),
        "trust_level": trust_level,
        "consistency_status": consistency_status,
        "consistency_flags": list(consistency_flags),
    }
    if diagnostics_path is not None:
        payload["diagnostics_path"] = str(diagnostics_path)
    return payload


def _merge_payload_metadata(payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    merged = dict(payload)
    merged.update(metadata)
    return merged


def _raw_regression_parameter_payload(
    regression: Any,
    *,
    model_type: str,
    gender: str,
) -> dict[str, Any]:
    from smii.pipelines.fit_from_measurements import serialise_smplx_parameters

    parameters: dict[str, np.ndarray] = {
        "betas": np.asarray(regression.betas, dtype=np.float32).reshape(1, -1),
        "body_pose": np.asarray(regression.body_pose, dtype=np.float32).reshape(1, -1),
        "global_orient": np.asarray(regression.global_orient, dtype=np.float32).reshape(1, -1),
        "transl": np.asarray(regression.transl, dtype=np.float32).reshape(1, -1),
    }
    for name in ("expression", "jaw_pose", "left_hand_pose", "right_hand_pose"):
        value = getattr(regression, name, None)
        if value is None:
            continue
        parameters[name] = np.asarray(value, dtype=np.float32).reshape(1, -1)

    return serialise_smplx_parameters(
        parameters,
        model_type=model_type,
        gender=gender,
        scale=1.0,
    )


def _enforce_fit_quality(
    *,
    detector: str,
    trust_level: str,
    consistency_status: str,
    consistency_flags: Sequence[str],
    require_high_trust_detector: bool,
    fail_on_consistency_errors: bool,
) -> None:
    if require_high_trust_detector and detector != "mediapipe":
        raise ValueError(
            "High-trust detector mode requires --detector mediapipe; bbox is only a coarse fallback."
        )
    if fail_on_consistency_errors and consistency_status == "FAIL":
        flags = ", ".join(consistency_flags) if consistency_flags else "unknown"
        raise ValueError(f"Image-fit consistency checks failed: {flags}")


def run_afflec_fixture_demo(
    *,
    images: Iterable[Path] | None = None,
    output_dir: Path | None = None,
    model_assets: Path | None = None,
    model_backend: str = "smplx",
    force: bool = False,
    clean_output: bool = False,
    detector: str = "bbox",
    fit_mode: str = "auto",
    refine_with_measurements: bool = True,
    require_high_trust_detector: bool = False,
    fail_on_consistency_errors: bool = False,
) -> Path:
    """Fit SMPL family parameters from the tongue-in-cheek Ben Afflec fixtures."""

    if importlib_util.find_spec("jsonschema") is None:
        raise ModuleNotFoundError(
            "jsonschema is required for Afflec measurement fitting. Install the 'jsonschema' dependency."
        )

    from smii.pipelines import (
        build_fit_diagnostics_report,
        regress_smplx_from_images,
        save_image_fit_observations,
        save_fit_diagnostics_report,
        save_regression_json,
    )
    from smii.pipelines.fit_from_measurements import (
        BodyMeshOutput,
        create_body_mesh,
        fit_smplx_from_measurements,
        generate_vertices_from_smplx_parameters,
        load_smplx_parameter_payload,
        plot_measurement_report,
        save_fit,
    )
    try:
        import trimesh  # type: ignore
    except Exception:
        trimesh = None  # type: ignore[assignment]

    image_paths = list(images or _default_afflec_images())
    # Expand directories to file lists to avoid silent no-ops
    if images is not None:
        image_paths = _expand_image_inputs(image_paths)
    if not image_paths:
        raise FileNotFoundError(
            "No Afflec fixture images were found. Provide --images to supply custom data."
        )

    target_dir = output_dir or Path("outputs/afflec_demo")
    if clean_output and target_dir.exists():
        print(f"Cleaning output directory {target_dir} ...")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Using Afflec images:")
    for path in image_paths:
        print(f"  - {path}")
    print(f"Detector: {detector}")

    params_path = target_dir / "afflec_smplx_params.json"
    mesh_path = target_dir / "afflec_body.npz"
    if params_path.exists() or mesh_path.exists():
        print(
            "Existing Afflec artifacts detected in the output directory. "
            "Pass --force to discard them or --clean-output to start fresh."
        )
        if force and target_dir.exists():
            print(f"--force supplied; deleting existing contents in {target_dir}")
            shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing new artifacts to {target_dir}")
    print("Regressing SMPL-X parameters from the Ben Afflec fixtures...")
    asset_root = Path(model_assets) if model_assets is not None else Path("assets") / model_backend
    regression = regress_smplx_from_images(
        image_paths,
        detector=detector,
        refine_with_measurements=refine_with_measurements,
        fit_mode=fit_mode,
        model_path=asset_root,
        model_type=model_backend,
    )

    _enforce_fit_quality(
        detector=detector,
        trust_level=regression.trust_level,
        consistency_status=regression.consistency_status,
        consistency_flags=regression.consistency_flags,
        require_high_trust_detector=require_high_trust_detector,
        fail_on_consistency_errors=fail_on_consistency_errors,
    )

    diagnostics_report = build_fit_diagnostics_report(regression)
    diagnostics_path = target_dir / "afflec_fit_diagnostics.json"
    save_fit_diagnostics_report(regression, diagnostics_path)
    print(f"Saved image-fit diagnostics to {diagnostics_path}")

    if regression.observations:
        observations_path = target_dir / "afflec_observations.json"
        save_image_fit_observations(regression.observations, observations_path)
        print(f"Saved image-fit observations to {observations_path}")

    raw_regression_path = target_dir / "afflec_raw_regression.json"
    save_regression_json(regression, raw_regression_path)
    print(f"Saved raw regression payload to {raw_regression_path}")

    if regression.measurement_fit is not None:
        result = regression.measurement_fit
    else:
        result = fit_smplx_from_measurements(regression.measurements)

    payload_metadata = _fit_payload_metadata(
        images=image_paths,
        detector=detector,
        fit_mode=regression.fit_mode,
        measurement_source=regression.measurement_source,
        refinement_applied=regression.measurement_fit is not None,
        trust_level=regression.trust_level,
        consistency_status=regression.consistency_status,
        consistency_flags=regression.consistency_flags,
        diagnostics_path=diagnostics_path,
    )
    result = replace(
        result,
        provenance=payload_metadata,
        raw_measurements={name: float(value) for name, value in regression.measurements.items()},
        fit_mode=regression.fit_mode,
        trust_level=regression.trust_level,
        consistency_status=regression.consistency_status,
        consistency_flags=regression.consistency_flags,
        diagnostics=diagnostics_report["summary"],
    )

    output_path = target_dir / "afflec_measurement_fit.json"
    save_fit(result, output_path)
    print(f"Saved fitted parameters to {output_path}")

    mesh_output: tuple[np.ndarray, np.ndarray] | BodyMeshOutput
    parameters_payload: dict[str, Any] | None = None
    try:
        if refine_with_measurements:
            mesh_output = create_body_mesh(
                result,
                model_path=asset_root,
                model_type=model_backend,
                return_parameters=True,
            )
            if not isinstance(mesh_output, BodyMeshOutput):  # pragma: no cover - defensive
                vertices, faces = mesh_output  # type: ignore[misc]
            else:
                parameters_payload = mesh_output.parameter_payload()
        else:
            parameters_payload = _raw_regression_parameter_payload(
                regression,
                model_type=model_backend,
                gender="neutral",
            )
            parameters, scale, payload_model_type, gender = load_smplx_parameter_payload(parameters_payload)
            vertices, faces = generate_vertices_from_smplx_parameters(
                parameters,
                scale=scale,
                model_path=asset_root,
                model_type=payload_model_type,
                gender=gender,
            )
            mesh_output = (vertices, faces)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "SMPL-compatible assets are required to generate the fitted body mesh. "
            f"Provision the '{model_backend}' bundle with `python tools/download_smplx.py --model {model_backend}` "
            "and re-run the command (use --assets-root to supply a custom path)."
        ) from exc
    if parameters_payload is not None:
        parameters_payload = _merge_payload_metadata(parameters_payload, payload_metadata)
        with params_path.open("w", encoding="utf-8") as stream:
            json.dump(parameters_payload, stream, indent=2)
        print(f"Saved SMPL-X parameter payload to {params_path}")

    if isinstance(mesh_output, BodyMeshOutput):
        parameters, scale, payload_model_type, gender = load_smplx_parameter_payload(parameters_payload)
        vertices, faces = generate_vertices_from_smplx_parameters(
            parameters,
            scale=scale,
            model_path=asset_root,
            model_type=payload_model_type,
            gender=gender,
        )

        if trimesh is None:
            print(
                "Warning: 'trimesh' is not installed; skipping watertight/repair checks for afflec-demo mesh."
            )
        else:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            if not mesh.is_watertight:
                repaired = repair_body_mesh_for_export(vertices, faces)
                if repaired is not None:
                    vertices, faces = repaired
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            if not mesh.is_watertight:
                raise ValueError(
                    "Fitted SMPL-X body is not watertight. Install PyMeshFix (`pip install pymeshfix`) "
                    "or repair the mesh manually with tools/repair_body_mesh.py."
                )
    else:  # pragma: no cover - fallback path without parameter record
        vertices, faces = mesh_output

    np.savez(mesh_path, vertices=vertices.astype(np.float32), faces=faces.astype(np.int32))
    print(f"Saved fitted body mesh to {mesh_path}")
    plot_path = plot_measurement_report(result, target_dir)
    if plot_path is not None:
        print(f"Generated measurement report plot at {plot_path}")
    if regression.consistency_status != "PASS":
        print(
            "Image-fit diagnostics status: "
            f"{regression.consistency_status} ({', '.join(regression.consistency_flags)})"
        )
    return output_path


def run_image_fitting_pipeline(
    *,
    images: Iterable[Path],
    output_dir: Path | None = None,
    subject_id: str | None = None,
    model_assets: Path | None = None,
    model_backend: str = "smplx",
    gender: str = "neutral",
    detector: str = "mediapipe",
    refine_with_measurements: bool = True,
    fit_mode: str = "auto",
    require_high_trust_detector: bool = False,
    fail_on_consistency_errors: bool = False,
) -> Path:
    """Run the image-based SMPL-X regression pipeline."""

    image_list = _expand_image_inputs(images)
    if not image_list:
        raise FileNotFoundError(
            "No RGB images were found. Provide at least one JPEG/PNG/AVIF path or directory."
        )

    subject = subject_id or _infer_subject_id(image_list)
    target_dir = output_dir or Path("outputs") / subject
    target_dir.mkdir(parents=True, exist_ok=True)

    from smii.pipelines import (
        build_fit_diagnostics_report,
        regress_smplx_from_images,
        save_image_fit_observations,
        save_fit_diagnostics_report,
        save_regression_json,
        save_regression_mesh,
    )

    print(f"Fitting SMPL-X parameters from {len(image_list)} image(s) using {detector}...")
    result = regress_smplx_from_images(
        image_list,
        detector=detector,
        refine_with_measurements=refine_with_measurements,
        fit_mode=fit_mode,
        model_path=Path(model_assets) if model_assets is not None else Path("assets") / model_backend,
        model_type=model_backend,
        gender=gender,
    )

    _enforce_fit_quality(
        detector=detector,
        trust_level=result.trust_level,
        consistency_status=result.consistency_status,
        consistency_flags=result.consistency_flags,
        require_high_trust_detector=require_high_trust_detector,
        fail_on_consistency_errors=fail_on_consistency_errors,
    )

    diagnostics_path = target_dir / f"{subject}_fit_diagnostics.json"
    save_fit_diagnostics_report(result, diagnostics_path)
    print(f"Saved image-fit diagnostics to {diagnostics_path}")

    if result.observations:
        observations_path = target_dir / f"{subject}_observations.json"
        save_image_fit_observations(result.observations, observations_path)
        print(f"Saved image-fit observations to {observations_path}")

    raw_json_path = target_dir / f"{subject}_regression_raw.json"
    save_regression_json(result, raw_json_path)
    print(f"Saved raw regression parameters to {raw_json_path}")

    json_path = target_dir / f"{subject}_smplx_params.json"
    regression_payload = _merge_payload_metadata(
        result.to_dict(),
        _fit_payload_metadata(
            images=image_list,
            detector=detector,
            fit_mode=result.fit_mode,
            measurement_source=result.measurement_source,
            refinement_applied=result.measurement_fit is not None,
            trust_level=result.trust_level,
            consistency_status=result.consistency_status,
            consistency_flags=result.consistency_flags,
            diagnostics_path=diagnostics_path,
        ),
    )
    json_path.write_text(json.dumps(regression_payload, indent=2), encoding="utf-8")
    print(f"Saved regression parameters to {json_path}")

    assets_root = Path(model_assets) if model_assets is not None else Path("assets") / model_backend

    mesh_path = target_dir / f"{subject}_smplx_body.npz"
    try:
        save_regression_mesh(
            result,
            mesh_path,
            model_path=assets_root,
            model_type=model_backend,
            gender=gender,
            use_measurement_refinement=refine_with_measurements,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "SMPL-compatible assets are required to generate the fitted body mesh. "
            f"Provision the '{model_backend}' bundle with `python tools/download_smplx.py --model {model_backend}` "
            "and re-run the command (use --assets-root to supply a custom path)."
        ) from exc

    print(f"Saved watertight mesh to {mesh_path}")
    if result.measurement_fit is not None:
        coverage = result.measurement_fit.measurement_report.coverage
        print(
            "Measurement refinement coverage: "
            f"{coverage:.0%} of manual metrics constrained"
        )
    if result.consistency_status != "PASS":
        print(
            "Image-fit diagnostics status: "
            f"{result.consistency_status} ({', '.join(result.consistency_flags)})"
        )

    return json_path


def build_cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="SeaMeInIt command launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    interactive = subparsers.add_parser(
        "interactive",
        help="Prompt-driven hard-shell clearance analysis",
    )
    interactive.add_argument("--shell", type=Path, help="Path to the hard shell mesh")
    interactive.add_argument("--target", type=Path, help="Path to the target mesh")
    interactive.add_argument("--poses", type=Path, help="Optional JSON pose sequence")
    interactive.add_argument("--output", type=Path, help="Directory for generated reports")
    interactive.add_argument(
        "--samples-per-segment",
        type=int,
        default=None,
        help="Interpolated samples inserted between keyframes",
    )

    afflec = subparsers.add_parser(
        "afflec-demo",
        help="Process the tongue-in-cheek Ben Afflec fixtures bundled with the repo",
    )
    afflec.add_argument(
        "--images",
        type=Path,
        nargs="*",
        help="Custom Ben Afflec-style annotated images to process",
    )
    afflec.add_argument(
        "--output",
        type=Path,
        help="Where to store the fitted parameter JSON",
    )
    afflec.add_argument(
        "--model-backend",
        choices=("smplx", "smplerx"),
        default="smplx",
        help=(
            "Which asset bundle to use when generating meshes. 'smplx' requires a licensed download; "
            "'smplerx' installs the S-Lab 1.0 manifest. Use tools/download_smplx.py to provision the assets."
        ),
    )
    afflec.add_argument(
        "--assets-root",
        type=Path,
        help=(
            "Override the asset directory. Defaults to assets/<model-backend> and must contain a manifest.json."
        ),
    )
    afflec.add_argument(
        "--force",
        action="store_true",
        help="Discard any existing Afflec outputs in the target directory before running.",
    )
    afflec.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the output directory entirely before running (safer than overwriting).",
    )
    afflec.add_argument(
        "--detector",
        choices=("bbox", "mediapipe"),
        default="bbox",
        help="Landmark detector to use. 'bbox' is fast/embedded; 'mediapipe' requires the vision extra.",
    )
    afflec.add_argument(
        "--fit-mode",
        choices=("auto", "heuristic", "reprojection"),
        default="auto",
        help="Image-fitting mode. 'auto' prefers reprojection optimization and falls back to heuristic fitting.",
    )
    afflec.add_argument(
        "--skip-measurement-refinement",
        action="store_true",
        help="Keep the raw image-regressed shape instead of applying measurement-model refinement.",
    )
    afflec.add_argument(
        "--require-high-trust-detector",
        action="store_true",
        help="Fail unless the run uses the higher-trust mediapipe detector path.",
    )
    afflec.add_argument(
        "--fail-on-consistency-errors",
        action="store_true",
        help="Fail when image-fit diagnostics report a FAIL status.",
    )

    fit_from_images = subparsers.add_parser(
        "fit-from-images",
        help="Regress SMPL-X parameters directly from RGB images",
    )
    fit_from_images.add_argument(
        "--images",
        type=Path,
        nargs="+",
        required=True,
        help="One or more image files or directories containing images",
    )
    fit_from_images.add_argument(
        "--output",
        type=Path,
        help="Directory for generated parameter files",
    )
    fit_from_images.add_argument(
        "--subject-id",
        type=str,
        help="Identifier used when naming generated artifacts",
    )
    fit_from_images.add_argument(
        "--model-backend",
        choices=("smplx", "smplerx"),
        default="smplx",
        help="Which asset bundle to use when generating meshes.",
    )
    fit_from_images.add_argument(
        "--assets-root",
        type=Path,
        help="Override the asset directory. Defaults to assets/<model-backend>.",
    )
    fit_from_images.add_argument(
        "--gender",
        choices=("neutral", "male", "female"),
        default="neutral",
        help="Gendered SMPL-X model variant to load.",
    )
    fit_from_images.add_argument(
        "--detector",
        choices=("mediapipe", "bbox"),
        default="mediapipe",
        help="Keypoint detector used for landmark extraction.",
    )
    fit_from_images.add_argument(
        "--fit-mode",
        choices=("auto", "heuristic", "reprojection"),
        default="auto",
        help="Image-fitting mode. 'auto' prefers reprojection optimization and falls back to heuristic fitting.",
    )
    fit_from_images.add_argument(
        "--skip-measurement-refinement",
        action="store_true",
        help="Disable measurement-model refinement of the regressed betas.",
    )
    fit_from_images.add_argument(
        "--require-high-trust-detector",
        action="store_true",
        help="Fail unless the run uses the higher-trust mediapipe detector path.",
    )
    fit_from_images.add_argument(
        "--fail-on-consistency-errors",
        action="store_true",
        help="Fail when image-fit diagnostics report a FAIL status.",
    )

    args = parser.parse_args(argv)

    if args.command == "interactive":
        launch_interactive_session(
            shell=args.shell,
            target=args.target,
            poses=args.poses,
            output=args.output,
            samples=args.samples_per_segment,
        )
        return 0

    if args.command == "afflec-demo":
        run_afflec_fixture_demo(
            images=args.images,
            output_dir=args.output,
            model_backend=args.model_backend,
            model_assets=args.assets_root,
            force=args.force,
            clean_output=args.clean_output,
            detector=args.detector,
            fit_mode=args.fit_mode,
            refine_with_measurements=not args.skip_measurement_refinement,
            require_high_trust_detector=args.require_high_trust_detector,
            fail_on_consistency_errors=args.fail_on_consistency_errors,
        )
        return 0

    if args.command == "fit-from-images":
        run_image_fitting_pipeline(
            images=args.images,
            output_dir=args.output,
            subject_id=args.subject_id,
            model_backend=args.model_backend,
            model_assets=args.assets_root,
            gender=args.gender,
            detector=args.detector,
            fit_mode=args.fit_mode,
            refine_with_measurements=not args.skip_measurement_refinement,
            require_high_trust_detector=args.require_high_trust_detector,
            fail_on_consistency_errors=args.fail_on_consistency_errors,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(build_cli())
