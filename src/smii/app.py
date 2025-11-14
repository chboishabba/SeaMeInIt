"""High-level command helpers for running SeaMeInIt demos."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "InteractiveSession",
    "launch_interactive_session",
    "run_afflec_fixture_demo",
    "run_image_fitting_pipeline",
]


_SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".avif",
}


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
    fixture_dir = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "afflec"
    return sorted(fixture_dir.glob("*.pgm"))


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
                if item.is_file() and item.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES
            ]
            expanded.extend(files)
        else:
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


def run_afflec_fixture_demo(
    *,
    images: Iterable[Path] | None = None,
    output_dir: Path | None = None,
    model_assets: Path | None = None,
    model_backend: str = "smplx",
) -> Path:
    """Fit SMPL family parameters from the bundled Afflec fixture images."""

    if importlib_util.find_spec("jsonschema") is None:
        raise ModuleNotFoundError(
            "jsonschema is required for Afflec measurement fitting. Install the 'jsonschema' dependency."
        )

    from smii.pipelines import fit_smplx_from_images
    from smii.pipelines.fit_from_measurements import (
        create_body_mesh,
        plot_measurement_report,
        save_fit,
    )

    image_paths = list(images or _default_afflec_images())
    if not image_paths:
        raise FileNotFoundError(
            "No Afflec fixture images were found. Provide --images to supply custom data."
        )

    print("Regressing SMPL-X parameters from Afflec imagery...")
    result = fit_smplx_from_images(image_paths)
    target_dir = output_dir or Path("outputs/afflec_demo")
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / "afflec_measurement_fit.json"
    save_fit(result, output_path)
    print(f"Saved fitted parameters to {output_path}")

    mesh_path = target_dir / "afflec_body.npz"
    asset_root = Path(model_assets) if model_assets is not None else Path("assets") / model_backend
    try:
        vertices, faces = create_body_mesh(result, model_path=asset_root)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "SMPL-compatible assets are required to generate the fitted body mesh. "
            f"Provision the '{model_backend}' bundle with `python tools/download_smplx.py --model {model_backend}` "
            "and re-run the command (use --assets-root to supply a custom path)."
        ) from exc
    np.savez(mesh_path, vertices=vertices, faces=faces)
    print(f"Saved fitted body mesh to {mesh_path}")
    plot_path = plot_measurement_report(result, target_dir)
    if plot_path is not None:
        print(f"Generated measurement report plot at {plot_path}")
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
        fit_smplx_from_images,
        save_regression_json,
        save_regression_mesh,
    )

    print(f"Fitting SMPL-X parameters from {len(image_list)} image(s) using {detector}...")
    result = fit_smplx_from_images(
        image_list,
        detector=detector,
        refine_with_measurements=refine_with_measurements,
    )

    json_path = target_dir / f"{subject}_smplx_params.json"
    save_regression_json(result, json_path)
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
        help="Process the Afflec measurement fixtures bundled with the repo",
    )
    afflec.add_argument(
        "--images",
        type=Path,
        nargs="*",
        help="Custom Afflec-annotated images to process",
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
        choices=("mediapipe",),
        default="mediapipe",
        help="Keypoint detector used for landmark extraction.",
    )
    fit_from_images.add_argument(
        "--skip-measurement-refinement",
        action="store_true",
        help="Disable measurement-model refinement of the regressed betas.",
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
            refine_with_measurements=not args.skip_measurement_refinement,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 0
