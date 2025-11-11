"""High-level command helpers for running SeaMeInIt demos."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "InteractiveSession",
    "launch_interactive_session",
    "run_afflec_fixture_demo",
]


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
    poses = poses if poses is not None else _prompt_optional_path("Optional pose JSON (leave blank for identity): ")
    output = output if output is not None else _prompt_optional_path(
        "Output directory (leave blank for auto-generated): "
    )
    samples = samples if samples is not None else _prompt_int(
        "Interpolated samples per keyframe segment", 0
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


def run_afflec_fixture_demo(
    *,
    images: Iterable[Path] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Fit SMPL-X parameters from the bundled Afflec fixture images."""

    if importlib_util.find_spec("jsonschema") is None:
        raise ModuleNotFoundError(
            "jsonschema is required for Afflec measurement fitting. Install the 'jsonschema' dependency."
        )

    from smii.pipelines import extract_measurements_from_afflec_images, fit_smplx_from_measurements
    from smii.pipelines.fit_from_measurements import save_fit

    image_paths = list(images or _default_afflec_images())
    if not image_paths:
        raise FileNotFoundError(
            "No Afflec fixture images were found. Provide --images to supply custom data."
        )

    print("Extracting measurements from Afflec imagery...")
    measurements = extract_measurements_from_afflec_images(image_paths)
    print(f"Recovered {len(measurements)} measurements.")

    result = fit_smplx_from_measurements(measurements)
    target_dir = output_dir or Path("outputs/afflec_demo")
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / "afflec_measurement_fit.json"
    save_fit(result, output_path)
    print(f"Saved fitted parameters to {output_path}")
    return output_path


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
        run_afflec_fixture_demo(images=args.images, output_dir=args.output)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 0


