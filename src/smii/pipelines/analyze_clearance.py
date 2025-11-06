"""CLI entry point for hard-shell clearance analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from suit_hard import Mesh, analyze_clearance, interpolate_poses

OUTPUT_ROOT = Path("outputs/clearance")

__all__ = ["load_mesh", "load_transforms", "run_clearance", "main"]


def load_mesh(path: Path) -> Mesh:
    """Load a triangle mesh from JSON or NPZ archives."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        vertices = np.asarray(payload.get("vertices"), dtype=float)
        faces = np.asarray(payload.get("faces"), dtype=int)
    elif suffix == ".npz":
        archive = np.load(path)
        vertices = np.asarray(archive["vertices"], dtype=float)
        faces = np.asarray(archive["faces"], dtype=int)
    else:
        raise ValueError(f"Unsupported mesh format: {path.suffix}")
    return Mesh(vertices=vertices, faces=faces)


def load_transforms(path: Path | None) -> Sequence[np.ndarray]:
    """Load pose transforms from a JSON document."""

    if path is None:
        return [np.eye(4)]
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    transforms = []
    for entry in payload:
        matrix = np.asarray(entry, dtype=float)
        if matrix.shape != (4, 4) and matrix.shape != (3, 3):
            raise ValueError("Pose transforms must be 3x3 or 4x4 matrices")
        transforms.append(matrix)
    if not transforms:
        raise ValueError("At least one pose transform must be provided")
    return transforms


def _ensure_output_dir(base_dir: Path | None, shell_path: Path, target_path: Path) -> Path:
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    derived = OUTPUT_ROOT / f"{shell_path.stem}_vs_{target_path.stem}"
    derived.mkdir(parents=True, exist_ok=True)
    return derived


def _write_json_report(path: Path, result_dict: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(result_dict, stream, indent=2)


def _write_text_report(path: Path, result_dict: dict[str, object]) -> None:
    lines = [
        "Hard Shell Clearance Report",
        "============================",
        f"Worst penetration: {result_dict['worst_penetration']:.4f} m",
        f"Best clearance: {result_dict['best_clearance']:.4f} m",
        f"Recommended offset: {result_dict['recommended_offset']}",
        "",
        "Pose Metrics:",
    ]
    for pose in result_dict["poses"]:
        lines.append(
            f"  Pose {pose['index']}: min_clearance={pose['min_clearance']:.4f} m, "
            f"max_penetration={pose['max_penetration']:.4f} m, contacts={pose['contact_count']}"
        )
    with path.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def _write_pose_table(path: Path, result_dict: dict[str, object]) -> None:
    header = "index,min_clearance,max_penetration,contact_count"
    rows = [header]
    for pose in result_dict["poses"]:
        rows.append(
            f"{pose['index']},{pose['min_clearance']:.6f},{pose['max_penetration']:.6f},{pose['contact_count']}"
        )
    with path.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(rows))


def run_clearance(
    shell_path: Path,
    target_path: Path,
    *,
    pose_path: Path | None = None,
    output_dir: Path | None = None,
    samples_per_segment: int = 0,
) -> Path:
    shell = load_mesh(shell_path)
    target = load_mesh(target_path)
    transforms = load_transforms(pose_path)
    sampled_transforms = interpolate_poses(transforms, samples_per_segment=samples_per_segment)
    result = analyze_clearance(shell, target, sampled_transforms)
    payload = result.to_dict()

    target_dir = _ensure_output_dir(output_dir, shell_path, target_path)
    _write_json_report(target_dir / "report.json", payload)
    _write_text_report(target_dir / "report.txt", payload)
    _write_pose_table(target_dir / "poses.csv", payload)
    return target_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyse clearance between a shell and target meshes."
    )
    parser.add_argument("shell", type=Path, help="Path to the hard shell mesh (JSON or NPZ).")
    parser.add_argument("target", type=Path, help="Path to the target mesh (JSON or NPZ).")
    parser.add_argument(
        "--poses",
        type=Path,
        help="Optional JSON list of pose transforms (each 4x4 or 3x3 matrix).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Directory to write clearance reports to (default: outputs/clearance/<shell>_vs_<target>).",
    )
    parser.add_argument(
        "--samples-per-segment",
        type=int,
        default=0,
        help="Number of interpolated poses to insert between key transforms.",
    )
    args = parser.parse_args(argv)

    run_clearance(
        args.shell,
        args.target,
        pose_path=args.poses,
        output_dir=args.output,
        samples_per_segment=args.samples_per_segment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
