#!/usr/bin/env python3
"""Utility for inspecting fitted body meshes with Trimesh."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a mesh stored as a NumPy archive and open an interactive viewer "
            "via Trimesh."
        )
    )
    parser.add_argument(
        "mesh",
        type=Path,
        help=(
            "Path to a NumPy .npz archive containing 'vertices' and 'faces' arrays. "
            "The Afflec demo writes this to outputs/afflec_demo/afflec_body.npz."
        ),
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help=(
            "Print mesh statistics and exit without launching the interactive viewer."
        ),
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help=(
            "Allow Trimesh to process the geometry (vertex merging, normal fixes, "
            "etc.). By default we preserve the original topology."
        ),
    )
    return parser.parse_args()


def load_mesh(archive_path: Path, process: bool) -> trimesh.Trimesh:
    if not archive_path.exists():
        raise SystemExit(f"Mesh archive not found: {archive_path}")

    data = np.load(archive_path)
    missing = required_keys(data.files, {"vertices", "faces"})
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise SystemExit(
            f"Mesh archive is missing required arrays: {missing_list}. "
            "Expected 'vertices' and 'faces'."
        )

    mesh = trimesh.Trimesh(
        vertices=data["vertices"],
        faces=data["faces"],
        process=process,
    )
    return mesh


def required_keys(existing: Iterable[str], expected: set[str]) -> set[str]:
    return expected.difference(existing)


def describe_mesh(mesh: trimesh.Trimesh) -> str:
    return (
        "Vertices: {vertices}\n"
        "Faces: {faces}\n"
        "Is watertight: {watertight}\n"
        "Components: {components}"
    ).format(
        vertices=len(mesh.vertices),
        faces=len(mesh.faces),
        watertight=mesh.is_watertight,
        components=mesh.body_count,
    )


def main() -> None:
    args = parse_args()
    mesh = load_mesh(args.mesh, process=args.process)

    print(describe_mesh(mesh))

    if args.info_only:
        return

    mesh.show()


if __name__ == "__main__":
    main()
