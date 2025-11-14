#!/usr/bin/env python3
"""Utility for inspecting fitted body meshes with Trimesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh

from smii.meshing import load_body_record


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
            "Path to a fitted body record (JSON or NumPy .npz) containing 'vertices' "
            "and 'faces' arrays. The Ben Afflec demo writes this to "
            "outputs/afflec_demo/afflec_body.npz."
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


def load_mesh(path: Path, process: bool) -> trimesh.Trimesh:
    if not path.exists():
        raise SystemExit(f"Mesh archive not found: {path}")

    record = load_body_record(path)
    mesh = trimesh.Trimesh(
        vertices=record["vertices"],
        faces=record["faces"],
        process=process,
    )
    return mesh


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
