#!/usr/bin/env python3
"""Inspect and optionally repair fitted SMPL-X body meshes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import trimesh

from smii.meshing import load_body_record


@dataclass(frozen=True)
class ComponentSummary:
    """Lightweight summary describing a connected mesh component."""

    index: int
    vertex_count: int
    face_count: int
    is_watertight: bool


def load_body_mesh(path: Path, *, process: bool = False) -> trimesh.Trimesh:
    """Load a fitted body mesh record into a :class:`trimesh.Trimesh`."""

    if not path.exists():
        raise SystemExit(f"Body record not found: {path}")

    record = load_body_record(path)
    return trimesh.Trimesh(
        vertices=record["vertices"],
        faces=record["faces"],
        process=process,
    )


def split_components(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    """Split a mesh into connected components without filtering by watertightness."""

    if len(mesh.faces) == 0:
        return []

    face_indices = np.arange(len(mesh.faces))
    groups = trimesh.graph.connected_components(mesh.face_adjacency, nodes=face_indices)
    components = [
        mesh.submesh([np.asarray(group, dtype=int)], append=True, repair=False)
        for group in groups
    ]
    components.sort(key=lambda component: (component.area, len(component.faces)), reverse=True)
    return components


def summarise_components(components: Sequence[trimesh.Trimesh]) -> list[ComponentSummary]:
    """Return statistics for each component in order."""

    return [
        ComponentSummary(
            index=index,
            vertex_count=len(component.vertices),
            face_count=len(component.faces),
            is_watertight=component.is_watertight,
        )
        for index, component in enumerate(components)
    ]


def select_body_component(components: Sequence[trimesh.Trimesh]) -> int:
    """Identify the primary body component (largest by face count)."""

    if not components:
        raise ValueError("Body mesh did not contain any components to inspect.")
    return max(
        range(len(components)),
        key=lambda idx: (components[idx].area, len(components[idx].faces)),
    )


def repair_body_component(
    components: Sequence[trimesh.Trimesh], *,
    component_index: int,
) -> tuple[bool, bool]:
    """Run simple repairs on the specified component and report watertightness."""

    target = components[component_index]
    before = target.is_watertight
    trimesh.repair.fill_holes(target)
    trimesh.repair.fix_normals(target, multibody=False)
    after = target.is_watertight
    return before, after


def assemble_components(components: Iterable[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Combine separate mesh components back into a single mesh."""

    components = tuple(components)
    if not components:
        raise ValueError("Cannot assemble an empty component sequence.")
    if len(components) == 1:
        return components[0]
    return trimesh.util.concatenate(components)


def format_summary(title: str, summaries: Sequence[ComponentSummary]) -> str:
    """Render a human-readable component summary."""

    lines = [title]
    for summary in summaries:
        lines.append(
            "  Component {index}: vertices={vertices}, faces={faces}, watertight={watertight}".format(
                index=summary.index,
                vertices=summary.vertex_count,
                faces=summary.face_count,
                watertight=summary.is_watertight,
            )
        )
    return "\n".join(lines)


def default_output_path(body_path: Path) -> Path:
    return body_path.with_name(f"{body_path.stem}_repaired.npz")


def write_body_mesh(mesh: trimesh.Trimesh, path: Path, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(
            f"Output path already exists: {path}. Use --force to overwrite existing files."
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix == ".npz":
        np.savez(
            path,
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32),
        )
    elif suffix == ".obj":
        mesh.export(path, file_type="obj")
    else:
        raise SystemExit("Only .npz and .obj outputs are supported.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a fitted body mesh, reporting connected components and optional "
            "repairs before undersuit generation."
        )
    )
    parser.add_argument(
        "body",
        type=Path,
        help=(
            "Path to the fitted body record (JSON or NumPy .npz) that should be "
            "validated before undersuit generation."
        ),
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Allow trimesh to process geometry when loading (default: disabled).",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help=(
            "Apply a minimal repair (fill holes, fix normals) to the largest component "
            "and write the repaired mesh."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination for the repaired mesh (defaults to <body>_repaired.npz).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mesh = load_body_mesh(args.body, process=args.process)

    components = split_components(mesh)
    summaries = summarise_components(components)
    print(format_summary("Connected components:", summaries))

    if not args.repair:
        return 0

    body_index = select_body_component(components)
    before, after = repair_body_component(components, component_index=body_index)
    status = "unchanged" if before == after else ("fixed" if after else "failed")
    print(
        "\nRepaired component {index} (largest by surface area): watertight {before} -> {after} "
        "[{status}].".format(index=body_index, before=before, after=after, status=status)
    )

    repaired_summaries = summarise_components(components)
    print("\n" + format_summary("After repair:", repaired_summaries))

    output_path = args.output or default_output_path(args.body)
    repaired_mesh = assemble_components(components)
    write_body_mesh(repaired_mesh, output_path, overwrite=args.force)
    print(f"\nWrote repaired mesh to {output_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
