"""Generate rigid hard-shell layers from a fitted body mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np

ArrayLike = np.ndarray

__all__ = ["ShellGenerationResult", "ShellGenerator", "ShellOptions"]


@dataclass(frozen=True, slots=True)
class ShellGenerationResult:
    """Result of running :class:`ShellGenerator`."""

    vertices: ArrayLike
    faces: ArrayLike
    thickness: ArrayLike
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ShellOptions:
    """Options that control shell generation."""

    default_thickness: float = 0.004
    region_masks: Mapping[str, ArrayLike] | None = None
    enforce_watertight: bool = True

    def iter_masks(self) -> Iterable[tuple[str, ArrayLike]]:
        masks = self.region_masks or {}
        for name, mask in masks.items():
            yield name, np.asarray(mask, dtype=bool)


class ShellGenerator:
    """Inflate a fitted body mesh into a rigid shell."""

    def __init__(self, *, seam_tolerance: float = 1e-5) -> None:
        self.seam_tolerance = float(seam_tolerance)

    def generate(
        self,
        body_output: Mapping[str, ArrayLike] | object,
        *,
        thickness_profile: Mapping[str, float] | Sequence[float] | ArrayLike | float | None = None,
        exclusions: Sequence[str | ArrayLike] | ArrayLike | None = None,
        options: ShellOptions | None = None,
    ) -> ShellGenerationResult:
        """Generate a shell mesh by offsetting vertices along their normals."""

        options = options or ShellOptions()
        vertices, faces = _extract_vertices_and_faces(body_output)

        if options.enforce_watertight and not _is_watertight(faces, tolerance=self.seam_tolerance):
            raise ValueError("Body mesh must be watertight before generating a rigid shell.")

        normals = _vertex_normals(vertices, faces)

        profile, profile_type = _resolve_thickness_profile(
            thickness_profile,
            vertex_count=vertices.shape[0],
            default_thickness=float(options.default_thickness),
            region_masks=options.region_masks or {},
        )

        exclusion_mask = _resolve_exclusions(
            exclusions,
            vertex_count=vertices.shape[0],
            region_masks=options.region_masks or {},
        )

        if np.any(profile < 0.0):
            raise ValueError("Shell thickness values must be non-negative.")

        profile = profile.copy()
        profile[exclusion_mask] = 0.0

        shell_vertices = vertices + normals * profile[:, None]
        surface_area = _surface_area(shell_vertices, faces)

        metadata: MutableMapping[str, object] = {
            "default_thickness": float(options.default_thickness),
            "profile_type": profile_type,
            "excluded_vertices": int(np.count_nonzero(exclusion_mask)),
            "watertight": bool(_is_watertight(faces, tolerance=self.seam_tolerance)),
            "surface_area": float(surface_area),
            "thickness_statistics": {
                "min": float(np.min(profile)) if profile.size else 0.0,
                "max": float(np.max(profile)) if profile.size else 0.0,
                "mean": float(np.mean(profile)) if profile.size else 0.0,
            },
        }

        if exclusion_mask.any():
            metadata["exclusion_ratio"] = float(np.count_nonzero(exclusion_mask) / profile.size)

        return ShellGenerationResult(
            vertices=shell_vertices,
            faces=faces.copy(),
            thickness=profile,
            metadata=dict(metadata),
        )


def _extract_vertices_and_faces(
    body_output: Mapping[str, ArrayLike] | object,
) -> tuple[ArrayLike, ArrayLike]:
    if isinstance(body_output, Mapping):
        vertices = body_output.get("vertices")
        faces = body_output.get("faces")
    else:
        vertices = getattr(body_output, "vertices", None)
        faces = getattr(body_output, "faces", None)

    if vertices is None or faces is None:
        raise TypeError("Body output must expose 'vertices' and 'faces'.")

    vertices_arr = np.asarray(vertices, dtype=float)
    faces_arr = np.asarray(faces, dtype=int)

    if vertices_arr.ndim == 3:
        vertices_arr = vertices_arr[0]
    if faces_arr.ndim == 3:
        faces_arr = faces_arr[0]

    if vertices_arr.ndim != 2 or vertices_arr.shape[1] != 3:
        raise ValueError("Vertices must be an (N, 3) array.")
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
        raise ValueError("Faces must be an (M, 3) array.")

    return vertices_arr, faces_arr


def _resolve_thickness_profile(
    profile: Mapping[str, float] | Sequence[float] | ArrayLike | float | None,
    *,
    vertex_count: int,
    default_thickness: float,
    region_masks: Mapping[str, ArrayLike],
) -> tuple[np.ndarray, str]:
    base = np.full(vertex_count, float(default_thickness), dtype=float)
    if profile is None:
        return base, "default"
    if isinstance(profile, (int, float)):
        base.fill(float(profile))
        return base, "uniform"
    if isinstance(profile, Mapping):
        for name, value in profile.items():
            if name not in region_masks:
                raise ValueError(f"Thickness profile references unknown region '{name}'.")
            thickness = float(value)
            if thickness < 0.0:
                raise ValueError("Thickness profile values must be non-negative.")
            mask = _validate_mask(region_masks[name], vertex_count)
            base[mask] = thickness
        return base, "regional"

    profile_arr = np.asarray(profile, dtype=float)
    if profile_arr.shape != (vertex_count,):
        raise ValueError("Per-vertex thickness profile must match the vertex count.")
    return profile_arr, "per_vertex"


def _resolve_exclusions(
    exclusions: Sequence[str | ArrayLike] | ArrayLike | None,
    *,
    vertex_count: int,
    region_masks: Mapping[str, ArrayLike],
) -> np.ndarray:
    if exclusions is None:
        return np.zeros(vertex_count, dtype=bool)

    if isinstance(exclusions, np.ndarray):
        mask = np.asarray(exclusions, dtype=bool)
        if mask.shape != (vertex_count,):
            raise ValueError("Exclusion mask must match vertex count.")
        return mask

    combined = np.zeros(vertex_count, dtype=bool)
    items: Iterable[str | ArrayLike]
    if isinstance(exclusions, (str, bytes)):
        items = [exclusions]
    elif isinstance(exclusions, Mapping):
        items = exclusions.values()
    elif isinstance(exclusions, Sequence):
        items = exclusions
    else:
        items = [exclusions]

    for item in items:
        if isinstance(item, str):
            if item not in region_masks:
                raise ValueError(f"Exclusion references unknown region '{item}'.")
            mask = _validate_mask(region_masks[item], vertex_count)
            combined |= mask
        else:
            mask = np.asarray(item, dtype=bool)
            if mask.shape != (vertex_count,):
                raise ValueError("Exclusion mask must match vertex count.")
            combined |= mask
    return combined


def _validate_mask(mask: ArrayLike, vertex_count: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != (vertex_count,):
        raise ValueError("Region mask must match vertex count.")
    return arr


def _vertex_normals(vertices: ArrayLike, faces: ArrayLike) -> np.ndarray:
    normals = np.zeros_like(vertices)
    tri_vertices = vertices[faces]
    tri_normals = np.cross(
        tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0]
    )
    for idx, face in enumerate(faces):
        normals[face] += tri_normals[idx]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return normals / norms


def _surface_area(vertices: ArrayLike, faces: ArrayLike) -> float:
    tri_vertices = vertices[faces]
    cross = np.cross(
        tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0]
    )
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def _is_watertight(faces: ArrayLike, *, tolerance: float) -> bool:
    edge_counts: dict[tuple[int, int], int] = {}
    for tri in faces:
        for idx in range(3):
            a = int(tri[idx])
            b = int(tri[(idx + 1) % 3])
            key = tuple(sorted((a, b)))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    return all(abs(count - 2) <= tolerance for count in edge_counts.values())
