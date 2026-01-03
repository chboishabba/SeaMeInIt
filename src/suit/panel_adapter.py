"""Adapters between panel domain objects and serialized payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .panel_model import Panel
from .panel_payload import PanelPayload


@dataclass(frozen=True, slots=True)
class PanelPayloadSource:
    """Mesh data needed to build a panel payload."""

    vertices: Sequence[tuple[float, float, float]]
    faces: Sequence[tuple[int, int, int]] = ()


def panel_to_payload(
    panel: Panel,
    source: PanelPayloadSource,
    *,
    include_surface_metadata: bool = True,
    include_budgets_metadata: bool = False,
) -> PanelPayload:
    """Convert a panel into the payload expected by pattern exporters."""

    if panel.surface_patch is None:
        raise ValueError("Panel surface_patch is required to build a payload.")

    surface = panel.surface_patch
    metadata: dict[str, Any] = dict(surface.metadata) if include_surface_metadata else {}
    if include_budgets_metadata and panel.budgets is not None:
        metadata["budgets"] = {
            "distortion_max": panel.budgets.distortion_max,
            "curvature_min_radius": panel.budgets.curvature_min_radius,
            "turning_max_per_length": panel.budgets.turning_max_per_length,
            "min_feature_size": panel.budgets.min_feature_size,
        }

    if surface.face_indices:
        if not source.faces:
            raise ValueError("Mesh faces are required when face_indices are provided.")
        faces = [source.faces[idx] for idx in surface.face_indices]
        vertex_ids = sorted({idx for face in faces for idx in face})
        vertices = [source.vertices[idx] for idx in vertex_ids]
        remap = {idx: new_idx for new_idx, idx in enumerate(vertex_ids)}
        remapped_faces = [
            (remap[a], remap[b], remap[c])
            for a, b, c in faces
        ]
    elif surface.vertex_indices:
        vertex_ids = list(surface.vertex_indices)
        vertices = [source.vertices[idx] for idx in vertex_ids]
        if len(vertices) < 3:
            raise ValueError("Panel surface_patch must include at least three vertices.")
        remapped_faces = [(0, i, i + 1) for i in range(1, len(vertices) - 1)]
    else:
        raise ValueError("Panel surface_patch must define vertex_indices or face_indices.")

    return PanelPayload(
        name=panel.panel_id,
        vertices=tuple(vertices),
        faces=tuple(remapped_faces),
        metadata=metadata,
    )


__all__ = ["PanelPayloadSource", "panel_to_payload"]
