"""Panel payload schema for pattern export inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class PanelPayload:
    """Serialized panel payload used by pattern exporters."""

    name: str
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PanelPayload":
        name = str(payload.get("name", "panel"))
        vertices = tuple(
            tuple(map(float, vertex))
            for vertex in payload.get("vertices", [])
        )
        faces = tuple(
            tuple(int(idx) for idx in face)
            for face in payload.get("faces", [])
        )
        metadata = dict(payload.get("metadata", {}) or {})
        return cls(name=name, vertices=vertices, faces=faces, metadata=metadata)

    def to_mapping(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "vertices": [list(vertex) for vertex in self.vertices],
            "faces": [list(face) for face in self.faces],
        }
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


__all__ = ["PanelPayload"]
