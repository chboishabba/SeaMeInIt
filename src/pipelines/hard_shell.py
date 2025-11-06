"""Hard-shell pipeline that emits attachment metadata for manufacturing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableSequence, Sequence

from ..schemas.validators import (
    HARD_LAYER_ATTACHMENT_SCHEMA_NAME,
    validate_attachment_catalog,
)
from ..suit_hard.attachments import AttachmentLayout, AttachmentPlanner, PanelSegment

__all__ = ["HardShellPipeline", "generate_hard_shell_payload"]


@dataclass(slots=True)
class HardShellPipeline:
    """Coordinate hard-shell panel metadata and attachment emission."""

    catalog: Mapping[str, object]
    schema_name: str = HARD_LAYER_ATTACHMENT_SCHEMA_NAME

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook trivial
        validate_attachment_catalog(self.catalog, schema_name=self.schema_name)

    def generate(self, panels: Sequence[PanelSegment | Mapping[str, object]]) -> Mapping[str, object]:
        """Return a payload containing attachment metadata and visualization aids."""

        panel_objects: list[PanelSegment] = [
            panel if isinstance(panel, PanelSegment) else PanelSegment.from_mapping(panel)
            for panel in panels
        ]
        planner = AttachmentPlanner(self.catalog)
        layout = planner.place(panel_objects)
        return _assemble_payload(panel_objects, layout, catalog=self.catalog)


def generate_hard_shell_payload(
    panels: Iterable[PanelSegment | Mapping[str, object]],
    *,
    catalog: Mapping[str, object],
    schema_name: str = HARD_LAYER_ATTACHMENT_SCHEMA_NAME,
) -> Mapping[str, object]:
    """Functional helper that mirrors :class:`HardShellPipeline.generate`."""

    validate_attachment_catalog(catalog, schema_name=schema_name)
    panel_objects: list[PanelSegment] = [
        panel if isinstance(panel, PanelSegment) else PanelSegment.from_mapping(panel)
        for panel in panels
    ]
    layout = AttachmentPlanner(catalog).place(panel_objects)
    return _assemble_payload(panel_objects, layout, catalog=catalog)


def _assemble_payload(
    panels: Sequence[PanelSegment],
    layout: AttachmentLayout,
    *,
    catalog: Mapping[str, object],
) -> Mapping[str, object]:
    panel_metadata: MutableSequence[Mapping[str, object]] = []
    for panel in panels:
        panel_metadata.append(dict(panel.to_metadata()))
    placements = [attachment.to_dict() for attachment in layout.attachments]
    visualization = layout.visualization_payload()

    return {
        "panels": list(panel_metadata),
        "attachments": {
            "catalog": catalog,
            "placements": placements,
            "panel_summary": {panel_id: dict(summary) for panel_id, summary in layout.panel_summary.items()},
        },
        "visualization": visualization,
    }
