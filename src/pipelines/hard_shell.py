"""Hard-shell pipeline that emits attachment metadata for manufacturing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableSequence, Sequence

from ..modules.power import PowerLayoutPlanner
from ..schemas.validators import (
    HARD_LAYER_ATTACHMENT_SCHEMA_NAME,
    POWER_MODULE_SCHEMA_NAME,
    validate_attachment_catalog,
    validate_power_catalog,
)
from ..suit_hard.attachments import AttachmentLayout, AttachmentPlanner, PanelSegment

__all__ = ["HardShellPipeline", "generate_hard_shell_payload"]


@dataclass(slots=True)
class HardShellPipeline:
    """Coordinate hard-shell panel metadata and attachment emission."""

    catalog: Mapping[str, object]
    power_catalog: Mapping[str, object] | None = None
    schema_name: str = HARD_LAYER_ATTACHMENT_SCHEMA_NAME
    power_schema_name: str = POWER_MODULE_SCHEMA_NAME

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook trivial
        validate_attachment_catalog(self.catalog, schema_name=self.schema_name)
        if self.power_catalog is not None:
            validate_power_catalog(self.power_catalog, schema_name=self.power_schema_name)

    def generate(
        self,
        panels: Sequence[PanelSegment | Mapping[str, object]],
        *,
        power_modules: Sequence[Mapping[str, object]] | None = None,
        power_catalog: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        """Return a payload containing attachment metadata and visualization aids."""

        panel_objects: list[PanelSegment] = [
            panel if isinstance(panel, PanelSegment) else PanelSegment.from_mapping(panel)
            for panel in panels
        ]
        planner = AttachmentPlanner(self.catalog)
        layout = planner.place(panel_objects)
        payload = _assemble_payload(panel_objects, layout, catalog=self.catalog)

        effective_power_catalog = power_catalog or self.power_catalog
        if effective_power_catalog and power_modules:
            power_planner = PowerLayoutPlanner(
                effective_power_catalog, schema_name=self.power_schema_name
            )
            power_layout = power_planner.allocate(power_modules)
            payload["power"] = power_layout.to_payload()

        return payload


def generate_hard_shell_payload(
    panels: Iterable[PanelSegment | Mapping[str, object]],
    *,
    catalog: Mapping[str, object],
    power_modules: Sequence[Mapping[str, object]] | None = None,
    power_catalog: Mapping[str, object] | None = None,
    schema_name: str = HARD_LAYER_ATTACHMENT_SCHEMA_NAME,
    power_schema_name: str = POWER_MODULE_SCHEMA_NAME,
) -> Mapping[str, object]:
    """Functional helper that mirrors :class:`HardShellPipeline.generate`."""

    validate_attachment_catalog(catalog, schema_name=schema_name)
    if power_catalog is not None:
        validate_power_catalog(power_catalog, schema_name=power_schema_name)

    panel_objects: list[PanelSegment] = [
        panel if isinstance(panel, PanelSegment) else PanelSegment.from_mapping(panel)
        for panel in panels
    ]
    layout = AttachmentPlanner(catalog).place(panel_objects)
    payload = _assemble_payload(panel_objects, layout, catalog=catalog)

    if power_catalog is not None and power_modules:
        power_layout = PowerLayoutPlanner(power_catalog, schema_name=power_schema_name).allocate(
            power_modules
        )
        payload["power"] = power_layout.to_payload()

    return payload


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
            "panel_summary": {
                panel_id: dict(summary) for panel_id, summary in layout.panel_summary.items()
            },
        },
        "visualization": visualization,
    }
