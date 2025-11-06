"""Utilities for generating undersuit panels with material assignments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..suit.material_model import MaterialCatalog

__all__ = ["UndersuitPipeline"]


@dataclass(slots=True)
class UndersuitPipeline:
    """Simplified undersuit generator that attaches material data to panels."""

    material_catalog: MaterialCatalog
    panel_templates: Sequence[str]
    default_stack_id: str
    metadata_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.default_stack_id not in self.material_catalog.stacks:
            raise KeyError(
                f"Default stack '{self.default_stack_id}' is not defined in the catalog."
            )
        self.panel_templates = tuple(self.panel_templates)

    def generate(
        self,
        *,
        subject_id: str,
        panel_materials: Mapping[str, str] | None = None,
        panel_blends: Mapping[str, tuple[str, str, float]] | None = None,
    ) -> dict[str, Any]:
        """Return undersuit panel records enriched with material metadata."""

        panel_materials = panel_materials or {}
        panel_blends = panel_blends or {}

        panels: list[dict[str, Any]] = []
        materials_metadata: dict[str, Any] = {}

        for panel_name in self.panel_templates:
            if panel_name in panel_blends:
                primary, secondary, ratio = panel_blends[panel_name]
                blend = self.material_catalog.blend(
                    primary, secondary, ratio, name=f"{panel_name} blend"
                )
                panel_payload = {
                    "name": panel_name,
                    "material_stack_id": f"blend:{blend.primary_id}:{blend.secondary_id}:{blend.ratio:.2f}",
                    "material_stack": blend.to_metadata(),
                }
            else:
                stack_id = panel_materials.get(panel_name, self.default_stack_id)
                stack = self.material_catalog.get_stack(stack_id)
                panel_payload = {
                    "name": panel_name,
                    "material_stack_id": stack.identifier,
                    "material_stack": stack.to_metadata(),
                }

            panels.append(panel_payload)
            materials_metadata[panel_name] = panel_payload["material_stack"]

        metadata: dict[str, Any] = {
            "subject_id": subject_id,
            "material_catalog_version": self.material_catalog.version,
            "materials": materials_metadata,
        }
        metadata.update(self.metadata_overrides)

        return {
            "panels": panels,
            "metadata": metadata,
        }
