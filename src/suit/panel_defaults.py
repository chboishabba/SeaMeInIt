"""Material-specific panel budget defaults."""

from __future__ import annotations

from enum import Enum
from typing import Mapping

from .panel_model import PanelBudgets


class SuitMaterial(str, Enum):
    NEOPRENE = "neoprene"
    WOVEN = "woven"


# Neoprene budgets: metres, radians-per-metre, and degrees (for distortion_max).
NEOPRENE_DEFAULT_BUDGETS = PanelBudgets(
    distortion_max=8.0,
    curvature_min_radius=0.015,
    turning_max_per_length=26.18,
    min_feature_size=0.01,
)

# Woven defaults are conservative placeholders for plain-weave fabrics.
WOVEN_DEFAULT_BUDGETS = PanelBudgets(
    distortion_max=4.0,
    curvature_min_radius=0.025,
    turning_max_per_length=13.09,
    min_feature_size=0.015,
)

MATERIAL_BUDGETS: Mapping[SuitMaterial, PanelBudgets] = {
    SuitMaterial.NEOPRENE: NEOPRENE_DEFAULT_BUDGETS,
    SuitMaterial.WOVEN: WOVEN_DEFAULT_BUDGETS,
}


def panel_budgets_for(material: SuitMaterial) -> PanelBudgets:
    return MATERIAL_BUDGETS[material]


__all__ = [
    "MATERIAL_BUDGETS",
    "NEOPRENE_DEFAULT_BUDGETS",
    "SuitMaterial",
    "WOVEN_DEFAULT_BUDGETS",
    "panel_budgets_for",
]
