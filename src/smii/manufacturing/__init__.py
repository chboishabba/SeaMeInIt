"""Manufacturing geometry, seam allowances, nesting, and vector pattern exports."""

from __future__ import annotations

from .allowances import (
    ALLOWANCE_BY_METHOD,
    compute_seam_allowance_offset,
)
from .grainlines import (
    compute_grainline_vector,
)
from .nesting import (
    NestingConfig,
    NestingResult,
    Placement,
    nest_charts_deterministic,
)
from .notches import (
    generate_seam_notches,
)
from .svg_export import (
    export_full_suit_svg,
)
from .validation import (
    ManufacturingValidationReport,
    validate_manufacturing_layout,
)

__all__ = [
    "ALLOWANCE_BY_METHOD",
    "ManufacturingValidationReport",
    "NestingConfig",
    "NestingResult",
    "Placement",
    "compute_grainline_vector",
    "compute_seam_allowance_offset",
    "export_full_suit_svg",
    "generate_seam_notches",
    "nest_charts_deterministic",
    "validate_manufacturing_layout",
]
