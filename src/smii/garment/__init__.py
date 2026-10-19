"""Garment pattern generation, seam topology, and conformal partitioning."""

from __future__ import annotations

from .chart_unwrap import (
    UnwrapConfig,
    count_uv_foldovers,
    extract_boundary_loops,
    solve_sparse_lscm,
    unwrap_chart,
)
from .full_suit_partition import (
    FullSuitPartitionConfig,
    FullSuitPartitioner,
    save_full_suit_pattern,
)
from .pattern_model import (
    BoundaryKind,
    FullSuitPattern,
    GarmentRegion,
    PatternChart,
    SeamConnection,
    SeamSide,
)
from .region_atlas import (
    RegionAtlas,
    RegionAtlasConfig,
    validate_region_coverage,
)
from .seam_topology import (
    SeamTopologyExtractor,
)

__all__ = [
    "BoundaryKind",
    "FullSuitPartitionConfig",
    "FullSuitPartitioner",
    "FullSuitPattern",
    "GarmentRegion",
    "PatternChart",
    "RegionAtlas",
    "RegionAtlasConfig",
    "SeamConnection",
    "SeamSide",
    "SeamTopologyExtractor",
    "UnwrapConfig",
    "count_uv_foldovers",
    "extract_boundary_loops",
    "save_full_suit_pattern",
    "solve_sparse_lscm",
    "unwrap_chart",
    "validate_region_coverage",
]
