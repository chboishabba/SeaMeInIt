"""ROM-aligned helpers for seam-aware kernels and constraints."""

from .basis import BasisMetadata, KernelBasis, KernelProjector, load_basis
from .constraints import ConstraintRegistry, ConstraintSet, load_constraints
from .gates import (
    CouplingManifest,
    CouplingRule,
    GateDecision,
    GateReason,
    RomGate,
    build_gate_from_manifest,
    load_coupling_manifest,
)
from .aggregation import (
    AggregationDiagnostics,
    EdgeHotspot,
    FieldStats,
    RejectionReason,
    RejectionReport,
    RomAggregation,
    RomSample,
    VertexHotspot,
    aggregate_fields,
)
from .seam_costs import SeamCostField, build_seam_cost_field

__all__ = [
    "BasisMetadata",
    "KernelBasis",
    "KernelProjector",
    "load_basis",
    "ConstraintRegistry",
    "ConstraintSet",
    "load_constraints",
    "CouplingManifest",
    "CouplingRule",
    "GateDecision",
    "GateReason",
    "RomGate",
    "load_coupling_manifest",
    "build_gate_from_manifest",
    "FieldStats",
    "AggregationDiagnostics",
    "VertexHotspot",
    "EdgeHotspot",
    "RejectionReason",
    "RejectionReport",
    "RomAggregation",
    "RomSample",
    "aggregate_fields",
    "SeamCostField",
    "build_seam_cost_field",
]
