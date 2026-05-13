"""ROM-aligned helpers for seam-aware kernels and constraints."""

from .basis import BasisMetadata, KernelBasis, KernelProjector, load_basis
from .basis_receipt import (
    BasisReceipt,
    DEFAULT_BASIS_BLOCKED_CONSUMERS,
    can_consume_basis_receipt,
    load_basis_receipt,
)
from .constraints import ConstraintRegistry, ConstraintSet, load_constraints
from .gates import (
    CouplingManifest,
    CouplingRule,
    GateDecision,
    GateReason,
    GateRuntimeState,
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
from .seam_costs import (
    SeamCostField,
    annotate_seam_graph_with_costs,
    build_seam_cost_field,
    load_seam_cost_field,
    save_seam_cost_field,
)

__all__ = [
    "BasisMetadata",
    "BasisReceipt",
    "DEFAULT_BASIS_BLOCKED_CONSUMERS",
    "KernelBasis",
    "KernelProjector",
    "can_consume_basis_receipt",
    "load_basis",
    "load_basis_receipt",
    "ConstraintRegistry",
    "ConstraintSet",
    "load_constraints",
    "CouplingManifest",
    "CouplingRule",
    "GateDecision",
    "GateReason",
    "GateRuntimeState",
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
    "annotate_seam_graph_with_costs",
    "build_seam_cost_field",
    "load_seam_cost_field",
    "save_seam_cost_field",
]
