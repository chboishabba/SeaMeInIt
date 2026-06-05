"""Seam graph utilities and solvers built on ROM-derived cost fields."""

from .edge_costs import EdgeCostResult, build_edge_costs
from .fabric_kernels import (
    FabricAssignment,
    FabricProfile,
    fabric_penalty,
    load_fabric_profile,
    load_fabrics_from_dir,
    rotate_grain,
)
from .cut_topology_receipt import (
    CutTopologyReceipt,
    can_consume_cut_topology_receipt,
    load_cut_topology_receipt,
)
from .cut_topology_repair_receipt import (
    CUT_TOPOLOGY_REPAIR_SCHEMA,
    CutTopologyRepairReceipt,
    can_consume_cut_topology_repair_receipt,
    load_cut_topology_repair_receipt,
)
from .correction_tree_materialization_receipt import (
    CORRECTION_TREE_MATERIALIZATION_SCHEMA,
    CorrectionTreeMaterializationEntry,
    CorrectionTreeMaterializationReceipt,
    can_consume_correction_tree_materialization_receipt,
    load_correction_tree_materialization_receipt,
)
from .correction_operator_scoring import OPERATOR_FAMILIES, price_correction_operator_tree
from .kernels import EdgeKernel, KernelWeights, build_edge_kernels, edge_energy
from .manufacturing_receipt import (
    ManufacturingReceipt,
    can_consume_manufacturing_receipt,
    load_manufacturing_receipt,
)
from .metric_correction_receipt import (
    CORRECTION_STATES,
    CORRECTION_TYPES,
    MetricCorrectionEntry,
    MetricCorrectionReceipt,
    can_consume_metric_correction_receipt,
    load_metric_correction_receipt,
)
from .metric_panelization import (
    CORRECTION_FAMILIES,
    MetricEnergyWeights,
    build_metric_panelization_payload,
    normalize_families,
)
from .mdl import MDLPrior, mdl_cost, mdl_terms
from .panel_unwrap_receipt import (
    PanelUnwrapReceipt,
    can_consume_panel_unwrap_receipt,
    load_panel_unwrap_receipt,
)
from .panel_serialization_competition import (
    PANEL_SERIALIZATION_BACKENDS,
    PANEL_SERIALIZATION_SCHEMA,
    SerializationCandidateReceipt,
    XATLAS_BACKEND,
    build_panel_serialization_competition_receipt,
    select_serialization_candidate,
    serialize_panel,
)
from .pda import solve_seams_pda
from .seam_cost_receipt import (
    SeamCostReceipt,
    can_consume_seam_cost_receipt,
    load_seam_cost_receipt,
)
from .seam_derivation import (
    FinishedSeamReceipt,
    can_consume_finished_seam_receipt,
    derive_finished_seams,
    load_finished_seam_receipt,
)
from .solver_promotion_receipt import (
    SolverPromotionReceipt,
    can_consume_solver_promotion_receipt,
    load_solver_promotion_receipt,
)
from .solver import PanelSolution, SeamSolution, solve_seams
from .task_profiles import TaskProfile, aggregate_rom_for_task, load_task_profile
from .solvers_mincut import solve_seams_mincut
from .solvers_sp import solve_seams_shortest_path
from .unwrap_benchmark import (
    GRAPH_ULTRAMETRIC_RECTANGLE,
    ORTHOGRAPHIC_SQUARE,
    SPHERE_RECTANGLE_CANDIDATES,
    SphereRectangleBenchmark,
    benchmark_sphere_rectangle_unwraps,
    build_uv_sphere_mesh,
)

__all__ = [
    "EdgeCostResult",
    "EdgeKernel",
    "FabricAssignment",
    "FabricProfile",
    "FinishedSeamReceipt",
    "KernelWeights",
    "ManufacturingReceipt",
    "MetricCorrectionEntry",
    "MetricCorrectionReceipt",
    "MetricEnergyWeights",
    "CutTopologyReceipt",
    "CutTopologyRepairReceipt",
    "CUT_TOPOLOGY_REPAIR_SCHEMA",
    "CORRECTION_TREE_MATERIALIZATION_SCHEMA",
    "CorrectionTreeMaterializationEntry",
    "CorrectionTreeMaterializationReceipt",
    "TaskProfile",
    "MDLPrior",
    "PanelSolution",
    "SerializationCandidateReceipt",
    "PanelUnwrapReceipt",
    "SeamCostReceipt",
    "SeamSolution",
    "SolverPromotionReceipt",
    "SphereRectangleBenchmark",
    "build_edge_costs",
    "build_edge_kernels",
    "build_metric_panelization_payload",
    "can_consume_manufacturing_receipt",
    "can_consume_metric_correction_receipt",
    "can_consume_cut_topology_receipt",
    "can_consume_cut_topology_repair_receipt",
    "can_consume_correction_tree_materialization_receipt",
    "can_consume_finished_seam_receipt",
    "can_consume_panel_unwrap_receipt",
    "can_consume_seam_cost_receipt",
    "edge_energy",
    "fabric_penalty",
    "load_fabric_profile",
    "load_fabrics_from_dir",
    "load_manufacturing_receipt",
    "load_metric_correction_receipt",
    "load_cut_topology_receipt",
    "load_cut_topology_repair_receipt",
    "load_correction_tree_materialization_receipt",
    "load_finished_seam_receipt",
    "load_panel_unwrap_receipt",
    "load_seam_cost_receipt",
    "can_consume_solver_promotion_receipt",
    "load_solver_promotion_receipt",
    "aggregate_rom_for_task",
    "mdl_cost",
    "mdl_terms",
    "load_task_profile",
    "normalize_families",
    "rotate_grain",
    "solve_seams",
    "solve_seams_pda",
    "solve_seams_mincut",
    "solve_seams_shortest_path",
    "CORRECTION_FAMILIES",
    "CORRECTION_STATES",
    "CORRECTION_TYPES",
    "GRAPH_ULTRAMETRIC_RECTANGLE",
    "ORTHOGRAPHIC_SQUARE",
    "OPERATOR_FAMILIES",
    "SPHERE_RECTANGLE_CANDIDATES",
    "PANEL_SERIALIZATION_BACKENDS",
    "PANEL_SERIALIZATION_SCHEMA",
    "XATLAS_BACKEND",
    "benchmark_sphere_rectangle_unwraps",
    "build_uv_sphere_mesh",
    "derive_finished_seams",
    "price_correction_operator_tree",
    "build_panel_serialization_competition_receipt",
    "select_serialization_candidate",
    "serialize_panel",
]
