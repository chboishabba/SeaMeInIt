# PanelUnwrapReceipt

`PanelUnwrapReceipt` is Gate 6 in the SMII receipt DAG. It is emitted by
`scripts/unwrap_panels.py` after promoted solver and cut-topology receipts have
produced a seam topology whose panels are topological disks. If the cut topology
contains typed correction operators, a promoted `MetricCorrectionReceipt` is
required before panel unwrap may promote. Branches are correction-tree/operator
nodes when typed, not intrinsic blockers.

Required fields:

- `solver_receipt_hash`: sha256 of the consumed solver receipt JSON.
- `panel_count`: number of extracted post-cut panels.
- `panels_all_disks`: copied from the solver topology gate.
- `per_panel_distortion`: distortion value for each panel.
- `worst_panel_distortion` and `mean_panel_distortion`: aggregate distortion
  diagnostics.
- `distortion_threshold`: maximum distortion allowed for promotion.
- `subdivision_iterations`: maximum subdivision retries used by any panel.
- `grain_directions`: one grain label per panel, `warp`, `weft`, or `bias`.
- `uv_hash`: sha256 of `panel_uvs.npz`.
- `seam_topology_hash`: copied from `SolverPromotionReceipt.seam_hash`.
- `cut_topology_receipt_hash`: sha256 of the consumed cut-topology receipt when
  present.
- `unwrap_backend`: `bootstrap_projection` or `lscm`.
- `backend_is_bootstrap`: whether the bootstrap projection backend produced the
  UVs.
- `distortion_margin`: threshold minus worst observed distortion.
- `panel_unwrap_blockers`: local blockers such as distortion or missing metric
  correction evidence.
- corrected residual fields from metric correction when available:
  `per_panel_corrected_residual`, `worst_corrected_residual`,
  `mean_corrected_residual`, `correction_payload_hash`, and
  `metric_correction_receipt_hash`.
- fabric-relative metric fields when available, mirrored from
  `FabricAwarePanelMetricReceipt`: `fabric_receipt_hash`,
  `fabric_profile_id`, `fabric_relative_threshold`, `fabric_metric_gate`,
  `stretch_compliance`, and `shear_compliance`.
- correction-tree fields when typed operators are present, mirrored from
  `CorrectionTreeReceipt`: `correction_tree_id`, `root_panel_label`,
  `operator_nodes`, `parent_node_id`, `operator_type`, `branch_degree`,
  `delta_metric_meaning`, and `metric_propagation_law`.
- correction-operator scoring fields when branch pricing runs, mirrored from
  `CorrectionOperatorScoringReceipt`: `branch_count`, `typed_branch_count`,
  `diagnostic_branch_count`, `residual_before`, `fabric_violation_before`,
  candidate operator scores, selected operators, and estimated residual/fabric
  violation after the priced operator tree.
- realized correction-operator fields when implementation exists, mirrored from
  `RealizedCorrectionOperatorReceipt`: realized operator count, unsupported
  operator blockers, local fabric-cone overrides, cut-sheet annotation labels,
  and realized residual/fabric violation after the operator tree. The current
  realized implementation supports `stretch_zone` and `gusset_corner` companion
  residual relief.
- `promotion`: ternary receipt state, `-1`, `0`, or `1`.
- `blocked_consumers`: downstream consumers blocked while non-promoted.

Promotion requires promoted solver/cut topology, disk-panel topology,
hash-matched seam and mesh inputs, promoted metric correction when typed
operators are present, no unpriced correction-tree nodes after operator scoring,
and all panel distortion/fabric-relative residual values at or below the
declared threshold. A non-promoted unwrap blocks `manufacturing`.
