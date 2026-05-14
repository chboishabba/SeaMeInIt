# PanelUnwrapReceipt

`PanelUnwrapReceipt` is Gate 6 in the SMII receipt DAG. It is emitted by
`scripts/unwrap_panels.py` after a promoted `SolverPromotionReceipt` has
produced a seam topology whose panels are topological disks.

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
- `promotion`: ternary receipt state, `-1`, `0`, or `1`.
- `blocked_consumers`: downstream consumers blocked while non-promoted.

Promotion requires a promoted solver receipt, disk-panel topology, hash-matched
seam edges, and all panel distortion values at or below the configured
threshold. A non-promoted unwrap blocks `manufacturing`.
