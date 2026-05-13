# SolverPromotionReceipt

`solver_promotion_receipt.json` is emitted by `scripts/solve_seams.py` after a
solver consumes a promoted `seam_cost_receipt.json`. It is the Gate 5 receipt
between receipted seam costs and panel unwrap.

Required fields:

- `seam_cost_receipt_hash`: SHA-256 of the promoted seam-cost receipt.
- `solver_mode`: solver family, such as `shortest_path`, `min_cut`, or
  `pda_mst`.
- `anchor_count` and `anchor_source`: anchor count and provenance.
- `connected_component_count`: number of components in the selected anchor
  subgraph before fallback.
- `anchor_fallback_used`: whether largest-component fallback was used.
- `seam_edge_count`, `seam_vertex_count`, and `total_seam_cost`: seam artifact
  size and aggregate cost.
- `panel_count` and `panels_are_disks`: topology handoff state for panel
  unwrap.
- `seam_hash`: SHA-256 of the saved `seam_edges.npz` artifact.
- `promotion`: `1` promoted, `0` diagnostic, `-1` rejected.
- `blocked_consumers`: explicit downstream consumers blocked when not promoted.

Promotion requires a promoted seam-cost receipt and panel topology that passes
the current disk proxy. Anchor fallback is allowed but recorded so anchor
selection can be audited instead of silently corrected.
