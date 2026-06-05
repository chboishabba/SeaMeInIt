# CutTopologyReceipt

`CutTopologyReceipt` sits between `SolverPromotionReceipt` and panel unwrap. It
is emitted by `scripts/validate_cut_topology.py` after a promoted solver receipt
has produced seam edges.

Required fields:

- `solver_receipt_hash`: sha256 of the consumed solver receipt JSON.
- `mesh_hash`: sha256 of the mesh NPZ used for topology validation.
- `seam_edges_hash`: sha256 of the consumed `seam_edges.npz`.
- seam graph counts: edge segments, vertices, connected components, endpoints,
  and branch vertices.
- panel counts: `panel_count`, `panel_face_counts`,
  `panel_boundary_edge_counts`, and `panels_are_disks`.
- typed operator counts: darts, gussets, relief cuts, easing, stretch zones, and
  aggregate `typed_operator_count`.
- classification counts for ordinary boundaries, typed operators, and invalid
  fragmentation.
- `cut_topology_blockers`, `blocked_consumers`, and ternary `promotion`.

Promotion requires a promoted solver receipt, matching seam hashes, disk-like
cut panels, and no invalid fragmentation. Branchy/open seam components may be
authorized only when classified as typed correction operators; accidental
fragmentation remains blocked.
