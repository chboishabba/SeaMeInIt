# Seam Cost Receipt

`SeamCostReceipt` is the Gate 4 sidecar for edge-cost artifacts consumed by
seam solvers. It binds promoted body and ROM-field receipts to a concrete
`seam_costs.npz` file and records whether the costs are admissible for solver
promotion.

Required fields:

- `rom_field_receipt_hash`: SHA-256 of the `ROMFieldReceipt` JSON used to build
  the costs.
- `body_receipt_hash`: SHA-256 of the promoted `BodyCarrierReceipt` JSON.
- `correspondence_receipt_hash`: SHA-256 of a promoted correspondence receipt,
  or `null` when solving natively on `A_v3240`.
- `solve_domain`: `A_v3240` for native carrier solves, or `B_v9438` for
  transfer-backed solves.
- `vertex_count`: number of vertex costs.
- `edge_count`: number of edge costs.
- `finite_cost_coverage`: fraction of edge costs that are finite.
- `cost_uniformity`: flat-cost diagnostic where `1.0` means effectively flat
  and `0.0` means highly varied.
- `peak_cost` and `mean_cost`: aggregate cost diagnostics.
- `weight_vector`: the cost weights or legacy aggregation weights used to
  produce the artifact.
- `costs_hash`: SHA-256 of the `seam_costs.npz` artifact.
- `promotion`: ternary promotion state, `-1`, `0`, or `1`.
- `blocked_consumers`: explicit consumers blocked when the receipt is not
  promoted.

Promotion requires:

- promoted `BodyCarrierReceipt`,
- promoted `ROMFieldReceipt`,
- promoted `CorrespondenceReceipt` for transfer-backed `B_v9438` solves, or
  native `solve_domain=A_v3240`,
- vertex-count agreement with the ROM-field receipt,
- high finite edge-cost coverage,
- non-flat edge costs (`cost_uniformity < 0.95`).

Unpromoted receipts default to blocking `solver_promotion`, `panel_unwrap`, and
`manufacturing`.
