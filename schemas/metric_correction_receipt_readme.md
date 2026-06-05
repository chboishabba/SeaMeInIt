# MetricCorrectionReceipt

`MetricCorrectionReceipt` is the sidecar bridge between promoted cut topology and
panel unwrap when cut topology contains typed correction operators. It is emitted
by `scripts/emit_metric_correction_receipt.py`.

Required fields:

- `solver_receipt_hash`: sha256 of the consumed solver receipt JSON.
- `cut_topology_receipt_hash`: sha256 of the consumed cut-topology receipt JSON.
- `seam_edges_hash`: sha256 of the consumed `seam_edges.npz`.
- `panels_requiring_correction`: panel labels covered by selected corrections.
- `corrections`: typed correction entries with `panel_label`,
  `correction_type`, `delta_metric_meaning`, raw/corrected residuals,
  `energy_terms`, `result_state`, and entry blockers.
- `raw_residual_total`, `corrected_residual_total`, and `residual_gate`.
- `metric_correction_blockers`, `blocked_consumers`, and ternary `promotion`.
- Optional `correction_payload_hash` for the source `corrections.json`.

Promotion requires matching solver/cut/seam hashes, compatible typed correction
entries when typed operators are present, corrected residuals within the gate,
and no metric-correction blockers. Missing typed correction evidence blocks
`panel_unwrap` and `manufacturing`.
