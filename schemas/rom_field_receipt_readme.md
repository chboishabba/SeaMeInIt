# ROM Field Receipt Payload

`rom_field_receipt.json` is emitted by `examples/rom_aggregate_from_samples.py`
when aggregation is run with a promoted `basis_receipt.json`.

Required fields:

- `basis_receipt_hash`: SHA-256 of the basis receipt file
- `samples_hash`: SHA-256 of the ROM sample JSON payload
- `aggregation_summary_hash`: SHA-256 of `aggregation_summary.json`
- `fields_hash`: SHA-256 of `rom_fields.npz`
- `pose_count` and `total_samples`
- `pose_source`
- `fields_computed`
- `vertex_count`
- `peak_pressure_max` and `peak_pressure_percentile95`
- `field_uniformity`
- `synthetic`
- ternary `promotion`
- `blocked_consumers`

Promotion requires a promoted basis receipt, aligned vertex/component counts,
finite aggregated fields, `field_uniformity < 0.95`, and either a non-synthetic
payload or explicit synthetic promotion.

Without a promoted receipt, ROM field outputs are diagnostic-only for seam-cost
promotion.

The next sidecar in the DAG is `seam_cost_receipt.json`, emitted by
`scripts/compute_seam_costs.py` after it validates this receipt, the referenced
`rom_fields.npz` hash, and the promoted body/correspondence solve-domain gates.
