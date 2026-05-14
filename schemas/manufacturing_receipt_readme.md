# ManufacturingReceipt

`ManufacturingReceipt` is Gate 7 in the SMII receipt DAG. It is emitted by
`scripts/generate_manufacturing_artifacts.py` after a promoted
`PanelUnwrapReceipt` and hash-matched `panel_uvs.npz` artifact exist.

Required fields:

- `panel_unwrap_receipt_hash`: sha256 of `panel_unwrap_receipt.json`.
- `panel_count`: number of flattened panels consumed.
- `manufacturing_method`: one of `home_sewing`, `overlock`, `flatlock`,
  `bonded`, `welded`, `laser_cut`, `3d_print`, or `eva_foam_cut`.
- `accessibility_level`: `consumer`, `industrial`, or `advanced`.
- `seam_allowance_hash`: sha256 of `seam_allowance.npz`.
- `seam_allowance_mean`, `seam_allowance_min`, `seam_allowance_max`: allowance
  summary in metres.
- `allowance_varies`: whether ROM pressure/shear gradients produced a
  non-constant allowance field.
- `grain_directions`: per-panel grain direction propagated from Gate 6.
- `panel_hashes`: per-panel hashes of UV arrays from `panel_uvs.npz`.
- `cutting_artifacts_hash`: sha256 of the generated cutting layout.
- `notches_present`, `labels_present`: cutting-layout readiness checks.
- `promotion`: ternary `-1`, `0`, or `1`.
- `blocked_consumers`: empty for promoted receipts; this is the end of chain.
- `notes`: diagnostic text, especially when `allowance_varies=false`.

Promotion requires a variable allowance field plus notch and label artifacts.
Flat allowance is diagnostic-only because it means the ROM fields did not drive
manufacturing-specific allowances.
