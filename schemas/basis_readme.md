# Kernel Basis Payload

Canonical body basis used for sprint work. Recommended NPZ keys:

- `basis`: float array shaped (N_vertices, K_components) (required)
- `vertices`: float array shaped (N_vertices, 3) (optional but used to verify alignment)
- `meta`: JSON-serializable dict (optional) with:
  - `source_mesh`: string identifier for the canonical mesh
  - `normalization`: description of area weighting / orthonormalization
  - `notes`: freeform annotations

Validation rules:

- `basis.shape[0]` must equal `vertices.shape[0]` when vertices are supplied.
- Components must be finite; no NaNs or infs.
- Vertex coordinates must be finite.

## Receipt sidecar

Production basis artifacts should be paired with `basis_receipt.json` emitted by
`scripts/generate_canonical_basis.py --body-receipt ...`. The receipt records:

- `carrier_receipt_hash`: SHA-256 of the body-carrier receipt file, not the mesh
  alone
- `basis_hash`: SHA-256 of the generated basis NPZ
- `basis_vertex_count` and `basis_dimension`
- `construction_method`
- `reconstruction_error` and `promotion_threshold`
- ternary `promotion`
- `blocked_consumers`

Without a promoted receipt, the basis NPZ is diagnostic-only for downstream ROM
field aggregation. The next required sidecar is `rom_field_receipt.json`; see
`schemas/rom_field_receipt_readme.md`.

This aligns with Sprint 1 exit criteria (CONTEXT.md:421-425).
