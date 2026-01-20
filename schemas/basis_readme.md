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

This aligns with Sprint 1 exit criteria (CONTEXT.md:421-425).
