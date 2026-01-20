# SeamInit Sprint 0 — Alignment & Contracts

Authoritative goals come from `CONTEXT.md` (see sprint definitions at CONTEXT.md:350-391 and acceptance tests at CONTEXT.md:683-728). This note captures the concrete artifacts we need to produce and where they should live.

## Object Map (D0.1)

- **ROM PDA / cocycles:** Definitions from `CONTEXT.md` drive admissibility. Action: codify PDA/coupling symbols and gates in a dedicated module stub (`smii/rom/gates.py`) so later code can call a single interface.
- **Kernel coefficients / basis:** Canonical body basis (B0) referenced by Sprint 1. Action: track B0 metadata alongside the matrix (source mesh, normalization, ordering) in a structured payload.
- **Mesh graph + seams:** Mesh graph (V, E), seam cut sets, and panel labels. Action: define seam graph payload schema compatible with exporters (edge list + per-edge admissibility + panel labels).
- **Couplings (anatomical + seam):** Shared registry of obligations/violations. Action: enumerate coupling ids, severity, and blocking rules in a JSON/YAML manifest so PDA gating and seam costs share constants.

## Interface Spec (D0.2)

- **Mesh / basis:** Use existing mesh schema (`schemas/body_unified.yaml`) and add a basis companion NPZ with keys `basis` (N×K) and `vertices` (N×3). Documented in `schemas/basis_readme.md` (new).
- **ROM samples:** NPZ or JSONL with `{pose_id, coeffs: {T,P,S,Sigma}}`; acceptance gate applied upstream. Schema to be added under `schemas/rom_samples.yaml`.
- **Fabric parameters:** Reuse `schemas/suit_materials.yaml` when possible; add stretch/anisotropy axes required for seam costs.
- **Seam outputs:** JSON with `edges`, `panels`, `edge_costs`, `edge_labels` (admissible/forbidden/preferred), and optional `seam_allowance`. Align with `exporters.patterns` expectations.

## Constraint Registry (D0.3)

- **Forbidden zones / anchors / symmetry:** Store under `data/constraints/`:
  - `forbidden_vertices.json`: vertex ids by region.
  - `anchors.json`: named landmark vertex ids.
  - `symmetry_pairs.json`: vertex/edge pairing for mirroring.
- **Panel count / seam length limits:** Config JSON with global defaults and garment-class overrides. Keep together with other constraint files.
- **Lookup interface:** Provide a thin helper in `smii/rom/constraints.py` to deliver deterministic labels per vertex/edge for solvers and validators.

## Next Actions to Exit Sprint 0

- Add the new schema stubs (`schemas/basis_readme.md`, `schemas/rom_samples.yaml`) and constraint manifests under `data/constraints/`.
- Create gate/constraint helper modules (`smii/rom/gates.py`, `smii/rom/constraints.py`) with clear docs and shape checks.
- Wire documentation of payload fields into the CLI README or `CONTEXT.md` references to keep future contributors aligned.
