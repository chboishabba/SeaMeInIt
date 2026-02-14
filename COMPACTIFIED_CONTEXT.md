# Compact Context Snapshot

Date: 2026-02-06

## Current Direction
- ROM is treated as a first-class, schedulable operator with explicit levels (L0–L3), artifacts, and stop criteria; no seam solver or fabric changes in Sprint R.
- L0/L1/L2 sweeps are minimal and curated; L3 is a completeness certificate rather than more sampling.
- MoCap augments density later; it does not define ROM limits.
- Seam solvers and PDA/MDL stack remain frozen while ROM formalisation is delivered.

## Sprint R Core Deliverables
- Code: `smii/rom/pose_schedule.py`, `smii/rom/completeness.py`, and `sampler_real --schedule`.
- Data: `data/rom/sweep_schedule.yaml`, `data/rom/task_profiles/*.yaml`.
- Outputs: `outputs/rom/rom_samples_L0.json`, `rom_samples_L1.json`, `rom_samples_L2.json`, `outputs/rom/rom_L3_certificate.json`.

## Completeness Metrics
- Envelope convergence (99% below ε).
- Seam cost rank stability (Spearman > 0.98).
- MDL mass saturation (incremental MDL contribution trends to zero).

## Operational Fixes To Track
- Add mtimes/hashes to detect no-op ROM/heatmap runs.
- Switch measurement fixtures to explicit `measurements.yaml` ingestion (PGMs deprecated).
- Add runtime perf attribution to flag “GPU-assisted” runs that are CPU-bound.

## Seam Debug Snapshot (2026-02-12)
- Ogre/pathology overlays are now documented as explicit failure modes in `docs/ogre_artifact_diagnostics.md`.
- `shortest_path` remains the baseline open-path solver, with optional controls:
  - `require_loop` (loop attempt),
  - `symmetry_penalty_weight` (mirrored edge mismatch),
  - `allow_unfiltered_fallback` (strict locality toggle),
  - `reference_vertices` (origin-mesh seam length diagnostics).
- TODO and changelog were aligned to these controls before implementation.

## Mesh Provenance Snapshot (2026-02-13)
- Added `docs/mesh_provenance_afflec.md` to disambiguate `afflec_body` vs `afflec_canonical_basis`.
- Current working branch has two topology families in artifacts:
  - base/demo pipeline: `3240` vertices (`outputs/afflec_demo/afflec_body.npz`),
  - legacy realshape/suit branch: `9438` vertices (`outputs/suits/afflec_body/base_layer.npz` and related seam runs).
- Added `scripts/audit_mesh_lineage.py` to produce JSON/CSV lineage checks over body, ROM costs/meta, ogre seam report, and reprojected seam report.

## Pipeline Position Snapshot (2026-02-13)
- Added `docs/seam_pipeline_intended_vs_observed.md` as the explicit alignment document for:
  - intended pipeline stage semantics,
  - user-observed and agent-observed mismatch behavior,
  - unresolved decision on canonical solve domain (base-first vs ROM-first).
- Roadmap/TODO now include a decision gate requiring lineage manifests and strict transfer acceptance before cross-topology seam interpretation.
- The document now contains a runnable A-vs-B protocol and decision-record section; policy freeze remains pending execution.
- Protocol execution `outputs/seams_run/domain_ab_20260213_051532` completed:
  - Strategy A passes native checks,
  - Strategy B fails strict reprojection gates,
  - reverse-direction NN transfer also collapses (not invertible in practice),
  - provisional Strategy A freeze for interpretable outputs.
- Added persistent map tooling (`scripts/build_mesh_vertex_map.py`) and map-driven reprojection (`--vertex-map-file`); current ogre<->afflec map still fails quality gates, indicating correspondence quality issue rather than seam-point sampling issue.
