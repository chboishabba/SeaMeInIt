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

## ROM Operator Clarification (2026-03-09)
- Resolved archived thread metadata with `robust-context-fetch`:
  - title: `Branch · Three-kernel coupling for ROM`
  - online UUID: `696f0c80-f2e0-8322-b8a3-7b59b1ce3835`
  - canonical thread id: `2732a8b3196238d99153d6dfe71992a95d59bd7e`
  - source used: `db`
  - main topics: ROM as an explicit operator, L0-L3 schedule/completeness, canonical basis + coefficient representation, Sprint R reality check
- Resolved archived thread metadata with `robust-context-fetch`:
  - title: `seameinit`
  - online UUID: unknown in archive
  - canonical thread id: `11a134a7c680f9cd5e4fe9d1be468f8cd21c23fd`
  - source used: `db`
  - main topics: roadmap/planning; not authoritative for `ogre == ROM invariant`
- Local conclusion to keep repo wording aligned:
  - archived ROM math/spec language supports "ROM as compressed operator over admissible pose space", not "ogre is the ROM invariant"
  - `human` and `ogre` remain topology/provenance labels for current artifact families (`v3240` vs `v9438`), not operator-level identities
  - the closest current operator-level ROM artifacts are the canonical basis, sampler coefficient samples/provenance, and schedule/certificate docs; seam-cost NPZs are already topology-bound projections
- Follow-through required in repo docs/TODO:
  - document "operator-level ROM vs domain-level artifacts" explicitly,
  - add an inspectable ROM artifact path so users can view the ROM object without relying on `ogre` renders.

## ROM Operator Reporting (2026-03-09)
- Implemented operator-level coefficient export in `smii.rom.sampler_real`:
  - new optional inputs/outputs: `--basis` and `--out-coeff-samples`
  - current exported field name is `seam_sensitivity`
  - coefficients are derived by encoding the sampled per-pose sensitivity field against the orthonormal basis
- Added static operator report CLI:
  - `scripts/render_rom_operator_report.py`
  - inputs: basis + ROM meta, optional coeff samples/envelope/certificate/costs/body
  - outputs: `index.html`, `report_manifest.json`, `coeff_summary.json`, PNG diagnostics
- Strategy 2 bundles can now include operator artifacts explicitly:
  - optional ROM inputs on `scripts/protocol_strategy2_bundle.py`
  - manifest entries now declare `artifact_level`, `role`, `topology`, and `domain`
  - bundle can render a ROM operator report under `rom_operator/` when basis + meta are supplied
