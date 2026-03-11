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

## Archive Refresh (2026-03-10)
- Refreshed the canonical chat archive via live pull into `/home/c/chat_archive.sqlite`:
  - requested online UUIDs: `17`
  - fetched OK: `17`
  - failures: `0`
  - source id: `pull_20260310T014918Z`
  - verification source after ingest: `db`
- Pulled/confirmed online UUID -> title -> canonical thread id:
  - `690c6ad9-c920-8323-8e32-5490f9b0fbd5` -> `PyPI publishing options` -> `f963ccac8381a441603ff2e5658e45990e343138`
  - `69166b40-4468-8320-8cc5-c7e7c45c576a` -> `Zip extraction hardening` -> `40663b18f8aa5979c71cef8db3eb6b437ecaa510`
  - `69168cea-d4b4-8323-8b00-27169f5ff22d` -> `Summarise project structure` -> `4869516943aa9ebcc0707e186ad0ffbf6bb3c5cb`
  - `6916c8e8-9db4-8323-8661-eda505ba0324` -> `Project clarification request` -> `648d9a4a7d4173e282ef988d472fe9c1888e6985`
  - `6916c180-1080-8320-ae2d-acc2e3ac3c23` -> `UV unwrapping explanation` -> `3562461ee45a9f6eb3b24f0cbd4a233161a7b60e`
  - `69172671-7330-8322-a28f-969907095ea4` -> `DensePose UV map generation` -> `777f6c8f9af698b336c3fdb3200e52960c4e7f7c`
  - `691e813e-1c84-8321-90ef-977d3804bb47` -> `UV unwrapping in Blender` -> `a39cc88c9628debdfc07a3b846dd1135230e9059`
  - `691674c6-1154-8320-bf5a-facef5aa5f81` -> `Watertight mesh repair` -> `6516d5174954dc5b11b1d8cc9e8b0d3b7d777b39`
  - `6909470b-9250-8324-961f-59559af5c6bd` -> `seameinit` -> `11a134a7c680f9cd5e4fe9d1be468f8cd21c23fd`
  - `695869e8-9df0-8323-93d4-46b440ba27f8` -> `Wetsuit Design Sources` -> `53051ba8bb7446eaecd960aa7fab50b849c06d7d`
  - `696ee8ce-4800-8323-a7de-e429c5fcaace` -> `Three-kernel coupling for ROM` -> `cb8859f40d2685674cd23159227e249340de377d`
  - `696f0c80-f2e0-8322-b8a3-7b59b1ce3835` -> `Branch · Three-kernel coupling for ROM` -> `2732a8b3196238d99153d6dfe71992a95d59bd7e`
  - `69707049-9248-8323-b22d-efb493470795` -> `Pose Sweep Strategy` -> `5fe149c7b3c1e841ab0f8e6419b9fd225a3f5db9`
  - `6986c771-a08c-839b-a99f-d052720c31eb` -> `Git submodule corruption fix` -> `de233abaf3133daa385d7e663813a82e36b4d901`
  - `6985aba5-277c-839e-a8b6-3c4761a66b4a` -> `Repo Goal Summaries` -> `02e657061f13d3ea8002a35a9d500af652de0439`
  - `698d5e21-6d54-839a-a127-088c1dc21227` -> `Seam Walker Troubleshooting` -> `0eff7f41332ca191629d9246ad3677518461fa55`
  - `699050a6-e13c-839a-9a66-be7653b4db13` -> `Seam Graph Generation Debug` -> `6d14ca5f93671d7fb8e923db48654ecb5ef63b42`
- Highest-value seam/ROM thread takeaways sharpened by the refresh:
  - `Seam Graph Generation Debug`: do not let `human` / `ogre` names stand in for verified morphology; stage/provenance naming is the safer identity contract, but observed morphology still needs to be logged separately because prior runs did produce ogre-like and flailing outcomes.
  - `Pose Sweep Strategy`: when meshes/ROM heatmaps look unchanged, treat it as a no-op until mtimes + content hashes prove otherwise.
  - `Seam Walker Troubleshooting`: solver quality should be improved by structural / flattenability constraints, not ad-hoc anatomy region penalties.
  - distilled roadmap note recorded at `docs/solver_kernel_roadmap_note_20260310.md`.
  - morphology debugging follow-through recorded in `docs/seam_pipeline_intended_vs_observed.md` under "Morphology Taxonomy For Debugging".

## ROM Operator Reporting (2026-03-09)
- Implemented operator-level coefficient export in `smii.rom.sampler_real`:
  - new optional inputs/outputs: `--basis` and `--out-coeff-samples`
  - current exported field name is `seam_sensitivity`
  - coefficients are derived by encoding the sampled per-pose sensitivity field against the orthonormal basis
- Added static operator report CLI:
  - `scripts/render_rom_operator_report.py`
  - inputs: basis + ROM meta, optional coeff samples/envelope/certificate/costs/body
  - outputs: `index.html`, `report_manifest.json`, `coeff_summary.json`
  - intended presentation contract:
    - analytic report visuals (coefficient bars, norms, summaries) should be DOM-native inside the HTML, not stored as standalone PNGs
    - topology-level media artifacts that already exist on disk (`overlay.png`, flex heatmaps, GIF/WebM orbits, map orbits) should be embedded/organized inside the report page rather than left as disconnected side files
- Strategy 2 bundles can now include operator artifacts explicitly:
  - optional ROM inputs on `scripts/protocol_strategy2_bundle.py`
  - manifest entries now declare `artifact_level`, `role`, `topology`, and `domain`
  - bundle can render a ROM operator report under `rom_operator/` when basis + meta are supplied
  - next alignment step: pass bundle render/map/seam media into the report so the page acts as the primary viewing surface

## Run Reference Pages (2026-03-09)
- Current output problem:
  - specialized pages exist (`rom_operator/index.html`), but there is no canonical single-page reference for one run root
  - compare runs can complete without GIF/WebM orbit media unless a renderer is invoked explicitly
  - temporary frame PNG directories are cleaned up by `render_variant_orbits.py` and `render_vertex_map_orbits.py`, but `render_seam_orbit.py` still leaves them behind
- Intended contract moving forward:
  - each run root should have one canonical HTML reference page that embeds all completed artifacts for that run
  - a higher-level index page should catalog runs and link to each run page
  - deliberate stills like `overlay.png` stay; temporary frame PNGs used to encode GIF/WebM should be deleted after encoding
  - run pages should ignore transient frame directories and legacy operator-report chart PNGs

## Morphology Debugging Phase (2026-03-10)
- Current priority is no longer "make more artifacts"; it is to make morphology changes attributable by stage.
- Working interpretation:
  - `ogre-like` and `flailing` are observed debug morphologies, not desired targets.
  - `flailing` is more likely to be close to the intended ROM phenomenon.
  - the current ROM aggregate/operator outputs are field-oriented and do not by themselves prove a morphology transform.
- Latest local formalism cross-check:
  - consulted `/home/c/Documents/code/ITIR-suite/all_code48.txt` as the newest available `all_code*.txt` snapshot.
  - useful Dashi-side takeaway: keep the kernel/operator separate from the admissibility lens; do not treat labels or downstream artifacts as the operator itself.
  - this supports the current morphology plan: observe morphology by stage first, then judge whether the operator field is coherent with those observations.
- Inverse-transform clarification:
  - earlier project intent included: fit SMPL-X body -> internalize/deform into an ogre-like or movement-respecting solve domain -> solve seams there -> invert back to the fitted body.
  - current repo reality is weaker: correspondence/reprojection and transfer diagnostics exist, but no proven inverse ROM/internalization transform exists.
  - the `all_code44.txt` / `all_code48.txt` formalism supports caution here because projection-style operators are generally not invertible except in trivial cases.
- Prioritized milestones:
  1. backfill morphology observations on known reference runs so `run_reference/index.html` pages stop defaulting to `unclassified`,
  2. emit representative posed/deformed ROM sample artifacts so flailing can be observed directly rather than inferred from neutral-body heatmaps,
  3. clarify whether "back to body" is an inverse transform or only approximate correspondence/reprojection,
  4. compare candidate ROM fields on one topology using those morphology artifacts,
  5. only then return to seam-solver sensitivity / anchor fallback work.
- Current bounded next step from the orchestrator viewpoint:
  - `M2.1`: define representative ROM sample selection policy
  - `M2b.1`: audit current code paths for any actual inverse-transform candidate
- These are intentionally separated from later implementation so the repo does
  not jump into artifact emission without first deciding what counts as a
  representative sample and whether any existing inverse claim is real.
- Contract now written at `docs/rom_sample_morphology_and_transfer_contract.md`:
  - representative sample anchors are `max_field_l2_norm`, `max_displacement_mean_norm`, `median_field_l2_norm`, and `max_weight`, with deterministic fill by descending field norm
  - emitted sample meshes stay on sampler-native topology and exist to show flailing/pose deformation directly
  - current back-transfer audit found no true geometry inverse in the repo; only basis-space encoding plus nearest-neighbor correspondence/reprojection
  - approximate transfer is acceptable only with explicit lineage, transfer labeling, and quality gates
- Baseline M2/M2b implementation is now landed:
  - `smii.rom.sampler_real` accepts `--out-rom-samples-dir` and `--rom-sample-count`
  - representative samples are exported as sampler-native posed mesh `.npz` files plus `rom_samples/rom_sample_manifest.json`
  - `render_run_reference.py` classifies those meshes as `rom_sample_pose`
  - `render_rom_operator_report.py` can display the representative sample manifest in a dedicated section
- Orchestrator next step is now M3:
  - compare `seam_sensitivity` against displacement magnitude and derivative magnitude on one topology using the new sample-morphology outputs as the visual/morphology reference
- New priority shift on 2026-03-11:
  1. audit the exact forward object behind historical `B_ogre`
  2. isolate the current Afflec crown/head-shape failure stage
  3. define the Afflec-facing back-transfer requirement around that exact forward object
  4. specify a video-input fitting path only after the crown failure is isolated
  5. audit SMPL-X body-shape coverage, especially feminine-body fitting behavior
- Why the shift happened:
  - current Afflec outputs are showing an egg-shaped / Green-Goblin-like skull crown
  - the historical `B_ogre` object is still not pinned down as geometry vs render/domain artifact
  - the user’s intended production path still requires ROM-domain seam work to land back on the fitted Afflec mesh
- Reference note for this shift:
  - `docs/body_fit_and_inverse_roadmap_20260311.md`
- First audit pass recorded at:
  - `docs/b_ogre_and_afflec_crown_audit_20260311.md`
- Current audit conclusions:
  - historical `B_ogre` was a real native solve object on `outputs/suits/afflec_body/base_layer.npz` (`9438` vertices, SHA256 `b122dc2cf8b075a5a5bcc0c124a075247268332203df7873c36de65e4027695c`) with paired ROM costs `outputs/rom/seam_costs_afflec_realshape_edges.npz` (SHA256 `750e0472648fff6a4f324cd4b34e78648dd8c878a1b2acbd85a0c3a3c57f50d8`)
  - old `9438 -> 3240` control transfer is not a valid inverse or production back-transfer:
    - `edge_retention_ratio = 0.0526`
    - `target_vertex_collision_ratio = 0.85`
    - `quality_ok = false`
  - current Afflec crown / egg-skull pathology is now narrowed away from parameter drift between the calibrated `complete2` and `complete3` runs
  - the likely remaining problem locus is late body generation / repair / export, or a latent fitted-geometry issue only visible there
  - the missing debugging surface is export-stage mesh checkpointing before and after repair/export
- Active M1 reference runs:
  - `outputs/comparisons/afflec_raw_vs_refined_20260309/`
  - `outputs/comparisons/afflec_same_topology_20260309/`
  - `outputs/comparisons/afflec_kernel_diagnostic_raw_20260310/`
  - `outputs/assets_bundles/20260309_062711__afflec_raw_refined_verify/`
- Planning state for this phase is now tracked in:
  - `.planning/spec.md`
  - `.planning/architecture.md`
  - `.planning/plan.md`
  - `.planning/status.json`
  - `.planning/devlog.md`
