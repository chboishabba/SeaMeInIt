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

## Dart / Shaping Formalism Refresh (2026-06-04)
- Resolved archived thread metadata with `robust-context-fetch` from the canonical local DB `/home/c/chat_archive.sqlite`; no web fetch was used:
  - title: `Repo planning blockers`
  - online UUID: `6a03e573-caa4-83ec-83ed-af05b723ed4c`
  - canonical thread id: `9382ee2cba0c06880b8d351e2055acb49e97a12d`
  - source used: `db`
  - relevant range: stitched lines `16670-16830`
- Supporting local DB context:
  - `Seam Walker Troubleshooting`
    - online UUID: `698d5e21-6d54-839a-a127-088c1dc21227`
    - canonical thread id: `0eff7f41332ca191629d9246ad3677518461fa55`
    - key point: open seams can be valid for plackets/darts/partial openings; cut-graph optimization should include cone/dart support so curvature can be concentrated instead of generating micro-seams.

## BT369 Sphere Unwrap Boundary (2026-06-05)
- Local implementation path: `src/smii/unwrap/sphere_bt369.py`.
- Sibling formal path: `../../dashi_agda/DASHI/Interop/SeaMeInItBT369SphereUnwrap.agda`.
- The rectangle is a serialization view, not the geometry. The source of truth is the equal-area sampled spherical carrier plus BT369 residual trits, 6-sector orientation, ternary prefixes, seam/braid tokens, MDL-bounded depth, and certificate metrics.
- The formal claim is benchmark-gated approximation over declared candidates, not a global isometry or perfect inverse for sphere-to-plane flattening.
- External competitor implementation path: `src/smii/unwrap/external_competitors.py`.
- Competitor docs path: `docs/unwrap_competitor_matrix.md`.
- The measured sphere slice now records receipts for BT369, equal-area,
  equirectangular, cubed-sphere, octahedral, and HEALPix when `healpy` is
  installed. The adversarial field suite adds constant, linear, harmonic,
  polar-cap, seam-stripe, checkerboard, localized-bump, binary-hemisphere, and
  band-limited fields so results can report per-field winners. Mesh/UV solvers
  remain optional diagnostic receipts until real adapters are bound.
- Seam derivation is now framed as an adaptive body atlas compiler:
  body/ROM/fabric evidence projects through a stable basis into fields, fields
  induce seam and panel topology, flattening residuals promote typed
  metric-correction operators, manufacturing allowances serialize the result,
  and `FinishedSeamReceipt` records the final body/ROM/fabric atlas receipt.
- Local runtime path: `src/smii/seams/seam_derivation.py`.
- CLI emission path: `scripts/generate_manufacturing_artifacts.py` with
  `--out-finished-seam-receipt` plus upstream body/ROM/fabric/basis hashes or
  receipt paths and seam-cost/solver/cut-topology/metric-correction receipt
  paths. `scripts/run_afflec_receipted_demo.py` now supplies those paths
  automatically after the cut-topology and metric-correction gates promote.
- JSON schema path: `schemas/finished_seam_receipt.schema.json`.
- Sibling formal path: `../../dashi_agda/DASHI/Interop/SeaMeInItROMSeamAtlas.agda`.
  - `UV unwrapping explanation`
    - online UUID: `6916c180-1080-8320-ae2d-acc2e3ac3c23`
    - canonical thread id: `3562461ee45a9f6eb3b24f0cbd4a233161a7b60e`
    - key point: high-curvature ridges need seams, darts, or controlled stretch zones before LSCM/ABF; projection-only flattening produces starburst artifacts.
- Repo-facing decision:
  - darts, gathers, easing, panel shaping, stretch zoning, variable knit, pleats, gussets, and bias orientation are all typed implementations of controlled metric mismatch injection.
  - A garment panel is not modeled only as a UV map `u : M -> R2`; the useful formal object is `(u, Delta g)`, where `Delta g` is the allowed metric modification field.
  - A dart is a discrete curvature operator: wedge removal in the flat domain that intentionally injects local Gaussian curvature when reassembled.
  - P3 cut-topology work should not merely prune all branch/junction structures until a simple seam graph passes. It should classify them as ordinary cut boundaries, typed dart/relief/gusset/easing operators, or invalid accidental fragmentation before authorizing panel unwrap.
- Follow-up DASHI/Agda cross-check:
  - checked sibling checkout `../../dashi_agda` from this repo.
  - `Docs/SeaMeInItROMKernelFormalism.md` and `DASHI/Interop/SeaMeInItROMKernelFormalism.agda` currently define a theorem-thin receipt surface for `BodyCarrier -> KernelBasis -> ROMOperator -> ProjectedField -> SeamGraph -> SeamCutPanelization -> ManufacturingReceipt`.
  - The current seam formalism is graph/panelization-only: `G = (V, E)`, `S subset E`, and `panels = connected components of G \\ S`.
  - The Agda surface has `SeamGraph`, `SeamCutPanelization`, and strict gates including `topologyGate` and `panelizationGate`, but no `Dart`, `Delta g`, `MetricCorrection`, wedge-removal, or curvature-insertion type yet.
  - `Docs/MeasurementSurfaceProjectionContract.md` and `scripts/hepdata_projection_contract.py` provide the right adjacent pattern: future Delta-bearing projections need declared semantics, metric propagation, explicit failure/degraded states, and no silent theorem-side consumption.
  - Local implication: SeaMeInIt should document and implement a receipt-level metric-correction contract before P3 allows typed dart/relief/gusset/easing structures to authorize unwrap.
- Additional DASHI patterns supplied by the 2026-06-04 review:
  - `DASHI/Physics/Closure/CrossDomainVariationalSpine.agda` defines the closest reusable conceptual shape for metric corrections: a typed variational object with `delta`, `projection`, `defect`, `admissibleGate`, observation quotient, and symmetry boundary.
  - `DASHI/Physics/Closure/HEPDataMeasurementSurfaceProjectionRejection.agda` gives a local pattern for result states and abstention blockers: ok/degraded/rejected/abstained, including missing delta meaning and missing metric propagation law.
  - `DASHI/Foundations/QuotientSetoidSurface.agda` is the strongest reusable foundation for quotient-stable ROM compression, equivalent pose regions, panel equivalence, and seam-cost/norm invariance over quotients.
  - `DASHI/Interop/ObservationTransportSpine.agda` gives the generic observation/transport/non-claim surface for making clear that body or ROM observations do not imply inverse recovery.
  - `DASHI/Metric/FibrePressureMetricBridge.agda` provides residual-budget and candidate-only promotion patterns relevant to seam/fabric coupling debt.
  - `DASHI/Core/UniversalOperatorBasis.agda` supplies join/coordinate-transport vocabulary for future merge layers over seam costs or body-space/ROM-space constraints.
  - `DASHI/Core/AuthorityBoundary.agda` separates citation authority from artifact authority; manufacturing receipts should preserve that boundary.
  - `DASHI/Combinatorics/TriadicVideoCodecObservationQuotient.agda` has side-information-backed reconstruction witnesses and non-promotion certificates useful for ROM compression/hypervoxel side information.
  - Sidecar conventions in DASHI support keeping `MetricCorrectionReceipt` separate from `SeamCutPanelization`: sidecars add extra receipted structure without mutating the core carrier, and remain candidate-only until consumed by later gates.
  - No useful generic garment graph topology library was found; keep connected-component/cut-boundary fields SeaMeInIt-local for now.
  - Preferred local pipeline reading is `SeamGraph -> SeamCutPanelization -> MetricCorrectionReceipt -> PanelUnwrapReceipt -> ManufacturingReceipt`.
  - Ordinary disk-like panels should not pretend to have successful corrections. They either do not require a correction or carry a neutral/non-required state. Branchy/open topology may promote only when required corrections are typed, receipted, admissible, and consumed by panel unwrap.
- `../animalexic` context check:
  - no `.agda` files were found there; it is implementation-first rather than theorem-first.
  - Its docs/runtime define a concrete lattice of `substrate -> candidate -> promoted` with explicit abstain/reject, plus `Candidate<T>`, `Promoted<T>`, `Receipt`, and `InvariantCheck`.
  - Its voxel/surfel guards implement grounded/plateau/ascended states, evidence accumulation, residual thresholds, temporal support, neighbor support, and replayable promotion guards.
  - Conceptual mapping: animalexic `grounded/plateau/ascended` is the runtime analogue of SMII `gateReject/gateDiagnostic/gateAdmissible`, but animalexic solves extrinsic 3D reconstruction while SMII solves intrinsic body-field/seam/metric development into 2D garment topology.
  - Animalexic can feed a stronger candidate/promoted `BodyCarrier`, but it must not bypass SMII gates. Solver seams, darts, panels, and manufacturing outputs remain candidates until SMII receipts promote them.
  - Runtime discipline to copy: kernels emit candidates; host/DASHI owns promotion; abstain by default; never let local numeric kernels mutate canonical promoted state directly.

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

## Receipt Orchestrator Gate 4 Snapshot (2026-05-13)
- Active implementation direction: SMII's receipt DAG is now the production
  promotion contract, not only documentation.
- Gates 0-7 currently emit real artifacts:
  - `BodyCarrierReceipt` from Afflec export checkpoints,
  - `CorrespondenceReceipt` from seam reprojection quality metrics,
  - `BasisReceipt` from canonical basis generation on a promoted carrier,
  - `ROMFieldReceipt` from sample aggregation with field-uniformity gating,
  - `SeamCostReceipt` from promoted body/field receipts and native-or-promoted
    solve-domain gating,
  - `SolverPromotionReceipt` from promoted seam costs with field-minima anchor
    provenance, fallback visibility, seam hashes, and panel-topology gating,
  - `PanelUnwrapReceipt` from promoted solver topology with per-panel
    distortion, grain direction, UV hashes, and explicit flattening
    non-promotion boundaries,
  - `ManufacturingReceipt` from promoted panel unwrap artifacts with
    hash-matched UVs, variable seam allowance fields, cutting-layout artifacts,
    notches, labels, method accessibility, and final promotion state.
- Remaining independent follow-up lanes:
  - orchestrator task runner,
  - ROM environment unblock,
  - Ruff integration.

## Receipt Chain Production Roadmap (2026-05-14)
- Added the local roadmap and runner for turning the seven manual demo commands
  into a single Afflec receipted run:
  `scripts/run_afflec_receipted_demo.py`.
- The runner executes the native `A_v3240` receipt path through cut topology,
  metric correction, panel unwrap, manufacturing, and finished seam receipt
  emission; it stops at first non-promoted receipt, emits `run_manifest.json`,
  keeps dry runs artifact-free, and accepts `--images` for curated Afflec/P3
  reference-image validation.
- Latest validation: bundled three-image MediaPipe path still blocks at Gate 0
  due skull residual / measurement-refinement quality. The curated seven-image
  P3 path promotes through Gate 5c (`BodyCarrierReceipt`, `BasisReceipt`,
  `ROMFieldReceipt`, `SeamCostReceipt`, `SolverPromotionReceipt`,
  `CutTopologyReceipt`, and `MetricCorrectionReceipt`) and blocks at Gate 6 on
  panel unwrap distortion/corrected residual budget. Next work is improving
  panelization/unwrap/correction quality, not loosening thresholds or adding
  missing receipt plumbing.
- The runner should default Gate 0 to `--detector mediapipe`, retain
  `--detector bbox` as an explicit diagnostic fallback, and forward
  `--require-high-trust-detector` for runs that should hard-fail coarse
  detector fallback.
- MediaPipe runtime checks completed locally: Holistic initialisation and
  single-image inference work, and direct Gate 0 execution emits artifacts
  under `outputs/demo/mp_cpu_test`. The receipt remains non-promoted because
  measurement refinement pushes final betas to `max_abs=11.84` and leaves
  `skull_rigidity_residual=0.3602` above the conservative `0.35` threshold.
- Gate 1 correspondence is intentionally skipped in this native demo path; it
  remains required for transfer-backed solve domains.
- Upgrade sequence after the runner:
  1. real LSCM/ABF/ARAP unwrap,
  2. pattern exporter wiring for true cut/stitch geometry,
  3. real ROM corpus fields,
  4. cotangent Laplace-Beltrami basis,
  5. production manufacturing package and Ruff.

## Gate 6c Pareto/Profile Boundary (2026-06-06)
- Parent-surgery diagnostics are now being extended with Pareto metadata:
  each candidate should report which hard metrics it improves, whether it sits
  on the frontier against sibling variants, and which profile would select it.
- The default acceptance rule remains unchanged:
  backend-serializable, score-improving, and non-regressing in worst distortion.
- Profile-relative scoring is diagnostic only for now; it should not silently
  override the scalar gate or the existing failure-field receipts.
- The next geometry family is `failure_wedge_relief`: a two-leg wedge/lens
  split derived from measured serializer failure fields. It is a fabric
  repartition, not fabric deletion, and is competed through the same scalar gate
  plus Pareto/profile diagnostics.
- Non-selected variants now need compact default-profile loss explanations
  (`lost_to_*` and metric deltas) so dominated variants and Pareto-useful but
  too-costly variants remain distinguishable in receipts.
- Empirical wedge result:
  `outputs/demo/afflec_receipted_curated_20260605_143059/panels_fabric_wedge2/panel_unwrap_receipt.json`
  keeps accepted parents at P1 `cutout_r1` and P3 `relief_split`; the bounded
  P1 `failure_wedge_relief` candidate is valid but dominated, while P0/P2 do
  not get wedge candidates from the current two-leg corridor generator.
- Guided dart-wedge result:
  `outputs/demo/afflec_receipted_curated_20260605_143059/panels_fabric_guided_dart/panel_unwrap_receipt.json`
  adds `pareto_guided_dart_wedge`, derived from the measured
  `failure_relief_path` rather than generic sinks. P2 receives a valid
  Pareto-frontier useful guided candidate (`score=10897.938`,
  `worst_distortion=5.241`, `charts=2`) that improves original worst
  distortion but still loses default-profile selection to `cutout_r2` and does
  not promote. Accepted parents remain P1 `cutout_r1` and P3 `relief_split`.
- Current next rung: `failure_lens_patch`. Path-family operators
  (`failure_relief_path`, drain/tree, wedge, and guided dart wedge) are now
  empirically insufficient for P0/P2 under the default hard gate. The open
  patch-family test is to isolate the measured failure support as a bounded
  replacement patch chart plus parent remainder, preserving all faces and
  competing under the unchanged serializer/score/distortion gate.
- Lens-patch rerun result:
  `outputs/demo/afflec_receipted_curated_20260605_143059/panels_fabric_lens_patch/panel_unwrap_receipt.json`.
  Accepted parents remain P1 `cutout_r1` and P3 `relief_split`; P0 gets a
  valid `failure_lens_patch` that improves score/foldovers but regresses worst
  distortion to `16.176`; P2 gets no bounded lens candidate. Current boundary:
  bounded patch repartition is still insufficient, so the next open family is
  true `gusset_parent_replacement` / broader operator-native patch geometry.
- Operator-basis search is now the next implemented diagnostic surface:
  materialized parent-domain decisions emit `smii.operator_basis_search.v1`,
  a depth-2/beam-8 search receipt over measured single-operator metric deltas.
  It can report profile/Pareto tree winners or declare the current local
  operator basis exhausted for a panel, but it does not claim true sequential
  tree rematerialization. That keeps native operator-tree materialization and
  the BT369 pattern serializer as explicit next rungs rather than hidden
  assumptions.
- Operator-basis-search Afflec/P3 rerun:
  `outputs/demo/afflec_receipted_curated_20260605_143059/panels_fabric_operator_basis_search/panel_unwrap_receipt.json`.
  Accepted parents remain P1 `cutout_r1` and P3 `relief_split`; P0 and P2 both
  report `basis_exhausted_at_depth=true` at depth 2. P2's retained diagnostic
  trees keep the useful relief/dart geometry evidence, but none can promote
  because measured singles still miss the hard gate and composed trees are not
  sequentially materialized.
