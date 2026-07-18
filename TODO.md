# TODO

Current planning entrypoint: `docs/roadmap/index.md`.
Current audit summary: `docs/roadmap/audit_findings_20260604.md`.
Afflec local reference-image manifest: `docs/roadmap/afflec_reference_images_20260604.md`.

## Active P0

- Make the documented quality gates reproducible:
  - define or correct `dev` and `test` extras for `pip install -e .[dev,test]`,
    or document the sibling ITIR venv (`../.venv`) as the required local
    runtime,
  - ensure `../.venv/bin/python -m pytest --maxfail=1 -q` remains clean,
  - declare Ruff, mypy, Hypothesis, pytest-mock, `jsonschema`, and other
    required test/dev dependencies in the chosen dependency surface.
- Resolve repository governance blockers before adding new generated artifacts:
  - reconcile README license claims with `LICENSE`,
  - audit tracked binary/generated files and replace generated outputs with
    regeneration commands,
  - decide fixture policy for image/body-derived assets,
  - ignore or relocate default `exports/` outputs.
- The 2026-06-04 Afflec reference images are an owner-approved manual binary
  exception for the calibration lane. Agents must not commit the binaries, but
  the project owner may force-add `assets/reference_images/afflec/` using the
  command block in `docs/roadmap/afflec_reference_images_20260604.md`.
- Keep Gate 0 body-trust calibration as the first production pipeline blocker.
  Reference-quality warnings and export checkpoints already exist; next:
  normalize the effective measurement-model authority, thread image-derived
  betas into bounded/anchored refinement, recompute diagnostics from the
  bounded candidate, and emit a hash-linked refinement receipt with
  `promote`/`abstain`/`reject`. Add `BodyCarrierReceipt` v2 lineage with an
  explicit canonical source. Diagnostic warnings must remain visible without
  implicitly becoming a universal authorization veto. Freeze smoke abstention
  and curated-P3 promotion tests. See
  `docs/roadmap/gate0_reference_and_p3_handoff.md`.
- P3 expanded Afflec evidence is recorded in
  `docs/roadmap/p3_afflec_gate0_evidence_20260604.md`: the curated seven-image
  MediaPipe lane promotes under the current skull threshold, while all-ref
  lanes fail on no-pose inputs. Proceed assuming the refined final-export
  topology change to `9384` vertices / `18764` faces is acceptable for now;
  add an export/topology diagnostic later before treating topology mutation as a
  quality blocker.
- Define the P3 back-transfer spec against the exact forward object and cutting
  constraints before replacing bootstrap unwrap/manufacturing behavior:
  topology/hashes, transfer mode, round-trip checks, seam retention/collapse,
  collision/load metrics, morphology preservation, grain/notch/label/cut-line
  promotion gates.
- P3 garment metric-correction/topology-gated unwrap is implemented locally:
  `CutTopologyReceipt` classifies ordinary boundaries, correction-tree/operator
  nodes, and invalid fragmentation; branches are not intrinsically blockers
  once typed. `MetricCorrectionReceipt` records selected corrections, residual
  gates, blockers, source hashes, and `correctionOk` /
  `correctionDegraded` / `correctionRejected` / `correctionAbstained`;
  schema guidance names `CorrectionTreeReceipt` fields such as
  `operator_nodes`, `operator_type`, `delta_metric_meaning`, and
  `metric_propagation_law`, plus `FabricAwarePanelMetricReceipt` fields such as
  `fabric_receipt_hash`, `grain_direction`, `fabric_relative_threshold`, and
  `fabric_metric_gate`. `scripts/emit_metric_correction_receipt.py` emits the
  bridge receipt; `scripts/unwrap_panels.py` requires a promoted metric receipt
  when typed operators are present; panel promotion is fabric-relative;
  `FinishedSeamReceipt` now composes promoted
  body/ROM/fabric/basis/seam/panel/correction/manufacturing receipts into a
  final adaptive body atlas serialization receipt; and
  `scripts/generate_manufacturing_artifacts.py` can emit
  `finished_seam_receipt.json` from supplied upstream receipts. The Afflec
  receipted demo runner now includes cut-topology and metric-correction gates
  and passes the finished-receipt arguments into Gate 7. Curated P3/Afflec
  validation now promotes through Gate 5c and blocks at Gate 6 because bootstrap
  panel distortion is ~0.21 against the 0.05 threshold and corrected residuals
  are ~0.15 against the 0.05 gate. Next, improve panelization/unwrap/correction
  quality rather than loosening the gate.
- Gate 6c parent surgery now records serializer-derived failure fields inside
  `serialization_competition_receipt.failure_fields` and offers a measured
  `failure_relief_path` variant when high-distortion or foldover faces form a
  strict face subset. This keeps the acceptance lemma unchanged
  (serializable, score-improving, no worst-distortion regression) while making
  the next parent split come from measured residual/foldover evidence instead of
  assuming branch-local isolation is sufficient.
- Fresh P3/Afflec evidence is recorded at
  `outputs/p3_afflec_failure_relief_20260605/panel_unwrap_fabric_full/panel_unwrap_receipt.json`:
  the full fabric/materialization-aware run emitted parent failure fields and
  competed `failure_relief_path`. P2 improved worst distortion (`5.41` to
  `4.97`) but regressed score, so it was correctly rejected. P0 reduced score
  but produced invalid chart domains. Accepted parents remain P1 `cutout_r1`
  and P3 `relief_split`; next parent surgery should be a stronger
  backend-valid multi-path relief tree rather than another branch-local split.
- Follow-up evidence is recorded at
  `outputs/p3_afflec_failure_relief_20260605/panel_unwrap_fabric_tree/panel_unwrap_receipt.json`:
  the materialized competition now emits a real `failure_relief_tree` variant
  alongside the older single-path family. It is still rejected on the current
  Afflec panels, but the receipt shows the next shape of the search space is now
  explicit rather than implied.
- The next parent-surgery rung is `failure_drain_path` / `failure_drain_tree`:
  route measured failure islands toward a release boundary instead of only
  separating bad components from the parent remainder, then rerun the same
  acceptance gate. This is the current implementation target for improving
  P0/P2.
- The first Afflec rerun with `failure_drain_*` is recorded at
  `outputs/p3_afflec_failure_relief_20260605/panel_unwrap_fabric_drain/panel_unwrap_receipt.json`:
  P1 now exposes a real `failure_drain_tree` candidate, but it is still
  rejected; P2 still prefers the lower-distortion `failure_relief_path` shape
  yet remains score-rejected; P0 does not gain a drain candidate from the
  current failure field. The acceptance gate is still honest, so the next
  improvement has to come from better drain geometry or better global scoring.
- The tightened single-path drain selection is recorded at
  `outputs/p3_afflec_failure_relief_20260605/panel_unwrap_fabric_drain3/panel_unwrap_receipt.json`:
  the selected drain path now prefers the seam boundary, but the best P1 drain
  variant still lands just above the original score and remains correctly
  rejected. The next gain has to come from a genuinely cheaper corridor, not
  another acceptance tweak.
- Add Pareto/profile diagnostics to the Gate 6c parent-surgery receipt:
  record per-variant improvement dimensions, Pareto-frontier membership, and
  profile-relative winner selection so a candidate that loses on the default
  scalar score still carries an explicit "useful but not promoted" receipt.
  Keep the existing acceptance gate unchanged.
- Add `failure_wedge_relief` as the next parent-surgery operator family:
  route two low-cost legs from a measured failure component toward panel or
  seam-boundary sinks, split a wedge/lens chart without deleting parent faces,
  and compete it under the existing backend/score/distortion gate. The receipt
  should also explain default-profile losses with `lost_to_*` and metric-delta
  diagnostics so P2-style Pareto-useful variants are not collapsed into a
  generic rejection.
- Bounded wedge evidence is recorded at
  `outputs/demo/afflec_receipted_curated_20260605_143059/panels_fabric_wedge2/panel_unwrap_receipt.json`:
  the chart-count guard prevents pathological wedge fragmentation. P1 receives
  a valid `failure_wedge_relief` candidate, but it is dominated. P0/P2 do not
  receive wedge candidates from the current two-leg corridor generator, so P2
  remains the useful single-relief-path case and the next geometry needs a
  stronger apex/sink pair generator rather than a looser gate.
- Guided dart-wedge evidence is recorded at
  `outputs/demo/afflec_receipted_curated_20260605_143059/panels_fabric_guided_dart/panel_unwrap_receipt.json`:
  `pareto_guided_dart_wedge` uses the measured `failure_relief_path` as guide
  evidence, materializes only a narrow apex-plus-two-leg dart/lens chart, and
  competes under the unchanged gate. P2 receives a valid Pareto-frontier useful
  candidate (`score=10897.938`, `worst_distortion=5.241`, `charts=2`), but it
  still loses default-profile selection to `cutout_r2` and does not promote.
  Accepted parents remain P1 `cutout_r1` and P3 `relief_split`.
- Implement `failure_lens_patch` as the first patch-family parent operator:
  derive a bounded support region from serializer failure components and
  Pareto-useful relief evidence, split it into a replacement patch chart plus
  parent remainder charts without face loss, run normal backend competition,
  and record Pareto/profile diagnostics. Keep the existing hard acceptance gate
  unchanged.
- Lens-patch evidence is recorded at
  `outputs/demo/afflec_receipted_curated_20260605_143059/panels_fabric_lens_patch/panel_unwrap_receipt.json`:
  P0 gets a valid `failure_lens_patch` that improves score/foldovers but
  regresses worst distortion (`16.176`) and is dominated by `relief_split`;
  P1 gets lens candidates but still accepts `cutout_r1`; P2 gets no bounded
  lens candidate from the current support generator. Next patch-family work
  needs true `gusset_parent_replacement` or a broader support generator rather
  than another bounded path/lens repartition.
- Implement `OperatorBasisSearchReceipt` as the next diagnostic surface:
  define a small operator basis from the measured parent-surgery variants,
  compose depth-2/beam-8 operator trees from the single-operator metric deltas,
  retain default/profile/Pareto winners, and mark whether the declared basis is
  exhausted for hard panels. This first version must be explicit that composed
  trees are diagnostic beam candidates, not true sequential rematerializations.
- Operator-basis-search v1 evidence is recorded at
  `outputs/demo/afflec_receipted_curated_20260605_143059/panels_fabric_operator_basis_search/panel_unwrap_receipt.json`:
  P0/P2 have no promoted measured single operation; the retained depth-two
  unordered cross-family delta combinations are diagnostic-only and do not
  establish sequential exhaustion. P1/P3 retain their existing single-operation
  promotions. Next implement `smii.operator_basis_search.v2`: materialize
  ordered residual-conditioned transitions, rerun the actual backends, retain
  search-useful children separately from hard-gate promotion, and bind any
  bounded no-result conclusion to generator/backend/policy/beam/deduplication
  hashes. Diagnose expressivity, serialization, panelization, and physical or
  policy failure before starting `bt369_pattern_serializer`.
- Graph/ultrametric unwrap scoring now has a local benchmark surface:
  `smii.seams.unwrap_benchmark` compares sphere-to-rectangle candidates across
  edge-length, area, angle, foldover, aggregate residual, and agreement-depth
  metrics. This is the formal ranking layer above numerical backends such as
  LSCM; it does not claim an isometric sphere-to-rectangle flattening.
- BT369 sphere serialization is implemented in `smii.unwrap.sphere_bt369`:
  `unwrap_sphere_bt369` samples via equal-area inverse pullback, records
  triadic cell prefixes, residual trits, 6-sector tangent orientation, seam
  tokens, MDL-bounded depth, and an export certificate. Next, replace the
  current deterministic triadic address helper with a real geodesic icosahedral
  carrier when production sphere fields need transport across assets.
- External unwrap competitor receipts are implemented in
  `smii.unwrap.external_competitors`: the sphere slice now measures BT369,
  equal-area, equirectangular, cubed-sphere, octahedral, and HEALPix when
  `healpy` is installed; it also runs an adversarial field suite over smooth,
  localized, seam-crossing, and discontinuous fields. Next, add xatlas and
  libigl LSCM/SLIM adapters behind the same receipt boundary.

## Existing Backlog

- R0. Build the receipt orchestrator as a promotion DAG, not a task list:
  `BodyCarrierReceipt -> Transform/CorrespondenceReceipt -> BasisReceipt ->
  ROMFieldReceipt -> SeamCostReceipt -> SolverPromotionReceipt ->
  CutTopologyReceipt -> MetricCorrectionReceipt -> PanelUnwrapReceipt ->
  ManufacturingReceipt -> FinishedSeamReceipt`.
  - R0.1 Carrier trust: `BodyCarrierReceipt` is implemented and
    `generate_undersuit` now blocks unpromoted carriers before artifact
    emission; `afflec-demo` now emits export-stage checkpoints plus
    `body_carrier_receipt.json` from the body fit/export path. Next, tighten
    skull/head plausibility thresholds against real Afflec runs.
  - R0.2 Correspondence: `TransformReceipt` / `CorrespondenceReceipt` is
    implemented with source/target hashes, residual metrics,
    collision/load metrics, retention, edge loss, explicit downstream blocks,
    and `A_T`; next, wire it to sampler-native correspondence export. Seam
    reprojection should emit receipts with separate full-surface load
    (`collision_ratio`) and seam-local collapse (`seam_transfer_collapse`) so
    transfer failures remain traceable instead of reintroducing metric-regime
    confusion through code.
  - R0.3 Field basis: `BasisReceipt` is implemented for canonical `B_0`
    provenance on a promoted physical carrier; `scripts/generate_canonical_basis.py`
    now emits `basis_receipt.json` when given a promoted body-carrier receipt
    and hard-blocks non-promoted carriers. Next, replace the bootstrap
    sinusoidal-QR basis with a cotangent Laplace-Beltrami eigenbasis and wire
    ROM field aggregation to require the promoted receipt.
  - R0.3a Orchestrator reader: `smii.orchestrator.read_receipt_dag` reads
    body/correspondence/basis/ROM-field/seam-cost/solver receipts from a run directory
    and reports the first blocker plus seam-solver eligibility. Next, wire
    existing CLIs to consult this reader before promotion.
  - R0.4 ROM aggregation: `ROMFieldReceipt` is wired into
    `examples/rom_aggregate_from_samples.py` for vertex-aligned ROM field
    summaries, with `field_uniformity` gating and synthetic promotion requiring
    an explicit flag; Gate 4 now consumes promoted `ROMFieldReceipt` before
    solver promotion.
  - R0.5 Topology/solver: `SeamCostReceipt` is implemented and
    `scripts/compute_seam_costs.py` emits `seam_cost_receipt.json` only after
    enforcing `A_body=+1`, `A_field=+1`, and either `A_T=+1` or
    `solve_domain=A_v3240`; `cost_uniformity` and finite coverage block flat or
    incomplete costs. `SolverPromotionReceipt` is implemented and
    `scripts/solve_seams.py` emits `solver_promotion_receipt.json` only from a
    promoted seam-cost receipt; it records field-minima anchors, component
    fallback usage, seam hashes, and panel-topology promotion state. Next, wire
    the legacy solver examples to consume the solver receipt when promoting.
  - R0.6 Panel/manufacture: `CutTopologyReceipt`, `MetricCorrectionReceipt`,
    and `PanelUnwrapReceipt` are implemented. `scripts/unwrap_panels.py` emits
    `panel_unwrap_receipt.json` only from promoted solver/cut topology, requires
    promoted metric correction when typed operators are present, and now supports
    real NumPy LSCM in addition to the bootstrap projection. Distortion and grain
    direction are receipted per panel, failed flattening produces an explicit
    non-promotion boundary. `smii.seams.correction_operator_scoring` now prices
    unresolved branch nodes as candidate darts/gussets/ease/stretch/grain/seam
    operators before final fabric-relative metric gating; it is an estimator
    receipt, not a cloth-simulation authority. Priced `stretch_zone` operators
    now realize as local fabric-cone overrides plus diagnostic cut-sheet
    annotations, and `gusset_corner` companion operators realize residual relief
    when stretch alone cannot satisfy the metric gate; true polygon geometry for
    dart/ease/grain/seam remains future work.
    `smii.seams.unwrap_benchmark`
    now provides the graph/ultrametric multi-metric comparison surface used to
    rank rectangle unwrap strategies before treating any numerical backend as
    promotable.
    `ManufacturingReceipt` is implemented and
    `scripts/generate_manufacturing_artifacts.py` emits final manufacturing
    receipts only from promoted panel unwrap artifacts with hash-matched UVs.
    Variable seam allowance fields, method accessibility, notches, labels, and
    cutting-layout hashes are receipted; flat allowance remains an explicit
    non-promotion diagnostic. Next, add an orchestrator task runner for the
    complete Gate 0-7 chain.
  - R0.7 Afflec receipted demo runner: `scripts/run_afflec_receipted_demo.py`
    now executes the native `A_v3240` receipt chain through cut topology,
    metric correction, panel unwrap, manufacturing, and
    `finished_seam_receipt.json`, stops at first non-promotion, emits
    `run_manifest.json`, and keeps `--dry-run` artifact-free. It defaults Gate
    0 to `--detector mediapipe`, exposes `--detector bbox` only as an explicit
    coarse diagnostic mode, forwards `--require-high-trust-detector` when a
    run should hard-fail detector fallback, and accepts `--images` for the
    curated P3 reference set. Do not commit generated demo outputs; regenerate
    them locally with the runner command. A bounded
    MediaPipe Gate 0 run completed under
    `outputs/demo/mp_cpu_test`: raw regression stayed plausible
    (`beta_max_abs=1.95`, high-trust detector), but measurement refinement
    pushed final betas to `max_abs=11.84` and the skull residual remained just
    above threshold (`0.3602 > 0.35`), so the receipt correctly stayed
    diagnostic-only. The curated seven-image P3 lane promotes Gate 0 and the
    full runner now promotes through Gate 5c before Gate 6 blocks on real panel
    distortion/correction budgets.
- Production upgrade roadmap after the receipt chain:
  1. validate Gate 6 graph/ultrametric unwrap scoring on P3/Afflec runs, then
     use it to compare LSCM/ABF/ARAP backend candidates,
  2. wire Gate 7 to the pattern exporter for SVG/PDF/DXF cut and stitch lines,
  3. replace synthetic Gate 3 fields with `sampler_real` corpus aggregation,
  4. replace the Gate 2 sinusoidal/QR basis with cotangent Laplace-Beltrami
     modes plus atlas weighting,
  5. finish cutter-readiness checks and add Ruff to the tooling gate.
- M1. Backfill morphology observations into run roots via `morphology_observations.json` so run-reference pages can state which artifacts are neutral-human, ogre-like, or flailing instead of defaulting to `unclassified` / `inherits_source_geometry`.
- M2. Add explicit ROM sample morphology outputs so the pipeline can show where flailing occurs: keep the neutral-body operator field, but also emit representative posed/deformed ROM sample artifacts rather than forcing users to infer morphology from seam heatmaps on a neutral body.
  - M2.1 define representative sample selection policy
    - fixed anchors: `max_field_l2_norm`, `max_displacement_mean_norm`, `median_field_l2_norm`, `max_weight`
    - default sample count: `4`
    - fill remaining slots by descending `field_l2_norm` after deduplication
  - M2.2 define artifact contract for sample meshes/renders/metadata
    - emit sampler-native posed mesh `.npz` files plus `rom_sample_manifest.json`
    - keep native topology; do not pretend these are inverse-mapped back to the fitted body
  - M2.3 emit minimal representative posed/deformed sample artifacts
    - baseline complete: `smii.rom.sampler_real --out-rom-samples-dir`
  - M2.4 surface those artifacts in run pages and operator reports
    - baseline complete: run pages auto-classify `rom_sample_pose`; operator report accepts `--sample-manifest`
- M2b. Resolve the inverse/back-transfer ambiguity: document whether the current "solve on ogre/internalized domain, then return to fitted body" step is only approximate correspondence/reprojection or a true inverse transform; if no true inverse exists, define the acceptance contract for the approximation explicitly.
  - M2b.1 audit current code paths for an actual inverse candidate
    - current audit says no geometry inverse exists; only basis-space encoding and mesh correspondence/reprojection
  - M2b.2 record current reality as correspondence/reprojection vs inverse
    - complete in docs/contract notes
  - M2b.3 define acceptance criteria for approximate transfer
    - require lineage, explicit transfer labeling, and quality metrics/gates
  - M2b.4 define what evidence would be required for a future true inverse
- M3. Use `outputs/comparisons/afflec_kernel_diagnostic_raw_20260310/` as the current reference for ROM-kernel interpretation and decide whether `seam_sensitivity = sum_j w_j |disp_hat · dV/dtheta_j|^2` is actually the design signal we want, or whether we should expose an alternate seam objective based on pure displacement magnitude and/or derivative magnitude instead of only the motion-direction-gated field.
- M4. Investigate shortest-path seam insensitivity under fixed topology: the same-topology comparisons in `outputs/comparisons/afflec_same_topology_20260309/` produced identical seam edge sets even after removing MDL and increasing ROM weights, while seam reports still warn `anchors disconnected; using largest connected component anchors`; next step is debugging anchor/component fallback and comparing shortest-path against `mincut`/`pda` on the same cost pairs.
- M5. Audit the exact historical forward object behind `B_ogre` before doing any more inverse/back-transfer design work.
  - M5.1 collect the historical seam/render/mesh artifacts for `B_ogre` — complete
  - M5.2 identify the exact mesh path/hash/topology behind `B_ogre` — complete
    - current forward object: `outputs/suits/afflec_body/base_layer.npz` (`9438`, SHA256 `b122dc2cf8b075a5a5bcc0c124a075247268332203df7873c36de65e4027695c`)
    - paired ROM costs: `outputs/rom/seam_costs_afflec_realshape_edges.npz` (`9438`, SHA256 `750e0472648fff6a4f324cd4b34e78648dd8c878a1b2acbd85a0c3a3c57f50d8`)
  - M5.3 determine whether `B_ogre` was real transformed geometry or mainly a render/domain artifact
    - narrowed but not closed: native `B_ogre` is a real `9438` solve object, but old ogre-like visuals still appear mixed with transfer/render artifacts
  - M5.4 record the forward-object decision note — complete in `docs/b_ogre_and_afflec_crown_audit_20260311.md`
- M6. Diagnose the current Afflec crown/head-shape failure.
  - M6.1 reproduce the egg-shaped / Green-Goblin-like skull crown on the latest current Afflec run — complete
  - M6.2 compare raw reprojection fit, refined fit, and repaired/exported mesh — partially complete
    - diagnostics/params are stable between calibrated `complete2` and `complete3`, but the repo still lacks enough pre-export mesh checkpoints
  - M6.3 decide whether the distortion comes from sparse input views, reprojection optimization, refinement, or repair/export
    - narrowed to late mesh generation / repair / export, or a latent fitted-geometry issue only visible there
  - M6.4 define a skull/head plausibility acceptance gate
    - next bounded implementation target
- M7. Define the Afflec-facing back-transfer requirement around the exact forward object from M5, not around generic nearest-neighbor reprojection.
  - M7.1 decide true inverse vs strict approximate transfer requirement
  - M7.2 define topology, round-trip, retention, collision, and morphology-preservation criteria
  - M7.3 design the implementation track tied to that exact forward object
- M8. Add a video-input fitting path, but only after M6 defines why the current photo path fails.
  - M8.1 define accepted video/frame workflow
  - M8.2 define frame selection/aggregation policy
  - M8.3 define provenance contract for video-derived fits
  - M8.4 implement CLI/path later
- M9. Audit SMPL-X body-shape coverage, especially feminine-body fitting behavior.
  - M9.1 audit current provider/config/model-selection behavior
  - M9.2 document current assumptions and limitations
  - M9.3 define intended fit-policy expectations for feminine bodies
  - M9.4 add fixtures/tests later
- Milestone framing: keep the product roadmap explicit as
  1. sewable fitted bodysuit,
  2. thermal/heat-distribution routing and cooling-loop integration,
  3. comfortable system packaging,
  4. later "iron man" hard-function modules.
- Calibrate the Afflec measurement-refinement handoff against the MediaPipe
  diagnostics outputs: MediaPipe raw regression is plausible and high-trust,
  but refinement still creates a large beta shift and leaves
  `skull_rigidity_residual` marginally above the promotion threshold. The
  diagnostic/coarse `bbox` fallback remains useful for smoke checks, but it is
  not the high-trust receipted demo default.
- Thread `afflec_fit_diagnostics.json` status into downstream body/ROM/seam manifests so low-trust Afflec runs are visibly marked outside the body-fit stage.
- Stop emitting report-generated analytical PNGs from `render_rom_operator_report.py`; render coefficient/norm summaries directly as DOM/SVG in `index.html`, and make the report page embed existing topology media artifacts (`overlay.png`, flex heatmaps, GIF/WebM orbits, map videos) from supplied paths/directories.
- Add canonical run reference pages and a runs index:
  - new run page should embed all completed assets under one run root (body, ROM, seams, overlays, heatmaps, GIF/WebM orbits, maps) and link to specialized subpages like `rom_operator/index.html`,
  - Strategy 2 bundles should emit this run page automatically and refresh both per-root indexes and one unified `outputs/index.html` catalog with timestamps and run-type labels,
  - ignore transient `*_frames*` directories and legacy operator-report chart PNGs in galleries.
- Make `render_seam_orbit.py` delete its temporary frame PNG directory after GIF/WebM encoding, matching other orbit renderers; keep deliberate still outputs such as `overlay.png` and front-view PNGs.
- Extend auto-split strategies (multi-cut, seam-aware) and propagate child-specific issues.
- Expose outline cleanup parameters (outlier threshold, smoothing iterations, simplify tolerance)
  in the pattern export CLI so garment makers can tune output fidelity.
- Wire PDA coupling manifest + gate decisions into ROM sampler/export paths with rejection logging (aggregation now emits structured gate ids/reasons).
- Expand ROM aggregation diagnostics (visuals, hotspot overlays) and connect to seam validators; demo stubs live at `examples/rom_hotspot_diagnostic.py` and `examples/rom_aggregate_from_samples.py`.
- Add output no-op verification for ROM/heatmap runs: log mtimes + content hashes for `afflec_body.npz`, `seam_costs`, heatmaps, and fitted params; warn when outputs are unchanged.
- Install/verify SMPL-X runtime extras (`smplx`, torch backend) in the active venv so `python -m smii.rom.sampler_real` can regenerate ROM artifacts (currently blocked by `ModuleNotFoundError: smplx` in this environment).
- Add runtime performance attribution: detect GPU vs CPU heavy compute paths and flag when a claimed GPU-assisted run is actually CPU-bound (log + metrics).
- Add an explicit mesh-lineage audit CLI (ingest -> body -> ROM -> seam report -> reprojection) that emits vertex counts, hashes, and mismatch flags as JSON/CSV.
- Persist body provenance in undersuit metadata (`body_vertex_count`, `body_face_count`, body hash) so historical `outputs/suits/*` runs can be disambiguated from later `outputs/afflec_demo/*` regenerations.
- Add a mesh registry/labeling helper that emits stable labels (`subject`, `stage`, `topology`, `body_hash`) and optional alias symlinks for human-friendly browsing (do not rename/overwrite in-place).
- Fix render-axis instability in `scripts/render_variant_orbits.py`: stop inferring “up axis” from max span; add explicit axis convention flags and record the convention in the render manifest.
- Add a required “vertex-map orbit” renderer for correspondence debugging (target colored by map distance; optional source overlay + subsampled correspondence lines) so claims about collision/retention have viewable artifacts.
- Record mesh-edge conformance metrics in every seam report (native and transferred) so “single scapula line” failures can be attributed to solver vs transfer.
- Regenerate and timestamp canonical basis artifacts with source-path metadata (do not overwrite ambiguous legacy files without an accompanying lineage manifest).
- Add reprojection quality gates (`max_distance`, `mean_distance`, edge-retention ratio) so seam transfer fails loudly when source/target meshes are not geometrically compatible.
- Add stage-level edge divergence report (`source edges` -> `mapped edges` -> `collapsed` -> `deduped`) to simplify root-cause triage for cross-topology seams.
- Make seam reprojection consume sampler-native `--out-correspondence` artifacts by default (and only fall back to ad-hoc NN map generation when missing).
- Add transform-native correspondence export for the ogre domain (`9438`) itself; current sampler-native map is `10475 -> 3240` and cannot be applied directly to `B_ogre (9438) -> base (3240)` seam transfer.
- Add export-stage Afflec body checkpoints (`raw reprojection mesh`, `refined pre-repair mesh`, `repaired/export-ready mesh`) so the skull/crown distortion can be localized before any further fit-policy or video-input changes.
- Decide canonical seam solve domain and freeze policy:
  - Option A: solve on `afflec_body` then evaluate/project to ROM.
  - Option B: solve on ROM/ogre domain then reproject to base.
  - Document acceptance metrics and a hard pass/fail checkpoint.
- Add per-run lineage manifest requirement (`body`, `rom_costs`, `solver_input_mesh`, `render_mesh`, vertex counts, hashes) and reject runs missing it.
- Replace semantic run labels (`A_base`, `B_ogre`) with topology-explicit labels (`A_v3240`, `B_v9438`, etc.) in protocol outputs to avoid morphology/name inversion confusion.
- Enforce artifact naming policy: all orbits/maps/reports must include `human|ogre` role + `vNNNN` topology tag in filename stems (see `docs/seam_pipeline_intended_vs_observed.md`).
- Enforce render orientation invariant: within a bundle, human+ogre must face the same direction.
  - Make `--axis-up auto` PCA-based (avoid raw axis-span heuristics; see `docs/seam_overlay_orientation.md`).
  - In vertex-map orbits, align source->target visualization in a shared canonical frame to avoid mirror/flip confusion.
  - Use that policy by default in `scripts/protocol_strategy2_bundle.py` and record it in manifests.
- Add a small "role registry" helper: given a mesh path, print `{role_guess, vertex_count, face_count, sha256}` and allow overriding role via CLI flags for protocol runs.
- Stop inferring `human|ogre` role from vertex counts in protocol scripts; require explicit `--base-role`/`--rom-role` and treat missing roles as an error (identity comes from provenance).
- Rename render flag `--canonicalize` to `--normalize-rotation` (keep `--canonicalize` as a deprecated alias for now) to avoid confusion with ROM/domain canonicalization.
- Run multi-loop seam solver (`shortest_path --sp-require-loop --sp-loop-count>=2`) on the `human` (`v3240`) morphology and persist a timestamped orbit + report.
- Improve strict loop feasibility diagnostics/actionability: per-panel loop-feasibility score and clearer reasons for `no path`/`loop closure unavailable` under `--sp-loop-strict`.
- Execute and record the A-vs-B protocol defined in `docs/seam_pipeline_intended_vs_observed.md` and freeze canonical solve policy (A or B) with dated decision rationale.
- Add quantitative A/B comparison metrics to Strategy 2 bundles and use them as acceptance gates (edge retention/collision, mesh-edge validity, length collapse).
- Regenerate canonical ROM basis via `python scripts/generate_canonical_basis.py --vertices <production mesh npy/npz> --body-receipt <run-root>/body_carrier_receipt.json --components <K> --harmonics 5 --output outputs/rom/canonical_basis.npz --receipt-output outputs/rom/basis_receipt.json`
  (do not commit the resulting NPZ; keep outputs/rom/ ignored), then run the
  sampler aggregator and Gate 4 cost emitter with real payloads:
  `PYTHONPATH=src python examples/rom_aggregate_from_samples.py --samples outputs/rom/afflec_sampler.json --basis outputs/rom/canonical_basis.npz --basis-receipt outputs/rom/basis_receipt.json --out-rom-fields outputs/rom/rom_fields.npz --out-rom-field-receipt outputs/rom/rom_field_receipt.json`
  followed by
  `PYTHONPATH=src python scripts/compute_seam_costs.py --body-receipt <run-root>/body_carrier_receipt.json --rom-field-receipt outputs/rom/rom_field_receipt.json --rom-fields outputs/rom/rom_fields.npz --mesh <production mesh npz> --solve-domain A_v3240 --out-costs outputs/rom/seam_costs.npz --out-seam-cost-receipt outputs/rom/seam_cost_receipt.json`,
  followed by
  `PYTHONPATH=src python scripts/solve_seams.py --seam-cost-receipt outputs/rom/seam_cost_receipt.json --costs outputs/rom/seam_costs.npz --mesh <production mesh npz> --out-dir outputs/seams --out-solver-receipt outputs/seams/solver_promotion_receipt.json`,
  then pass the receipted `--seam-costs outputs/rom/seam_costs.npz` into
  `generate_undersuit` to annotate seams. When no real sampler is available,
  generate a plumbing-only one via `scripts/generate_synthetic_rom_sampler.py
  --body outputs/afflec_demo/afflec_body.npz --components <K> --samples 8 --out
  outputs/rom/afflec_sampler.json` (meta.synthetic=true) and swap in a real
  sampler at the same path when ready.
- Standardize per-body ROM dataset naming/versioning and add a caching policy/CLI so `sampler_real` outputs can be reused without reruns; document the refresh path alongside the cache location.
- Add a streaming MoCap ROM envelope pass (AMASS/contact/dance/clinical) that projects poses to ROM coefficients, emits per-dimension envelopes/density, tags rare/contact-only regimes, and optionally probes boundary poses with the FD kernel to flag mechanically hostile regions; keep outputs as JSON “ROM certificate” artifacts.
- Extend MDL with dynamic terms (velocity/acceleration/inertia, impact tolerance atlas, optional contact probability) and surface them in seam/fabric decisions without altering ROM kernels; document the injury/pain data sources used.
- Implement Sprint R spec (`docs/rom_levels_spec.md`): wire `data/rom/sweep_schedule.yaml` + `data/rom/task_profiles/*.yaml`, add `smii/rom/pose_schedule.py` + `smii/rom/completeness.py`, extend `sampler_real --schedule`, generate L0/L1/L2 sample artifacts, and emit `outputs/rom/rom_L3_certificate.json` with envelope deltas + rank correlations; seed acceptance tests for reproducibility and downstream stability checks.
- Execute Sprint ROM-L1 (`docs/sprint_rom_l1.md`): add legality/collision scoring, chain-aware sweep schedules, a chain-sensitive displacement proxy, and integrate these fields into seam cost diagnostics; publish L1 addendum in `docs/rom_levels_spec.md`.
- Sprint S1 (ROM-driven seam optimization):
  - Edge cost construction from vertex costs (mean/max/length-weighted integral) with unit tests — landed in `smii/seams/edge_costs.py`.
  - Deterministic seam solver (MST baseline) with `SeamSolution` API in `smii/seams/solver.py`; extend with shortest-path/min-cut variants as needed.
  - Constraint integration baseline (forbidden regions hard-fail, symmetry penalty, panel connectivity warnings) shipped; refine policies and error surfaces.
  - Diagnostics and explainability: overlays (PNG/SVG) of seams vs ROM heatmaps, JSON cost attribution (per-term and top-N avoided high-cost regions), example driver `examples/solve_seams_from_rom.py`.
  - PDA seam optimizer stack (kernels, MDL prior, moves, PDA controller) shipped in `smii/seams/{kernels,mdl,moves,pda}.py`; next: tune weights, add visual/debug outputs, and wire CLI/driver.
- Sprint S2 (production + diagnostics):
  - Add solver variants (`solve_seams_shortest_path`, `solve_seams_mincut`) reusing kernel+MDL objective and compare to MST baseline — implemented; extend benchmarks.
  - Add shortest-path mode controls for semantics work: `require_loop`, `symmetry_penalty_weight`, strict-local fallback, and reference-body diagnostics; validate on non-warped base mesh parity runs.
  - Ship diagnostics: ROM heatmap + seam overlays, avoided high-cost region highlights, per-seam kernel/MDL breakdown, stability/witness report; output PNG/SVG + JSON — scaffold added (`smii/seams/diagnostics.py` with threshold highlighting), still need richer visuals/witness.
  - Comparative evaluation script/notebook covering MST, PDA-MST, PDA-SP/mincut with metrics (total cost, seam length, panel count, max ROM cost intersected, perturbation stability) — initial script `examples/compare_seam_solvers.py` added; add notebook + perturbation metrics.
  - CLI/example driver `examples/solve_seams_from_rom.py` with config-driven weights/MDL (`configs/kernel_weights.yaml`, `configs/mdl_prior.yaml`) and solver selection emitting seams + diagnostics — defaults added; still need sample inputs/assets.
- Sprint S3 (fabric-aware, task-weighted):
  - Fabric kernels: incorporate stretch/shear mismatch and grain alignment into `EdgeKernel`; add fabric YAML loader (`configs/fabrics/*.yaml`) and penalties/tests.
  - Task profiles: load task mixtures (`configs/tasks/*.yaml`) and feed task-weighted ROM aggregation (`aggregate_rom(samples, task_profile)`).
  - Regime layer: extend PDA state with fabric assignments and grain rotations; add moves (`switch_fabric`, `rotate_grain`), manufacturability/MDL modifiers.
  - Diagnostics: per-panel rationale (fabric vs ROM), overlays showing fabric regimes + seams, stability under ROM and grain jitter; add task/fabric-aware example driver `examples/task_fabric_seam_demo.py`.
  - Add loop-mode panel controls (`min_panels`, `max_panels`) and chart-complexity regularization to prevent over-fragmented seam sets.
