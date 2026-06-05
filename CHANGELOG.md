## Unreleased

- Extended Gate 6c parent-surgery receipts with diagnostic Pareto/profile
  metadata:
  - added per-variant improvement/regression reporting for the hard metrics
    already used by the unwrap gate,
  - recorded Pareto-frontier membership for the materialized relief/drain
    candidates without changing the acceptance rule,
  - added profile-relative winner scoring so debug/manufacturing viewpoints can
    see a different preferred candidate even when the default scalar gate
    still rejects the variant,
  - added compact default-profile loss explanations for non-selected variants.
- Added the `failure_wedge_relief` parent-surgery family:
  - derives a two-leg wedge/lens candidate from serializer failure fields to
    panel or seam-boundary sinks,
  - materializes the wedge as a fabric-preserving chart split,
  - bounds wedge candidates by estimated chart count to avoid pathological
    parent-remainder fragmentation,
  - competes the new charts under the existing score/distortion/serializability
    gate and Pareto/profile receipt diagnostics.
- Added the `pareto_guided_dart_wedge` parent-surgery family:
  - derives a narrow two-leg dart/lens chart from a measured
    `failure_relief_path` candidate,
  - keeps the full failure component as guide evidence while materializing only
    the apex and local leg corridor,
  - bounds local exit-pair search and wedge size so large parent panels remain
    practical to compete,
  - records the candidate through the existing Pareto/profile diagnostics
    without changing scalar promotion.
- Added the `failure_lens_patch` parent-surgery target to the Gate 6c roadmap:
  - derives a bounded replacement patch support from serializer failure fields,
  - materializes the patch support as a separate face-backed chart and preserves
    the parent remainder,
  - keeps all promotion rules unchanged while testing whether patch-family
    geometry beats the exhausted path-family relief operators.
- Implemented and measured `failure_lens_patch` on the Afflec/P3 chain:
  - P0 gets a valid bounded lens patch that improves score and foldovers but
    regresses worst distortion to `16.176`, so it remains rejected,
  - P1 receives lens candidates but still promotes `cutout_r1`,
  - P2 receives no bounded lens patch candidate from the current support
    generator,
  - accepted parents remain P1 `cutout_r1` and P3 `relief_split`.
- Added `OperatorBasisSearchReceipt` diagnostics for materialized parent
  chart-domain decisions:
  - emits `smii.operator_basis_search.v1` per source parent panel,
  - composes depth-2, beam-8 operator trees from already measured
    single-operator metric deltas,
  - records profile winners, Pareto frontier membership, and exhausted-basis
    blockers without claiming true sequential tree rematerialization.
- Added the finished seam/body atlas receipt surface:
  - added `smii.seams.seam_derivation.FinishedSeamReceipt` and
    `derive_finished_seams` to compose promoted body, ROM, fabric, basis,
    seam-cost, solver, cut-topology, metric-correction, panel-unwrap, and
    manufacturing evidence into a final pattern atlas receipt,
  - wired `scripts/generate_manufacturing_artifacts.py` to emit
    `finished_seam_receipt.json` when supplied the upstream receipt hashes and
    paths,
  - wired `scripts/run_afflec_receipted_demo.py` through cut-topology and
    metric-correction gates and into the finished seam receipt emission path,
  - added `--images` support to the Afflec receipt runner, aligned its demo
    basis width with the bundled four-coefficient ROM sample payload, and
    inferred typed cut-topology operators from metric-panelization correction
    payloads,
  - added `schemas/finished_seam_receipt.schema.json` for the emitted receipt
    and claim-boundary flags,
  - added sibling Agda `DASHI.Interop.SeaMeInItROMSeamAtlas` to formalize the
    adaptive body atlas bridge over existing ROM kernel, unwrap competitor, and
    garment pattern receipts,
  - documented that exported patterns are serializations of the certified
    body/ROM/fabric seam atlas, not the geometry or a manufacturing authority
    claim by themselves.
- Added an external unwrap competitor harness:
  - added `smii.unwrap.external_competitors` with JSON-serializable run
    receipts, common benchmark metrics, declared-slice winner selection, and
    graceful unavailable receipts for optional tools,
  - measures dependency-light sphere candidates for BT369, cylindrical
    equal-area, equirectangular, cubed-sphere, octahedral, and HEALPix carriers
    when their dependencies are installed,
  - added an adversarial synthetic field suite with per-field winners and a
    winner histogram,
  - documents the competitor matrix and keeps the claim boundary at
    declared-benchmark winner, not global optimum or sphere-plane isometry.
- Added the BT369 sphere unwrap export surface:
  - added `smii.unwrap.sphere_bt369` with equal-area inverse pullback sampling,
    residual trits, 6-sector tangent orientation, ternary cell prefixes,
    seam/braid counts, MDL-bounded refinement depth, and JSON-serializable
    certificates,
  - recorded the theorem boundary in DASHI/Agda as a benchmark-gated
    approximation receipt, not a global isometry claim.
- Added a graph/ultrametric unwrap benchmark surface:
  - added `smii.seams.unwrap_benchmark` to generate a deterministic
    sphere-to-rectangle comparison mesh,
  - ranks graph/ultrametric rectangle unwrap, LSCM, bootstrap projection, and
    orthographic projection candidates across edge-length, area, angle,
    foldover, aggregate residual, and agreement-depth metrics,
  - added tests proving the graph/ultrametric rectangle strategy wins the
    declared benchmark without claiming a zero-distortion sphere flattening,
  - documented that the graph/ultrametric scoring layer is the formal gate above
    numerical UV backends.
- Implemented the P3 metric-correction and real Gate 6 unwrap lane:
  - added `scripts/emit_metric_correction_receipt.py` to emit
    `MetricCorrectionReceipt` from solver, cut-topology, seam-edge, and
    correction payload hashes,
  - wired `scripts/run_p3_afflec_transfer_chain.py` to stop at an explicit
    `metric_correction` stage when typed operators lack promoted correction
    evidence,
  - added a shared seam unwrap backend module and a real NumPy `lscm` path for
    `scripts/unwrap_panels.py`,
  - updated `PanelUnwrapReceipt` to accept `lscm` as a non-bootstrap backend,
  - documented cut-topology, metric-correction, and panel-unwrap receipt fields.
- Added visible progress and rough ETA reporting to
  `scripts/run_p3_afflec_transfer_chain.py` so long P3/Afflec validation runs
  show the active stage, command, elapsed time, estimated remaining time, and
  command-level finish/failure markers.
- Documented the dart / metric-correction formalism gap:
  - recorded the local archive thread IDs and formulas framing garment panels
    as `(u, Delta g)` and darts as discrete curvature operators,
  - cross-checked `../../dashi_agda` and noted that its current SeaMeInIt Agda
    surface formalizes seam graph/panelization receipts but not darts,
    `MetricCorrection`, or `Delta g`,
  - incorporated the reusable DASHI patterns for typed variational objects,
    projection-style result states, sidecar carriers, typed blockers, and
    required-vs-admissible correction gates,
  - added the wider reusable DASHI lemma/surface stack for quotient stability,
    observation transport, residual budgets, coordinate joins, authority
    boundaries, and side-information-backed non-promotion certificates,
  - recorded `../animalexic` as the runtime governance analogue for
    candidate-first reconstruction feeding SMII body carriers without bypassing
    SMII seam/dart/panel/manufacturing gates,
  - updated P3 docs/TODOs to require a receipt-level metric-correction contract
    before branchy/open topology can authorize panel unwrap or manufacturing.
- Added the Afflec receipted demo runner and production roadmap:
  - added `scripts/run_afflec_receipted_demo.py` to execute the native
    `A_v3240` Gate 0-7 chain as one command,
  - the runner now defaults Gate 0 to `--detector mediapipe`, supports
    `--detector bbox` as an explicit coarse diagnostic mode, and forwards
    `--require-high-trust-detector` to `afflec-demo`,
  - the runner stops at first non-promoted receipt, writes `run_manifest.json`,
    and uses exit codes `0` promoted, `1` expected blocker, `2` hard failure,
  - dry runs print the planned command chain without creating run artifacts,
  - added `docs/smii_production_roadmap.md` with the current demo flow,
    implementation upgrade steps, and future formal objects,
  - documented the native `A_v3240` Afflec demo runner behavior in the receipt
    orchestrator notes,
  - recorded the bounded MediaPipe Gate 0 diagnosis: direct execution emits
    artifacts, but the Afflec receipt remains diagnostic-only because
    measurement refinement inflates final betas and the skull residual stays
    marginally above threshold,
  - updated TODO/context to make Gate 6/7 production geometry and Gate 0
    refinement calibration the next upgrade lanes after the runner.
- Wired manufacturing receipt emission into the final fabrication gate:
  - added `ManufacturingReceipt` for hash-linked seam allowance and cutting
    layout artifacts,
  - added `scripts/generate_manufacturing_artifacts.py` to require promoted
    panel unwrap receipts, verify panel UV hashes, derive variable seam
    allowance from ROM pressure/shear gradients, and emit final manufacturing
    receipts,
  - flat seam allowance is now recorded as `allowance_varies=false` and blocks
    promotion instead of silently falling back to constant allowances,
  - the receipt DAG reader now loads `manufacturing_receipt.json` so Gate 7
    can promote from a real artifact.
- Wired panel unwrap receipt emission into the fabrication boundary:
  - added `PanelUnwrapReceipt` for hash-linked `panel_uvs.npz` artifacts,
  - added `scripts/unwrap_panels.py` to require promoted solver receipts,
    verify seam topology hashes, reject incomplete topology before flattening,
    and emit panel unwrap receipts,
  - panel unwrap promotion now records per-panel distortion, worst/mean
    distortion, subdivision usage, grain directions, UV hashes, and downstream
    manufacturing blocks,
  - the receipt DAG reader now loads `panel_unwrap_receipt.json` and exposes
    `can_manufacture()`.
- Wired solver promotion receipt emission into the topology lane:
  - added `SolverPromotionReceipt` for hash-linked `seam_edges.npz` artifacts,
  - added `scripts/solve_seams.py` to require promoted seam-cost receipts,
    verify cost hashes, select field-minima anchors, and emit solver receipts,
  - anchor fallback is now recorded through `connected_component_count` and
    `anchor_fallback_used` instead of remaining an invisible warning,
  - panel topology is now the Gate 6 boundary through `panels_are_disks`,
  - the receipt DAG reader now loads `solver_promotion_receipt.json` and exposes
    `can_unwrap_panels()`.
- Wired seam-cost receipt emission into the topology lane:
  - added `SeamCostReceipt` for hash-linked `seam_costs.npz` artifacts,
  - added `scripts/compute_seam_costs.py` to enforce promoted body and ROM-field
    receipts plus the native-or-promoted-correspondence solve-domain rule before
    cost promotion,
  - promoted seam costs now record finite coverage, cost uniformity, cost
    summary diagnostics, weight vectors, and downstream solver/panel blocks,
  - the receipt DAG reader now requires promoted seam costs for seam-solver
    eligibility.
- Wired ROM field receipt emission into sample aggregation:
  - `examples/rom_aggregate_from_samples.py` now accepts `--basis-receipt` and
    can emit `rom_fields.npz` plus `rom_field_receipt.json`,
  - aggregation hard-gates on promoted `BasisReceipt` before emitting Gate 3
    receipts,
  - emitted receipts bind basis/sample/summary/field hashes, sample counts,
    field names, pressure peak diagnostics, `field_uniformity`, and synthetic
    promotion boundaries.
- Wired basis receipt emission into canonical basis generation:
  - `scripts/generate_canonical_basis.py` now accepts `--body-receipt` and
    emits `basis_receipt.json` when the body carrier is promoted,
  - unpromoted body carriers hard-block basis construction before artifact
    emission,
  - emitted receipts hash both the carrier receipt file and generated basis
    artifact, record the bootstrap construction method, and gate promotion on
    relative reconstruction error against a static contact-pressure proxy.
- Wired correspondence receipt emission into seam reprojection:
  - `scripts/reproject_seam_report.py` can now write
    `correspondence_receipt.json` after quality metrics are computed,
  - emitted receipts bind source/target mesh hashes, mean/max distance,
    full-surface load ratio, seam-local transfer collapse, edge retention, and
    `A_T`,
  - collapsed nearest-neighbor transfers now become explicit `A_T=-1`
    diagnostic-only receipts with downstream solver/panel blocks.
- Wired Gate 0 into the Afflec export path:
  - `smii.app afflec-demo` now emits `afflec_body_raw_reprojection.npz`,
    `afflec_body_refined_pre_repair.npz`, and `body_carrier_receipt.json`,
  - the receipt binds source image hashes, checkpoint hashes, final mesh hash,
    mesh counts, measurement residuals, confidence, and a conservative
    skull/crown plausibility residual,
  - added `smii.orchestrator.read_receipt_dag` as a minimal run-directory
    promotion-state reader with first-blocker and seam-solver eligibility
    helpers.
- Added the receipt-orchestrator control surface:
  - new `docs/receipt_orchestrator.md` defines the carrier -> correspondence
    -> basis -> ROM field -> seam cost -> solver -> panel -> manufacturing
    receipt DAG,
  - README and TODO now foreground the receipted promotion order,
  - `BodyCarrierReceipt` now defaults non-promoted receipts to explicit
    downstream consumer blocks (`generate_undersuit`, `seam_cost_field`,
    `panel_unwrap`),
  - `generate_undersuit` now checks the concrete `generate_undersuit` consumer
    gate on body receipts.
- Added first-class correspondence and basis receipt primitives:
  - `CorrespondenceReceipt` / `TransformReceipt` records source/target mesh
    hashes, transfer residuals, load/collision and retention metrics,
    downstream blocks, and diagnostic NN-collapse gating,
  - `BasisReceipt` records `B_0` provenance over a carrier receipt with vertex
    count, basis dimension, construction method, reconstruction error,
    downstream blocks, and promotion state.

- Added a first-pass audit note for the two new blocking morphology issues:
  - new `docs/b_ogre_and_afflec_crown_audit_20260311.md`
  - pinned the historical `B_ogre` forward object to `outputs/suits/afflec_body/base_layer.npz` (`9438`) plus `outputs/rom/seam_costs_afflec_realshape_edges.npz`
  - recorded that the old `9438 -> 3240` control transfer is too lossy to count as an inverse/back-transfer
  - narrowed the current Afflec crown pathology away from parameter drift and toward late mesh generation / repair / export or a latent fitted-geometry issue revealed there

- Added a prioritized body-fit / inverse roadmap note:
  - new `docs/body_fit_and_inverse_roadmap_20260311.md`
  - priority order now starts with auditing the exact historical `B_ogre` object and diagnosing the current Afflec crown/head-shape failure
  - video-input fitting and SMPL-X feminine-body coverage are now explicitly queued after those blocking audits
- Added an explicit ROM sample morphology and transfer contract:
  - documented deterministic representative-sample anchors and fill policy,
  - documented the sample artifact contract as sampler-native posed meshes plus `rom_sample_manifest.json`,
  - recorded the current inverse audit result: no true geometry inverse exists today, only basis-space encoding and correspondence/reprojection transfer,
  - documented acceptance criteria for approximate transfer and requirements for any future true inverse.
- Implemented the baseline ROM sample morphology output path:
  - `smii.rom.sampler_real` now accepts `--out-rom-samples-dir` and `--rom-sample-count`,
  - the sampler can emit representative sampler-native posed sample meshes plus `rom_samples/rom_sample_manifest.json`,
  - coefficient exports now retain per-pose observation stats alongside the encoded field samples.
- Integrated representative sample morphology artifacts into the viewing/reporting surfaces:
  - `scripts/render_run_reference.py` classifies exported sample meshes as `rom_sample_pose` in the morphology audit,
  - `scripts/render_rom_operator_report.py` accepts `--sample-manifest` and renders a dedicated representative-sample section,
  - `scripts/protocol_strategy2_bundle.py` can pass a sample manifest through to the operator report.
- Clarified the historical inverse-ROM/internalization intent and current limitation:
  - documented that an earlier project goal was to solve seams on an internalized/ogre-like morphology and then invert back to the fitted SMPL-X body,
  - recorded that the current repo does not provide a proven inverse transform for that step; today it provides correspondence/reprojection plus transfer diagnostics,
  - aligned project docs with the broader milestone framing: sewable bodysuit first, then thermal routing/cooling, then comfortable systems packaging, then later harder "iron man" modules.
- Refined the morphology roadmap into smaller executable planning steps:
  - broke `M2` into sample selection, artifact contract, minimal emission, and report integration,
  - broke `M2b` into code-path audit, current-reality documentation, approximation acceptance, and future inverse requirements,
  - updated planning state so the next bounded step is `M2.1 + M2b.1`.
- Added morphology-debugging planning state under `.planning/`:
  - new `.planning/spec.md`, `.planning/architecture.md`, `.planning/plan.md`, `.planning/status.json`, and `.planning/devlog.md`,
  - current prioritized milestone order is:
    1. backfill morphology observations on reference runs,
    2. emit explicit ROM sample morphology artifacts,
    3. compare candidate ROM fields on one topology,
    4. revisit seam-solver sensitivity after morphology attribution is clearer.
- Backfilled morphology observations on the current reference runs:
  - added `morphology_observations.json` to the main comparison, same-topology, kernel-diagnostic, and verified bundle runs,
  - run pages can now distinguish normal-human body/render outputs from operator-only field artifacts and from target-geometry transfer artifacts,
  - override files now also create manual audit rows for artifacts that are not auto-detected by the run-page audit (for example kernel-diagnostic field images).
- Added a stage-by-stage morphology audit to run reference pages:
  - `scripts/render_run_reference.py` now emits a `morphology_audit` ledger in `run_report_manifest.json`,
  - the run page shows where geometry actually changes versus where only fields, seams, or reprojection/rendering change,
  - run roots can provide `morphology_observations.json` or `morphology_audit_overrides.json` to record observed categories such as `normal_human`, `ogre_like`, or `flailing` for specific artifacts.
- Changed ROM report visualization/output policy:
  - `scripts/render_rom_operator_report.py` now renders analytic coefficient visuals directly into `index.html` as DOM/SVG charts instead of emitting report-generated PNG files,
  - the report accepts repeatable `--media-path` inputs and embeds existing topology-level media artifacts such as overlays, heatmaps, GIFs, and WebMs directly in the page,
  - `scripts/protocol_strategy2_bundle.py` now renders the operator report after bundle media exists and passes render/map/seam directories into the report so the page acts as the primary viewing surface.
- Added run-level reference and index pages:
  - new `scripts/render_run_reference.py` emits a canonical single-run HTML page that groups body/ROM/seam/media/manifests for one run root,
  - new `scripts/render_run_index.py` catalogs runs across one or more roots, adds inferred timestamps plus run-type labels, and links to each run reference page,
  - Strategy 2 bundles now emit `run_reference/index.html`, refresh an `outputs/assets_bundles/index.html` catalog, and also refresh a unified `outputs/index.html` catalog across bundle/comparison/seam run roots when present.
- Cleaned up seam-orbit temporary artifacts:
  - `scripts/render_seam_orbit.py` now removes its temporary frame PNG directory after GIF/WebM encoding,
  - deliberate still outputs remain on disk and are surfaced by run reference pages.
- Added a real image-space SMPL-X fitting path:
  - `smii.pipelines.fit_from_images.regress_smplx_from_images` now supports `fit_mode=auto|heuristic|reprojection`,
  - reprojection mode builds per-image 2D observation artifacts, optimizes shared betas plus per-image pose/camera parameters against joint reprojection loss, and records optimization metrics,
  - `smii.app afflec-demo` and `fit-from-images` now emit `*_observations.json` sidecars and default to `fit_mode=auto`,
  - when reprojection cannot run in `auto` mode, the pipeline falls back to the heuristic path and records the fallback in diagnostics.
- Recalibrated the Afflec image-fit pipeline and made it auditable:
  - `smii.pipelines.fit_from_images` now uses raw photo-derived measurements directly for refinement instead of re-measuring an intermediate mesh, which removes the earlier unit/sign pathology in the `bbox` path,
  - raw regression payloads now include `images_used`, `detector`, `fit_mode`, `measurement_source`, trust status, confidence summaries, and consistency flags,
  - new diagnostics artifact `afflec_fit_diagnostics.json` / `<subject>_fit_diagnostics.json` reports raw vs refined stages, beta magnitudes, and fit warnings,
  - `afflec_measurement_fit.json` and `afflec_smplx_params.json` now persist explicit image-fit provenance and consistency metadata,
  - `smii.app afflec-demo` and `fit-from-images` gained stricter control flags for refinement/trust enforcement (`--skip-measurement-refinement`, `--require-high-trust-detector`, `--fail-on-consistency-errors`).
- Added operator-level ROM inspection artifacts and reporting:
  - `smii.rom.basis.KernelProjector` now supports field->coefficient encoding (`encode`, `encode_batch`) for orthonormal bases,
  - `smii.rom.sampler_real` accepts `--basis` + `--out-coeff-samples` and exports per-pose `seam_sensitivity` coefficient samples alongside existing seam-cost outputs,
  - new `scripts/render_rom_operator_report.py` renders a static `index.html` plus JSON summaries from basis/meta/coeff/certificate artifacts,
  - the ROM operator report now explicitly flags basis/body/cost/meta topology mismatches via `consistency_status` and `consistency_flags`, and labels topology-level artifacts using their own vertex counts rather than inheriting the basis topology.
- Extended Strategy 2 bundle manifests to classify artifact semantics:
  - `scripts/protocol_strategy2_bundle.py` accepts optional ROM operator inputs (`--rom-basis`, `--rom-meta`, `--rom-envelope`, `--rom-certificate`, `--rom-coeff-samples`),
  - bundle manifests now include per-artifact `artifact_level`, `role`, `topology`, and `domain`,
  - when basis + meta are provided, the bundle renders an operator report under `rom_operator/`.
- Clarified ROM/operator vs topology/domain semantics in docs and planning:
  - `docs/mesh_provenance_afflec.md` now states explicitly that `ogre` is a working topology/provenance label for the `v9438` branch, not "the ROM invariant",
  - `docs/seam_pipeline_intended_vs_observed.md` now separates operator-level ROM inspection from mesh-orbit artifacts,
  - `COMPACTIFIED_CONTEXT.md` records archived thread metadata from the canonical ROM/planning chats,
  - `TODO.md` now includes explicit follow-through for operator-level ROM inspection artifacts and bundle labeling.
- Formalized body/mesh provenance protocol and reduced ambiguity from misleading filenames:
  - added `docs/body_lineage_protocol.md` and updated runner/provenance docs to prefer timestamped output roots,
  - documented that fixed paths under `outputs/*` are not stable provenance and must be backed by hashes/counts.
- Stabilized orbit rendering morphology:
  - `scripts/render_variant_orbits.py` now defaults to an explicit up-axis (`--axis-up`, default `y`) instead of inferring up-axis from max span (which can rotate T-poses and create “ogre-like” silhouettes),
  - render manifests now record axis conventions and canonicalization settings.
- Added correspondence visualization tooling:
  - new `scripts/render_vertex_map_orbits.py` renders mesh-only (and optional correspondence-line) orbit artifacts for a vertex-map NPZ, with a timestamped `map_manifest.json`.
- Made Strategy 2 bundles unambiguous by encoding morphology roles in filenames:
  - `scripts/protocol_strategy2_bundle.py` now requires explicit `--base-role` / `--rom-role` provenance labels (no vertex-count inference),
  - all Strategy 2 renders/maps/seam reprojections now use stems containing `<role>_v<vertex_count>` and the direction (`native`, `reprojected_seams_from_*`).
- Enforced a shared orientation heuristic by default:
  - orbit renderers now default `--axis-up auto` and use PCA + robust tail statistics to infer a stable width/depth/up frame from geometry,
  - `scripts/render_seam_orbit.py` supports aligning a mesh render into a reference mesh canonical frame (`--align-to-mesh`) for bundle comparability,
  - `scripts/render_vertex_map_orbits.py` aligns source canonical into target canonical in `--axis-up auto` mode (avoids correspondence “pincushion” caused by axis drift),
  - Strategy 2 default `--axis-width` changed to `none` (do not override auto width unless explicitly requested).
- Made Strategy A vs B a controlled experiment:
  - new `scripts/seam_compare_metrics.py` computes seam graph stats, mesh-edge validity, seam lengths, and reprojection quality from a Strategy 2 bundle,
  - `scripts/protocol_strategy2_bundle.py` now writes `manifests/seam_compare_metrics.json` on every run.
- Added mesh registry tooling:
  - new `scripts/mesh_registry.py` emits timestamped `mesh_registry.json` with hashes/topology/bbox + a units guess,
  - optional `--alias-dir` writes descriptive symlink aliases instead of renaming meshes.
- Made seam edge validity explicit in outputs:
  - `examples/solve_seams_from_rom.py` now records mesh-edge conformance metrics in `seam_report.json` (`mesh_edge_valid_ratio`, counts),
  - `scripts/reproject_seam_report.py` now records the same metrics for transferred seam reports when target faces are available.
- Hardened local test execution after adding submodules:
  - configured pytest in `pyproject.toml` to collect only `tests/` (avoids submodule test collection),
  - added `tests/__init__.py` so `tests.helpers.*` imports resolve consistently.
- Made SMPL-X provider import-safe when `smplx` is not installed:
  - `avatar_model.providers.smplx` now defers importing `smplx` until after asset checks, so tests and tooling can run without the runtime dependency unless SMPL-X is actually instantiated.
- Fixed pattern export regressions uncovered by full-suite runs:
  - `_cleanup_panel_outline` no longer shrinks low-vertex outlines via Laplacian smoothing (restores expected PDF tiling behavior),
  - seam allowance offset failures now emit structured codes (`SEAM_ALLOWANCE_OFFSET_FAILED`) and matching issue entries.
- Traced the ogre-generation correspondence stage in `smii.rom.sampler_real` and added native map export:
  - new CLI flag `--out-correspondence <path.npz>` writes bidirectional source/target vertex maps with distances and collision metrics at sampler runtime,
  - remap stage now reuses the computed target->source map (no duplicate NN pass) and records correspondence artifact metadata in `out-meta`.
- Improved orbit artifact naming in `scripts/render_variant_orbits.py`:
  - preserves canonical `overlay_*` outputs for compatibility,
  - adds descriptive aliases containing run/body/cost/render settings/timestamp,
  - summary CSV/JSON now include `*_descriptive` artifact paths.
- Added per-run render sidecar manifests in `scripts/render_variant_orbits.py`:
  - writes `render_input_manifest.json` plus timestamped copies,
  - captures render body/cost hashes + stats, seam report provenance, and render params.
- Upgraded shortest-path loop mode:
  - new solver options `strict_loop` and `loop_count`,
  - strict loop mode drops non-simple loop fallbacks instead of accepting open/non-simple seams,
  - loop mode now supports selecting multiple disjoint simple loops when available.
- Refined loop diagnostics:
  - panel warnings now avoid propagating rejected candidate-loop warnings when a strict-valid loop set is selected,
  - reduces false-alarm `non-simple` warnings on visually/graph-valid final loop outputs.
- Extended seam CLI loop controls:
  - `--sp-loop-count`,
  - `--sp-loop-strict/--no-sp-loop-strict`.
- Added shortest-path loop regression coverage:
  - strict mode dropping invalid loops,
  - multi-loop selection on disjoint cycle graphs.
- Verified environment behavior for ROM sampler:
  - local project venv (`/home/c/Documents/code/ITIR-suite/.venv`) has `torch`/`yaml` but no `smplx`,
  - sampler now runs via `/opt/conda/bin/python` after making `sampler_real` less dependency-coupled.
- Reduced sampler coupling to unrelated pipeline dependencies:
  - `smii.rom.sampler_real` now decodes SMPL payloads locally (no import of `fit_from_measurements`/`jsonschema` chain),
  - `pose_schedule` import is now lazy (only in `--schedule` mode), so JSON `--poses` mode no longer requires `pyyaml`.
- Executed A-vs-B protocol run `outputs/seams_run/domain_ab_20260213_101158` with fresh sampler outputs:
  - regenerated sampler artifacts at `outputs/rom/domain_ab_20260213_101158/`,
  - established that sampler-native map is `10475 -> 3240` and is incompatible with `B_ogre` seam source `9438`,
  - control `9438 -> 3240` reprojection still fails quality (`retention=0.0526`, `collision=0.85`, mesh-edge-valid-ratio `0.0`).
- Executed loop-mode shortest-path probe `outputs/seams_run/looping_probe_20260213_103313`:
  - `--sp-require-loop` runs on both `v3240` and `v9438` still emitted non-simple/non-closable loop warnings,
  - control transfer from `v9438 -> v3240` remained collapsed (`edge_retention=0.0476`, collision `0.85`).
- Executed base-layer-focused strict multi-loop run `outputs/seams_run/base_layer_multiloop_20260213_111639`:
  - solve target: `outputs/suits/afflec_body/base_layer.npz` (`9438`),
  - config: `--sp-require-loop --sp-loop-strict --sp-loop-count 2 --sp-loop-waypoints 2`,
  - result remains sparse in upper panels (several panels empty) with anchor-disconnect and loop-closure warnings.
- Documented user visual cross-check for `domain_ab_20260213_101158`:
  - transferred control output remains non-conforming,
  - native solves remain mesh-edge valid,
  - morphology appeared inverted relative to `A_base`/`B_ogre` naming; docs now treat those names as provisional labels.
- Executed A-vs-B protocol run `outputs/seams_run/domain_ab_20260213_095932`:
  - A and B native solves remain mesh-edge valid (`mesh_edge_valid_ratio=1.0`),
  - B->base reprojection still fails strict quality (`reproject_exit_code=2`, retention `0.0526`, collision `0.85`, `mesh_edge_valid_ratio=0.0`),
  - lineage + decision metrics written under the run root.
- Executed the A-vs-B protocol run `outputs/seams_run/domain_ab_20260213_051532` and recorded outcomes in `docs/seam_pipeline_intended_vs_observed.md`:
  - Strategy A native checks pass,
  - Strategy B reprojection fails strict quality gates (`mean/max distance`, `edge retention`, `collision ratio`),
  - reverse-direction NN transfer (`A_base -> ogre`) also fails (edge collapse to zero),
  - provisional freeze on Strategy A, Strategy B remains diagnostic.
- Added persistent full-mesh correspondence tooling:
  - new `scripts/build_mesh_vertex_map.py` writes source<->target vertex maps with distance/collision metadata,
  - `scripts/reproject_seam_report.py` now supports `--vertex-map-file` and bidirectional map reuse (`source_to_target` or `target_to_source` arrays),
  - reprojection metadata now records mapping mode and map artifact path.
- Added a runnable A-vs-B seam-domain checkpoint protocol to `docs/seam_pipeline_intended_vs_observed.md`:
  - timestamped command sequence for Strategy A (base-native) and Strategy B (ROM-native + reprojection),
  - strict quality gates and freeze rule,
  - decision-record section for dated policy selection.
- Added `docs/seam_pipeline_intended_vs_observed.md` to explicitly capture:
  - intended pipeline stage purposes,
  - user-observed vs agent-observed behavior for `variant_matrix_20260213_035227`,
  - unresolved solve-domain decision (`solve-on-base` vs `solve-on-ROM`).
- Updated provenance/diagnostics documentation for alignment:
  - `docs/mesh_provenance_afflec.md` now links to the intended-vs-observed seam note,
  - `docs/ogre_artifact_diagnostics.md` now reflects mixed ROM+render contributors and the current uncertainty position.
- Updated planning docs:
  - `ROADMAP.md` now includes a seam-domain/topology-lineage checkpoint and acceptance gate,
  - `TODO.md` now tracks canonical solve-domain decision, lineage manifest requirement, and A-vs-B protocol.
- Added README link to the new seam pipeline status document.
- Documented run-specific mismatch case for `variant_matrix_20260213_035227` in `docs/ogre_artifact_diagnostics.md` (mouth/groin seam placement on source topology, scapula-only line after failed reprojection).
- Added reprojection quality gating in `scripts/reproject_seam_report.py`:
  - emits `edge_retention_ratio`, `unique_target_vertices`, `target_vertex_collision_ratio`, `quality_ok`, and `quality_violations`,
  - supports thresholds (`--max-mean-distance`, `--max-distance`, `--min-edge-retention`),
  - supports threshold `--max-target-collision-ratio` for many-to-one mapping collapse,
  - supports `--strict-quality` to fail on poor topology transfer.
- Added explicit Afflec mesh-lineage documentation in `docs/mesh_provenance_afflec.md` and linked it from `README.md`.
- Clarified topology expectations in provenance-related docs:
  - `docs/rom_real_sampler.md` now treats body vertex count as run-specific (not fixed to 9438),
  - `docs/pipeline_runner.md` now references lineage docs and avoids stale hardcoded hash assumptions,
  - `docs/ogre_artifact_diagnostics.md` now calls out expected 9438->3240 branch mismatch behavior.
- Added `scripts/audit_mesh_lineage.py` to emit timestamped JSON/CSV audits for body/ROM/seam/reprojection compatibility checks with hash and index diagnostics.
- Documented ogre/pathology seam artifact signatures and known failure modes in `docs/ogre_artifact_diagnostics.md`, including baseline solver behavior and control toggles.
- Restored starburst-control compatibility knobs across seam solvers:
  - `solve_seams(..., max_branch_degree, branch_penalty_weight)` now enforces optional branch-degree-aware spanning selection,
  - `solve_seams_pda(..., max_branch_degree, branch_penalty_weight)` now threads branch controls into initial solution, witness checks, and candidate scoring,
  - `solve_seams_mincut(...)` accepts branch-control kwargs for API compatibility and records them in metadata.
- Revalidated seam starburst regression suite with branch controls enabled (`tests/seams/test_starburst_regression.py`).
- Added vertex-topology mismatch guards:
  - `examples/solve_seams_from_rom.py` now fails early on body vs ROM cost length mismatch,
  - `scripts/render_variant_orbits.py` now fails when seam edge indices are out of range for the selected render body.
- Added explicit seam-topology transfer utility `scripts/reproject_seam_report.py` for controlled source->target seam index reprojection with distance diagnostics.
- Added seam-report provenance metadata (`body_path`, vertex counts, ROM cost path) in `examples/solve_seams_from_rom.py` outputs.
- Expanded basis/sampler provenance fields:
  - `scripts/generate_canonical_basis.py` now stores `source_path` and `vertex_count` in basis metadata,
  - `scripts/generate_synthetic_rom_sampler.py` now stores `body_path` in sampler metadata.
- Added shortest-path solver controls for seam semantics debugging:
  - `require_loop` loop-attempt mode with explicit closure warnings,
  - `symmetry_penalty_weight` mirrored-edge penalty term,
  - `allow_unfiltered_fallback` strict locality control,
  - optional `reference_vertices` metrics to compare seam lengths on an origin mesh.
- Extended `examples/solve_seams_from_rom.py` shortest-path CLI with loop/symmetry/fallback/reference-body arguments for repeatable debug runs.
- Added shortest-path regression tests covering loop mode behavior and symmetry-penalty accounting.
- Updated ROM formalisation docs (Sprint R levels/schedule/completeness), sprint status, and compact context snapshot; aligned TODOs to new deliverables and operational checks.
- Added R6-lite spline fitting with curvature-bound fallback and split gating in boundary regularization.
- Added opt-in auto-split (`--auto-split`) and split helpers, plus structured issue metadata in patterns output.
- Expanded SVG annotations with severity styling, legend, and annotation levels.
- Added stress and spline tests for boundary regularization; fixed seam loop indexing for canine axis tests.
- Made undersuit pipeline test mesh watertight to satisfy generator requirements.
- Fixed suit_hard attachment validation import for test collection.
- Added seam length reconciliation with `SEAM_MISMATCH` issues based on seam partner metadata.
- Propagated seam-aware split metadata (`seam_avoid_ranges`, `seam_midpoint_index`) into exporter panels.
- Captured the missing afflec regression warning explicitly in tests.
- Documented TODO hygiene guidance link from README to `CONTEXT.md`.
- Added seam partner metadata normalization to derive seam-aware split ranges and documented the schema.
- Added multi-page PDF tiling with selectable page sizes for pattern exports.
- Added panel validation gate aggregation to surface ok/warning/error status in pattern metadata.
- Added seam cost NaN-safe aggregation (zero-fill with warnings) and vertex mapping policy flags (`--vertex-map`, `--max-map-distance`) to the ROM sampler provenance path.
- Added ROM-driven seam optimization scaffolding: edge cost derivation modes, deterministic MST seam solver with constraint enforcement, PDA kernel/MDL stack, and regression tests.
- Added solver variants (shortest-path, min-cut) sharing kernel+MDL objective, diagnostics/report scaffold, and example CLI `examples/solve_seams_from_rom.py` for end-to-end runs.
- Added default kernel/MDL config YAMLs, richer diagnostics overlay highlighting high-cost seams, and comparison script `examples/compare_seam_solvers.py` for benchmarking MST/PDA/SP/mincut.
