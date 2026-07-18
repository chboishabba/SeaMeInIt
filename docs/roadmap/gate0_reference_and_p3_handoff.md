# Gate 0 Reference Policy And P3 Handoff

Status date: 2026-06-04

This note records the current P1/P3 handoff after checking the available
formalism and multiview reference material.

## Reference Sources

- DASHI/Agda: the live checkout is `../../dashi_agda` from this repo. It is the
  formal reference source; the bundled snapshots in this repo are only
  secondary context.
- Multiview: `animalexic` was found at `/home/c/Documents/code/animalexic`. It
  is a stereo/multiview reconstruction repo, not a human body-shape repo, but
  its calibration and promotion rules transfer well to Gate 0.
- Cutting/manufacturing: local constraints come from
  `docs/pattern_flattening.md`, `docs/undersuit_generation.md`,
  `docs/receipt_orchestrator.md`, and `docs/smii_production_roadmap.md`.

## Gate 0 Reference-Set Policy

The current Afflec body fit should be treated as underconstrained when the
reference set is sparse, mostly same-perspective, or likely long-lens. This is
consistent with the observed crown/head issue: perspective cues are weak, so
the optimizer can satisfy visible features while stretching hidden or weakly
observed skull shape.

Promotion policy:

- do not reject same-perspective or long-lens references outright,
- mark them with `WARN:low_view_diversity` and/or
  `WARN:long_lens_flattening_risk`,
- preserve those warnings as monotone evidence, but distinguish diagnostic
  severity from refinement and raw-body authorization; a warning may cause
  refinement to abstain without making every body decision impossible,
- require materially different yaw/pitch coverage before treating the fit as a
  calibrated anthropometric baseline,
- keep `bbox` as diagnostic-only and prefer MediaPipe for production-style
  Gate 0 runs,
- keep every subject in a separate reference set with explicit provenance.

Reference-set manifests should record:

- `subject_id`
- source references or regeneration/download commands
- image hashes when local files exist
- detector and fit mode
- acquisition notes, including likely lens/perspective bias
- inclusion/exclusion reasons
- identity confidence and any uncertainty

Do not commit binary reference images. If additional Afflec or comparison-subject
references are needed, record commands/manifests in TODO or ignored local run
roots.

## Parallel Subject Lane

A Brad Pitt or similar parallel subject lane is useful as a non-promotional
control only. It should test whether Gate 0 detects celebrity-reference bias and
view-diversity weakness. It must not be used to tune Afflec-specific thresholds
or commit binary photos.

Recommended control command shape:

```bash
../.venv/bin/python -m smii.pipelines.fit_from_images \
  --images <ignored-reference-root> \
  --output <ignored-output-root> \
  --fit-mode auto \
  --detector mediapipe \
  --require-high-trust-detector
```

## Formal Promotion Constraints

SeaMeInIt receipts remain the operational formalism for this repository. DASHI
contributes the formal control vocabulary that should shape those receipts:

- `../../dashi_agda/UFTC_Lattice.agda` models severity as a join semilattice:
  max severity wins and severity propagation is monotone. A downstream stage
  must not mask an upstream warning or blocker.
- `../../dashi_agda/DASHI/Execution/Contract.agda` separates per-step state,
  source projection, projected deltas, MDL descent, basin preservation, and
  eigen/overlap obligations. SeaMeInIt should likewise keep body trust,
  transfer quality, ROM/basis quality, solver quality, and manufacturing
  quality as separate receipt fields rather than one blended score.
- `../../dashi_agda/DASHI/Algebra/ProjectionVsInvertible.agda` proves that a
  kernel that is both projection-like and invertible collapses to identity.
  For SeaMeInIt, this is the formal reason not to describe nearest-neighbor
  correspondence or lossy reprojection as an inverse.
- `../../dashi_agda/DASHI/Core/AuthorityBoundary.agda` distinguishes citation
  authority from artifact authority. For SeaMeInIt, a documented body/image
  source or external citation does not authorize a machine-readable body,
  transfer, or manufacturing artifact unless the artifact receipt exists.
- `../../dashi_agda/fracdash-impl/README.md` treats executable status as a
  surrogate unless it mirrors the upstream admissibility witness. For
  SeaMeInIt, diagnostic CLIs and generated reports are surrogate evidence until
  they produce the required promoted receipt.

Required invariants:

- production artifacts promote only from promoted upstream receipts,
- non-promotion must be explicit and must block downstream consumers,
- warning/blocker severity must be monotone: downstream receipts can add
  evidence, but they cannot erase upstream severity without a named clearing
  witness,
- `A_body=+1`, `A_field=+1`, `A_seam_cost=+1`, and either `A_T=+1` or native
  `solve_domain=A_v3240` are required before solver promotion,
- correspondence/reprojection must never be called an inverse without a defined
  forward transform, inverse or pseudo-inverse, round-trip geometry error,
  seam-structure preservation evidence, and thresholds.

## Gate 0 Refinement Authority

The present reference warnings and export checkpoints are implemented. The
remaining defect is that measurement refinement is candidate generation without
a separate authority boundary. The next implementation must normalize and hash
the effective numerical policy actually consumed by the solver, including the
measurement model, scale rule, beta bounds, prior/anchor weights, and solver
settings. The candidate must be anchored to image-derived betas, constrained
during the solve, and fully re-evaluated after solving.

The initial persisted objects are a refinement policy, one refinement receipt,
and `BodyCarrierReceipt` v2. The receipt contains candidate evidence, a
`promote`/`abstain`/`reject` decision, blockers/warnings, and policy/input/
candidate/output hashes. A body receipt identifies the canonical source, the
selected pre-repair checkpoint/hash, and the final repaired/exported hash. A
refinement abstention never silently selects the refined candidate; the raw
image fit proceeds only through its own body guard. The final export, not only
the pre-repair checkpoint, must satisfy the downstream body trust checks.

## P3 Back-Transfer Requirement

P3 should define the return path against the exact forward object, not generic
nearest-neighbor reprojection.

The back-transfer spec must include:

- source and target topology tags and hashes,
- the forward object identity,
- transfer mode: true inverse, pseudo-inverse, or approximate correspondence,
- round-trip checks when possible,
- seam retention and collapse limits,
- collision/load metrics,
- morphology preservation expectations,
- explicit labels for transferred vs native artifacts.

Until those exist, transfer-backed results are diagnostic-only.

## Cutting And Manufacturing Preferences

The manufacturing target is a sewable, maker-readable pattern package, not just
a mathematically flattened mesh.

Hard constraints:

- minimize seam count while keeping panels sewable,
- avoid starburst, spiky, or jagged outlines,
- enforce sewability and distortion budgets before flattening,
- split panels when budgets are exceeded,
- align grain/stretch direction with body movement and material anisotropy,
- separate stitch lines and cut lines,
- emit grainlines, notches, folds, labels, and seam partner metadata,
- preserve scale in SVG/PDF/DXF outputs,
- block manufacturing promotion on flat allowances, missing labels/notches,
  invalid methods, hash mismatches, or failed panel unwrap.

Panel unwrap promotion must prove promoted solver receipt hash, panel count,
disk-like panels, bounded per-panel/worst distortion, grain directions, UV hash,
and seam topology hash. Manufacturing promotion must hash the panel unwrap
receipt, seam allowance artifact, cutting layout, panel UV hashes, grain
directions, notches, labels, method, and accessibility decision.

For P3 debugging, the flat pattern view is itself a key metric. Every blocked
panel unwrap that emits UVs should also emit diagnostic cut-sheet artifacts that
show what a maker would actually be asked to cut:

- a 3D mesh seam overlay bound to the promoted body mesh and solver seam edges,
- a UV diagnostic view for the raw flattening,
- a face-backed flat cut sheet showing UV triangle mesh, true patch boundary
  edges, solver seam edge segments, grainline, panel labels, and distortion
  values,
- an explicit diagnostic summary recording receipt hashes, mesh/seam hashes,
  panel face counts, boundary edge counts, seam graph summary, seam segment
  counts, distortion margin, blockers, and `manufacturing_authorized=false`.

The cut-sheet diagnostic is not a manufacturing receipt and must not clear the
panel unwrap blocker. It exists to make the failure legible: if the sheet still
looks like a single distorted body-shaped patch, a starburst, or an uncut mesh,
the next task is seam topology, darts, and relief cuts, not threshold
relaxation.

P3 now has an explicit cut-topology gate between solver and unwrap:

- `solver/cut_topology_receipt.json` records mesh/seam hashes, seam graph
  endpoints/branches/components, panel face counts, boundary edge counts,
  disk checks, typed dart/gusset counts, and cut-topology blockers.
- `solver/dart_relief_candidates.json` is diagnostic-only guidance for the next
  solver/cutting pass. It uses an angle-deficit developability proxy to propose
  typed dart or relief-cut candidates, but it does not authorize panel unwrap.
- `solver/metric_correction_receipt.json` records the promoted or blocked
  correction bridge when typed operators are present.
- `panel_unwrap_receipt.json` may consume a promoted cut-topology receipt via
  `cut_topology_receipt_hash` and a promoted metric-correction receipt via
  `metric_correction_receipt_hash` when typed operators are present; without
  those, new P3 chains should stop at the cut-topology or metric-correction
  blocker.

Context check: local archive DB lookup on 2026-06-04 resolved
`UV unwrapping explanation` (`canonical_thread_id`
`3562461ee45a9f6eb3b24f0cbd4a233161a7b60e`) and `Repo planning blockers`
(`canonical_thread_id` `9382ee2cba0c06880b8d351e2055acb49e97a12d`) from
`~/chat_archive.sqlite`, with no web fetch. The archived guidance matches this
policy: projection shadows are not real patterns; high-curvature regions need
seams or darts; per-panel LSCM/ABF-style unwraps need real triangle patches and
maker-readable annotations.

Follow-up context check on 2026-06-04 used `robust-context-fetch` against the
same local DB and resolved:

- `Repo planning blockers`
  - online UUID: `6a03e573-caa4-83ec-83ed-af05b723ed4c`
  - canonical thread id: `9382ee2cba0c06880b8d351e2055acb49e97a12d`
  - relevant range: stitched lines `16670-16830`
- `Seam Walker Troubleshooting`
  - online UUID: `698d5e21-6d54-839a-a127-088c1dc21227`
  - canonical thread id: `0eff7f41332ca191629d9246ad3677518461fa55`
- `UV unwrapping explanation`
  - online UUID: `6916c180-1080-8320-ae2d-acc2e3ac3c23`
  - canonical thread id: `3562461ee45a9f6eb3b24f0cbd4a233161a7b60e`

The sharper archived formalism says darts, gathers, easing, panel shaping,
stretch zoning, variable knit, pleats, gussets, and bias orientation are typed
implementations of controlled metric mismatch injection. The useful garment
object is therefore `(u, Delta g)`, not just a UV projection `u`, where
`Delta g` is the allowed metric modification field. A dart is a discrete
curvature operator: wedge removal in the flat domain that intentionally injects
local curvature when reassembled.

P3 implication: do not fix `open_or_branched_seam_graph` by blindly pruning all
junctions until the current simple-graph validator passes. The next topology
repair should classify each junction as ordinary panel boundary, typed
dart/relief/gusset/easing operator, or invalid fragmentation. Only ordinary
cut boundaries need to satisfy the simple panel-boundary graph rules; typed
metric-correction operators need explicit receipt fields before they can
authorize unwrap or manufacturing.

DASHI/Agda cross-check on 2026-06-04:

- `../../dashi_agda/Docs/SeaMeInItROMKernelFormalism.md` and
  `../../dashi_agda/DASHI/Interop/SeaMeInItROMKernelFormalism.agda` currently
  model the SeaMeInIt lane as
  `BodyCarrier -> KernelBasis -> ROMOperator -> ProjectedField -> SeamGraph ->
  SeamCutPanelization -> ManufacturingReceipt`.
- The formal seam definition is graph/panelization-only: `G = (V, E)`,
  `S subset E`, and `panels = connected components of G \\ S`.
- The Agda records include `SeamGraph`, `SeamCutPanelization`,
  `SeaMeInItROMKernelSurface`, `topologyGate`, and `panelizationGate`; they do
  not yet include a `Dart`, `MetricCorrection`, `Delta g`, wedge-removal, or
  curvature-insertion type.
- `../../dashi_agda/Docs/MeasurementSurfaceProjectionContract.md` and
  `../../dashi_agda/scripts/hepdata_projection_contract.py` are the nearest
  existing pattern for the missing contract: Delta-bearing artifacts must state
  semantics, metric propagation, diagnostics, and rejected/degraded outcomes
  before downstream theorem-adjacent gates consume them.
- `../../dashi_agda/DASHI/Physics/Closure/CrossDomainVariationalSpine.agda`
  supplies the closest conceptual pattern: typed variational objects carry
  delta, projection, defect, gate, observation quotient, and claim boundary.
  Metric corrections should mirror that shape as SeaMeInIt-local receipts.

P3 now treats `dart_relief_candidates.json` as advisory until the emitted
`MetricCorrectionReceipt` promotes. Branchy/open structures become admissible
only when `CutTopologyReceipt` classifies them as typed operators and the metric
receipt records compatible promoted corrections. Invalid fragmentation remains
blocked.

The intended P3 carrier order is:

```text
SeamGraph
  -> SeamCutPanelization
  -> MetricCorrectionReceipt
  -> PanelUnwrapReceipt
  -> ManufacturingReceipt
```

`MetricCorrectionReceipt` is a separate sidecar/consumer carrier, not a mutation
of `SeamCutPanelization`. It defines local result states
`correctionOk`, `correctionDegraded`, `correctionRejected`, and
`correctionAbstained`, plus typed blockers for missing `Delta g` meaning,
missing metric propagation law, missing shaping intent, missing panel unwrap
compatibility, and manufacturing review. Disk-like panels may unwrap without a
metric-correction success claim; branchy/open panels may promote only when
required corrections are typed, receipted, admissible, and consumed by panel
unwrap.

End-to-end SMII reading:

```text
input body / pose / fabric / task
  -> receipted BodyCarrier
  -> finite KernelBasis over the body
  -> ROMOperator for pose/task projection
  -> ProjectedField tension / pressure / shear / support
  -> CouplingCocycle debts and residuals
  -> weighted SeamGraph over the body carrier
  -> SeamCutPanelization into graph-cut panels
  -> optional MetricCorrectionReceipt for darts/ease/shaping
  -> PanelUnwrapReceipt
  -> ManufacturingReceipt
  -> promoted artifact only if all required gates are admissible
```

This reading keeps "where the fabric is cut" separate from "why this panel
needs shaping correction." It also keeps the formal claim theorem-thin: a
declared and receipted metric correction is not a proof of physical safety,
body fit, or manufacturing validity.

Reusable DASHI surfaces for later formalization:

- `DASHI/Foundations/QuotientSetoidSurface.agda`: quotient-stable transport and
  norm surfaces for ROM compression cells, equivalent pose regions, panel
  equivalence, and seam-cost invariance.
- `DASHI/Interop/ObservationTransportSpine.agda`: observation/transport
  surfaces, lossy quotient boundaries, joint observation narrowing, and
  non-claim governance for body/ROM observations.
- `DASHI/Metric/FibrePressureMetricBridge.agda`: residual-budget and
  candidate-only promotion patterns for pressure/tension/seam-fabric debt.
- `DASHI/Core/UniversalOperatorBasis.agda`: join and coordinate-transport
  vocabulary for future body-space/ROM-space constraint merging.
- `DASHI/Core/AuthorityBoundary.agda`: citation/reference authority is not
  artifact/manufacturing authority.
- `DASHI/Combinatorics/TriadicVideoCodecObservationQuotient.agda`:
  side-information-backed witnesses and non-promotion certificates analogous
  to ROM compression/hypervoxel side information.

`../animalexic` provides a concrete runtime governance analogue, not a garment
topology solver. It has no `.agda` files, but its documented IR and guard
scripts separate numeric kernel output from canonical promoted state:

```text
frames / stereo / promoted depth
  -> candidate 3D points
  -> voxel or surfel accumulation
  -> guarded grounded / plateau / ascended states
  -> selected object / Poisson backend
```

Its reconstruction math is extrinsic: photometric/stereo residuals, camera
projection, voxel or surfel evidence, temporal support, neighbor support,
threshold gates, and optional Poisson reconstruction. SMII's math is intrinsic:
promoted body carrier, body-attached fields, seam graph, panel cuts, metric
corrections, and 2D garment development. Animalexic can feed a better candidate
or promoted `BodyCarrier`, but it must not bypass SMII's carrier, ROM, field,
topology, correction, unwrap, or manufacturing receipts.
