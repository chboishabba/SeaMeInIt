# Solver / Kernel Roadmap Note

Date: 2026-03-10

This note distills four refreshed archived threads into concrete roadmap
guidance for the ROM kernel and seam solver work:

- `Branch · Three-kernel coupling for ROM`
  - online UUID: `696f0c80-f2e0-8322-b8a3-7b59b1ce3835`
  - canonical: `2732a8b3196238d99153d6dfe71992a95d59bd7e`
- `Pose Sweep Strategy`
  - online UUID: `69707049-9248-8323-b22d-efb493470795`
  - canonical: `5fe149c7b3c1e841ab0f8e6419b9fd225a3f5db9`
- `Seam Walker Troubleshooting`
  - online UUID: `698d5e21-6d54-839a-a127-088c1dc21227`
  - canonical: `0eff7f41332ca191629d9246ad3677518461fa55`
- `Seam Graph Generation Debug`
  - online UUID: `699050a6-e13c-839a-9a66-be7653b4db13`
  - canonical: `6d14ca5f93671d7fb8e923db48654ecb5ef63b42`
- Follow-up context refresh on 2026-06-04 also resolved:
  - `Repo planning blockers`
    - online UUID: `6a03e573-caa4-83ec-83ed-af05b723ed4c`
    - canonical: `9382ee2cba0c06880b8d351e2055acb49e97a12d`
    - relevant stitched range: `16670-16830`

## Decisions

1. ROM remains an operator over admissible pose space, not a mesh/orbit artifact.
- The kernel roadmap should continue to treat ROM as:
  - pose schedule,
  - sampled field/coefficient artifacts,
  - completeness / certificate logic.
- `human` / `ogre` outputs are not the ROM object and should not be used as the semantic reference for kernel correctness.

2. The current kernel should be judged against explicit field semantics, not visual intuition alone.
- Current finite-difference kernel computes a motion-direction-gated sensitivity field.
- Roadmap implication:
  - keep `seam_sensitivity` as one operator output,
  - but compare it explicitly against alternate candidate fields such as:
    - displacement magnitude,
    - derivative magnitude,
    - chain/legality-weighted variants.
- Kernel work should answer:
  - which field best represents the design signal we actually want,
  - rather than assuming the current field is final because it is implemented.

3. No-op detection must be a first-class invariant in ROM/body reruns.
- If mesh / hotspot outputs look unchanged, treat the run as a no-op until mtimes
  and content hashes prove otherwise.
- Roadmap implication:
  - every body/ROM/seam pipeline stage should emit stable hashes/lineage,
  - run reports should surface whether outputs were regenerated or reused.

4. Seam quality should be improved through structural constraints, not anatomy folklore.
- Do not encode “crotch bad”, “mouth bad”, etc. as special-case costs.
- Instead, solver work should focus on:
  - loop / panel constraints,
  - valid cut-graph structure,
  - flattenability / sewability objectives,
  - explicit fragmentation control.

5. Starburst/porcupine failure is a real optimization pathology and should be treated as such.
- Optimizing distortion without cut-complexity control encourages over-fragmented,
  semantically useless seams.
- Roadmap implication:
  - add chart/cut complexity regularization,
  - add flattenability-aware scoring,
  - do not accept “low distortion” alone as success.

6. Darts and shaping are typed metric-correction operators, not just seam graph noise.
- The archived `Repo planning blockers` range frames darts, gathers, easing,
  panel shaping, stretch zoning, variable knit, pleats, gussets, and bias
  orientation as implementations of controlled metric mismatch injection.
- A 2026-06-04 check of `../../dashi_agda` found that the current
  SeaMeInIt-facing Agda surface is still receipt-thin and graph/panelization
  oriented: `SeamGraph`, `SeamCutPanelization`, `topologyGate`, and
  `panelizationGate` exist, but no `Dart`, `MetricCorrection`, `Delta g`,
  wedge-removal, or curvature-insertion type exists yet.
- The adjacent DASHI `MeasurementSurface -> ProjectionResult` contract is the
  useful template for adding this: Delta semantics, metric propagation,
  diagnostics, degraded/rejected states, and claim boundaries must be explicit
  before theorem-adjacent consumers can use the result.
- `DASHI/Physics/Closure/CrossDomainVariationalSpine.agda` provides the closest
  conceptual reusable shape: typed objects with delta/projection/defect/gate
  and observation quotient. Metric corrections should mirror that vocabulary
  without importing a domain-specific clothing theorem that does not exist.
- Reusable DASHI lemma/surface stack for future formal work:
  - `QuotientSetoidSurface`: quotient-stable ROM compression, equivalent pose
    regions, panel equivalence, and seam-cost/norm invariance.
  - `ObservationTransportSpine`: observation/transport/non-claim governance for
    body fit, garment fit, projected fields, and non-inverse recovery claims.
  - `FibrePressureMetricBridge`: residual budgets and candidate-only promotion
    gates for seam/fabric pressure and tension obligations.
  - `UniversalOperatorBasis`: join and coordinate-transport vocabulary for
    merging seam costs or body-space/ROM-space constraints.
  - `AuthorityBoundary`: citation/reference authority does not imply artifact
    or manufacturing authority.
  - `TriadicVideoCodecObservationQuotient`: side-information-backed witness and
    non-promotion certificate patterns for ROM compression side information.
- Roadmap implication:
  - model garment panels as `(u, Delta g)`, not as a UV map alone,
  - treat darts as discrete curvature operators / local wedge-removal
    compensation,
  - classify cut-graph branches as ordinary boundaries, typed
    dart/relief/gusset/easing operators, or invalid fragmentation before
    promoting unwrap,
  - add a receipt-level metric-correction contract before allowing typed
    dart/relief/gusset/easing structures to satisfy panel unwrap gates,
  - define local result states `correctionOk`, `correctionDegraded`,
    `correctionRejected`, and `correctionAbstained`,
  - define typed blockers such as `missingDeltaMetricMeaning`,
    `missingMetricPropagationLaw`, `missingShapingIntentReceipt`,
    `missingPanelUnwrapCompatibility`, and
    `missingManufacturingReviewReceipt`,
  - distinguish `correctionRequired=false` from an admissible correction so
    ordinary disk-like panels do not pretend to carry shaping operators,
  - avoid solving `open_or_branched_seam_graph` by pruning all branches without
    preserving typed garment intent.

7. Animalexic is the runtime reconstruction-governance analogue, not the SMII topology model.
- `../animalexic` has no Agda surface, but its docs/runtime provide a concrete
  candidate/promotion discipline: kernels emit candidates, host/DASHI owns
  promotion, abstain/reject are explicit, and replayable guards check residual,
  temporal, multi-view, confidence, and hard invariants.
- Its math is extrinsic reconstruction:
  image/stereo/depth evidence, camera projection, voxel/surfel accumulation,
  residual thresholds, temporal support, neighbor support, and optional Poisson
  reconstruction.
- SMII's math is intrinsic garment development:
  receipted body surface, body-attached basis, ROM/projected fields,
  seam/cut topology, metric corrections, panel unwrap, and manufacturing gates.
- Roadmap implication:
  - animalexic can feed a candidate or promoted `BodyCarrier`,
  - it must not bypass SMII gates for seam topology, darts, unwrap, or
    manufacturing,
  - copy its candidate-first runtime discipline for seam solver output and dart
    candidates.

8. Morphology outcomes and artifact labels must be separated explicitly.
- Historical runs produced visually distinct outcomes now described as:
  - `ogre-like`: stretched/compressed face-heavy or non-human-looking aggregate
    body appearance,
  - `flailing`: pose/deformation outputs that look more like unstable or extreme
    motion than a settled transformed morphology.
- These are debug observations, not desired targets.
- Roadmap implication:
  - docs and reports must record where each observed morphology appears:
    body fit, ROM sample pose, aggregate field render, seam overlay, or
    cross-topology reprojection,
  - stage/topology labels such as `fit_v3240`, `base_layer_v9438`, `rom_*`
    should remain the primary artifact identity,
  - `human` / `ogre` names should not be trusted without an accompanying
    morphology note and lineage details.

## Near-Term Actions

1. Kernel track:
- Keep the new kernel diagnostic page as the reference inspection surface.
- Compare `seam_sensitivity` against at least one alternate candidate field on the
  same topology before changing solver objectives.

2. Cut-topology / metric-correction track:
- Add a receipt schema for garment metric corrections before panel unwrap
  consumes branchy/open topology as authorized dart or relief intent.
- Suggested minimum vocabulary: `dart`, `relief_cut`, `gusset`, `easing`,
  `stretch_zone`, `variable_knit`, `pleat`, and `bias_orientation`.
- Suggested minimum fields: affected panel/faces/edges, correction kind,
  local `Delta g` or declared proxy, metric propagation/energy diagnostics,
  gate state, blocker list, and source receipt hashes.
- Aggregate `metricCorrectionGate` should be derived from required correction
  gates where feasible; if stored, it must be accompanied by an aggregation
  receipt explaining how it was computed.

3. Solver track:
- Investigate shortest-path insensitivity under fixed topology.
- Prioritize:
  - anchor/component fallback debugging,
  - comparison against `mincut` and `pda`,
  - loop/panel constraints as problem-definition controls.

4. Sewability track:
- Add explicit flattenability / fragmentation terms to the seam roadmap.
- Treat “porcupine seams” as a failure of the objective, not a weird but acceptable output.

5. Morphology debugging track:
- Add a dedicated note/report section that logs, per run:
  - whether the output looks neutral-human, ogre-like, or flailing,
  - whether that appearance comes from body geometry, posed samples, field
    rendering, or reprojection/render normalization,
  - which artifact should be treated as authoritative for that observation.

## Current Practical Conclusion

The kernel question and the solver question are separable:

- Kernel side:
  the current field is mathematically coherent, but may not be the final design signal.
- Solver side:
  current shortest-path behavior appears too insensitive to cost-field changes on a fixed topology, so solver diagnostics and structural constraints are the next priority.
