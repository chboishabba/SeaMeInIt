# Flattened Panel Export

The undersuit pipeline now produces flattened panels through the
`SeamGenerator` → `PatternExporter` flow defined in
`src/suit/seam_generator.py` and `src/exporters/patterns.py`. Panels are
segmented by measurement loops, annotated with seam metadata, and then unwrapped
with either the lightweight plane projection backend or the conformal LSCM
solver.

Backend choice is only one layer of the formal unwrap problem. The local
`smii.seams.unwrap_benchmark` module now provides a graph/ultrametric benchmark
for sphere-to-rectangle candidates. It scores edge-length residual, area
residual, angle residual, foldover/non-injectivity, aggregate residual, and
agreement-depth distance, then ranks rectangle unwrap strategies. This gives the
project a regression target for the DASHI-style graph/ultrametric formalism
without claiming that a curved sphere admits a zero-distortion rectangular
flattening.

## Automatic export via `generate_undersuit`

Running the standard undersuit CLI now emits ready-to-cut pattern files
alongside the meshes. Use the `--pattern-backend` flag to select the flattening
method:

```bash
python -m smii.pipelines.generate_undersuit \
  outputs/afflec_demo/afflec_body.npz \
  --measurements data/measurements/sample.json \
  --pattern-backend lscm
```

Outputs land under `outputs/suits/<record>/`:

- `patterns/undersuit_pattern.(svg|dxf|pdf)` – flattened panels in vector
  formats.
- `metadata.json["patterns"]` – records the backend, panel names, warning
  flags, measurement loops, and seam attributes used during flattening.
- `panel_payload.json` / `seams_payload.json` – JSON sources that can be
  re-flattened later or re-meshed with different tolerances.

Every run recomputes measurement loops, seams, and panel slices from the base
layer, so the new method stays in sync with updated body metrics without manual
editing.

## Re-flattening or iterating on seam allowances

Use the dedicated pattern exporter CLI when you need to tweak seam allowances,
swap backends, or regenerate files without rerunning the full undersuit solve:

```bash
python -m smii.pipelines.export_patterns \
  --mesh outputs/suits/afflec_bodysuit/panel_payload.json \
  --seams outputs/suits/afflec_bodysuit/seams_payload.json \
  --backend lscm \
  --scale 1.0 \
  --seam-allowance 0.01 \
  --pdf-page-size a4 \
  --annotate-level summary \
  --output exports/patterns/afflec_bodysuit
```

This command accepts the serialized panel payload and seam overrides produced by
the pipeline. Any panel missing an explicit entry in the seams JSON inherits the
`--seam-allowance` default provided on the command line.

## Payload schema cheatsheet

- `panel_payload.json` contains `{"panels": [{"name": "...", "vertices": [...],
  "faces": [...]}]}`. Vertices are 3D coordinates in metres; faces form a
  triangle fan that defines the developable patch passed to the backend.
- `seams_payload.json` maps panel names to metadata. At minimum set
  `"seam_allowance": <metres>`. Optional keys include `seam_partner` and
  `seam_length_tolerance` for seam length reconciliation. Use `seam_partners`
  to author explicit edge-to-edge intent; each entry captures edge index ranges
  on the ordered outline and the partner panel/edge. If `seam_partners` are
  present, the pipeline derives `seam_avoid_ranges` for seam-aware splitting
  unless you set `seam_avoid_ranges` explicitly. `seam_midpoint_index` can still
  override split placement when a seam edge is meant to be the cut axis.
  Additional keys are preserved inside exported metadata so downstream tooling
  can distinguish stitch types or reinforcements.

Example seam partner entry (edge indices refer to the ordered outline used by
the flattening backend):

```json
{
  "seam_partners": [
    {
      "edge": [12, 20],
      "partner_panel": "panel_B",
      "partner_edge": [4, 12],
      "role": "primary",
      "zone": "side_torso"
    }
  ]
}
```

You can hand author these files for custom garments as long as they keep the
same structure.

## Demo output model

The current demo output model is the panel payload emitted by
`src/smii/pipelines/generate_undersuit.py` and exercised in
`examples/undersuit_pattern_export.py`. It matches the `PanelPayload` dataclass
in `src/suit/panel_payload.py` and the exporter `Panel3D` shape, even before the
full `Panel` schema (see `CONTEXT.md` lines 564-604) is wired through.

A recent run example is `outputs/suits/afflec_body/metadata.json`, where
`patterns.panels` lists the exported panel names alongside file outputs.

## Outline cleanup and annotations

Flattened outlines are post-processed to keep sewing patterns clean and usable:

- consecutive duplicate vertices are removed
- extreme outlier edges are dropped when they exceed 3x the median edge length
- boundaries are Laplacian smoothed with an interior constraint to avoid drift
- polylines are simplified with Douglas-Peucker to reduce jagged nodes

Annotation metadata can include `grainline`, `notches`, `folds`, and `label`
entries. The exporter renders these as dedicated layers in SVG/DXF/PDF output,
with `panel-outline` drawn from the seam outline and explicit `seam-outline` and
`cut-outline` layers showing seam vs. seam-allowance geometry.

## Backend, scaling, and warnings

- `simple` backend performs a PCA-based projection and is fast enough for smoke
  tests.
- `lscm` backend requires NumPy and python-igl (libigl bindings) but keeps
  angular distortion bounded and should be used for production patterns.
- `smii.seams.unwrap_benchmark` compares graph/ultrametric rectangle unwrap
  candidates against numerical backends before promotion policy treats a backend
  as production-ready.
- `scale` globally scales panel outlines before writing SVG/DXF/PDF.
- Per-panel seam allowances override the exporter default; mismatches are
  surfaced via `metadata["panel_warnings"]`.
- Regularization issues are exported as `metadata["panel_issues"]` with severity
  and per-issue fields, plus `metadata["panel_issue_summaries"]` for rollups.
- When auto-splitting is enabled, `metadata["auto_split"]` records the split
  strategy (single/seam-aware/multi-cut, or mixed) and resulting panel count.
- Per-panel seam metadata can set `split_strategy` (`single_cut`,
  `seam_aware`, `multi_cut`) to override adaptive auto-split behavior.
- Seam length mismatches emit `SEAM_MISMATCH` issues when `seam_partner` metadata
  is present.
Auto-split guidance follows `CONTEXT.md` lines 4555-4675 for pipeline gating.

The exported PDF includes a textual summary of all panels, seam allowances, and
the backend used. SVG and DXF files embed seam allowance values as attributes so
local CAD or nesting software can pick them up automatically.

## P3 diagnostic cut-sheet outputs

`scripts/run_p3_afflec_transfer_chain.py` emits diagnostic pattern views after
`scripts/unwrap_panels.py` writes `panel_uvs.npz`, even when panel unwrap blocks
manufacturing. These files live under
`outputs/p3_afflec_transfer_chain_20260604/panel_unwrap/diagnostics/` by
default:

- `mesh_seam_overlay.png` renders the promoted 3D body mesh with solver seam
  edges overlaid.
- `panel_uv_diagnostic.(svg|png)` shows only raw UV point clouds plus convex
  hulls and measured distortion color. It is not the face-backed panel
  topology, and it should not be read as the unwrapped cut shape.
- `diagnostic_flat_cut_sheet.svg` shows the face-backed flat cut sheet: UV
  triangle mesh, patch boundary edges, solver seam edge segments, grainline,
  labels, and per-panel distortion.
- `diagnostic_2d_patterns.svg` remains a coarse hull preview only.
- `diagnostic_pattern_summary.json` hashes the source panel receipt, UVs, mesh,
  seam edges, and diagnostic outputs, and records
  `manufacturing_authorized=false`.

These diagnostics answer the practical question "what would this currently ask
someone to cut out?" They are deliberately not `cutting_layout.svg` and do not
produce `manufacturing_receipt.json`. The summary distinguishes solver seam
edge segments from true cuttable pattern pieces via `seam_graph_summary`,
`panel_seam_segment_counts`, and patch boundary counts. If the flat cut sheet is
a single distorted body-shaped patch, a starburst, or an uncut mesh, the blocker
is seam topology/darts/relief cuts before a real LSCM/ABF manufacturing unwrap.

P3 now separates this into a dedicated cut-topology receipt before unwrap:

- `solver/cut_topology_receipt.json` validates whether solver seam edge
  segments are an actual cut graph.
- `solver/dart_relief_candidates.json` is diagnostic-only. It proposes typed
  dart or relief-cut candidates from an angle-deficit developability proxy and
  paths them to an existing seam or mesh boundary.
- `panel_unwrap_receipt.json` may include `cut_topology_receipt_hash` when it
  consumed a promoted cut-topology receipt.
- `panel_unwrap_receipt.json` also nests
  `serialization_competition_receipt.failure_fields` when Gate 6c measures
  per-face serializer failures. These fields identify foldover faces,
  high-distortion faces, distortion ridge edges, and candidate
  `failure_relief_path` splits. The split is still only promoted by the existing
  competition gate: backend-serializable, lower score, and no worst-distortion
  regression.
- The 2026-06-05 P3/Afflec materialization-aware rerun records the current
  empirical boundary at
  `outputs/p3_afflec_failure_relief_20260605/panel_unwrap_fabric_full/panel_unwrap_receipt.json`:
  P0 generated a lower-score `failure_relief_path` candidate but failed chart
  validity; P2 generated a backend-valid candidate with lower worst distortion
  but higher score. Both were rejected by the unchanged acceptance rule.
- A follow-up run at
  `outputs/p3_afflec_failure_relief_20260605/panel_unwrap_fabric_tree/panel_unwrap_receipt.json`
  now emits `failure_relief_tree` as a distinct materialized variant label when
  the failure field is partitioned into multiple disconnected components. On the
  current Afflec data it still does not promote, but the receipt now exposes the
  tree-shaped search space explicitly.
- The next rung under evaluation is `failure_drain_path`: the measured failure
  island is routed toward a release sink on the panel boundary, so the parent
  surgery is a corridor-to-boundary move rather than only an island partition.
  This should be treated as a new search family, not a relaxation of the
  existing score/distortion/serializability acceptance rule.
- The first drain-family Afflec rerun at
  `outputs/p3_afflec_failure_relief_20260605/panel_unwrap_fabric_drain/panel_unwrap_receipt.json`
  keeps the same boundary explicit: P1 exposes a genuine `failure_drain_tree`
  candidate, but it is still rejected; P2 still rejects on score after the
  lower-distortion relief path; P0 does not materialize a drain candidate from
  its current failure field. The criterion remains backend-serializable,
  score-improving, and non-regressing in worst distortion.
- The tightened single-path drain selection at
  `outputs/p3_afflec_failure_relief_20260605/panel_unwrap_fabric_drain3/panel_unwrap_receipt.json`
  prefers the seam boundary and reduces the drain search to one selected path,
  but the best P1 drain variant still stays just above the original score and
  remains rejected. That is a selection boundary, not a gate bug.
- The next diagnostic rung is `VariantParetoReceipt` plus profile-relative
  selection scoring: each candidate should report which metrics it improves,
  whether it remains on the Pareto frontier against sibling variants, and
  whether a debug-geometry or manufacture-oriented profile would pick a
  different winner even when the default scalar gate still rejects it. This is
  diagnostic only; it does not loosen the existing serializability or
  distortion gate.
- The next operator-family rung is `failure_wedge_relief`: the failure field
  now derives a two-leg wedge/lens candidate from a measured failure component
  to boundary or seam-boundary sinks. The materialized split creates a separate
  wedge chart plus parent remainder charts without deleting parent fabric, then
  enters the same backend competition and Pareto/profile receipt path as the
  previous relief and drain families.
- Variant diagnostics also report the default-profile loss explanation for
  non-selected candidates (`lost_to_*` plus metric deltas). This is meant to
  distinguish "geometrically useful but too costly" from "dominated by every
  tracked metric" without changing promotion.
- The bounded wedge-family Afflec rerun at
  `outputs/demo/afflec_receipted_curated_20260605_143059/panels_fabric_wedge2/panel_unwrap_receipt.json`
  keeps the same hard boundary. P1 receives a valid bounded
  `failure_wedge_relief` candidate, but it is dominated and not useful under
  the tracked metrics. P0 and P2 do not receive a wedge candidate from the
  current failure-field geometry; P2 still carries the useful-but-not-promoted
  `failure_relief_path` boundary.
- The guided dart-wedge Afflec rerun at
  `outputs/demo/afflec_receipted_curated_20260605_143059/panels_fabric_guided_dart/panel_unwrap_receipt.json`
  adds `pareto_guided_dart_wedge`, seeded from the measured one-path relief
  receipt instead of generic boundary sinks. The candidate is bounded to a
  narrow apex-plus-two-leg dart/lens chart and preserves parent faces. P2 now
  receives a valid frontier/useful guided candidate: score `10897.938`, worst
  distortion `5.241`, two charts. It improves the original worst distortion
  (`5.411`) but still loses default-profile selection to `cutout_r2` and does
  not promote under the unchanged score-plus-distortion gate. Accepted parents
  remain P1 `cutout_r1` and P3 `relief_split`.
- The next patch-family experiment is `failure_lens_patch`: derive a bounded
  support region from the measured serializer failure component and/or the
  Pareto-useful relief path, materialize that region as its own replacement
  patch chart, keep the parent remainder face-backed, and send the combined
  atlas through the same backend competition. This is intentionally different
  from path-family relief: it tests whether isolating a local metric region as
  a patch beats another narrow line split while preserving all original parent
  faces and leaving the hard acceptance gate unchanged.
- The bounded lens-patch Afflec rerun at
  `outputs/demo/afflec_receipted_curated_20260605_143059/panels_fabric_lens_patch/panel_unwrap_receipt.json`
  keeps the accepted parents unchanged: P1 `cutout_r1` and P3 `relief_split`.
  P0 receives a valid `failure_lens_patch` candidate that improves score and
  foldovers (`7154.330`, `713`) but regresses worst distortion to `16.176`, so
  it is dominated by `relief_split` under the unchanged hard gate. P2 receives
  no bounded lens candidate from the current support generator and remains the
  Pareto-useful path/dart frontier case.
- The next rung is `OperatorBasisSearchReceipt`
  (`smii.operator_basis_search.v1`): stop treating each local parent surgery as
  a one-off experiment and retain a bounded beam over small operator trees
  composed from the already measured single-operator deltas. The first receipt
  is diagnostic only: it can show that a depth-2 tree beats all single variants
  by profile/Pareto score, or that the declared path/patch basis is exhausted
  for a hard panel, but it does not claim true sequential rematerialization
  until a native operator-tree materializer is implemented.
- The native serializer fallback should be tracked separately as a
  `bt369_pattern_serializer` target. LSCM/xatlas/bootstrap remain backend
  serializers; the BT369-native path should start from the operator atlas,
  correction prefixes, fabric cone, and residual cells instead of treating the
  chart as an ordinary UV island.
- The first operator-basis-search Afflec rerun at
  `outputs/demo/afflec_receipted_curated_20260605_143059/panels_fabric_operator_basis_search/panel_unwrap_receipt.json`
  preserves the hard boundary: accepted parents remain P1 `cutout_r1` and P3
  `relief_split`; P0 and P2 emit `basis_exhausted_at_depth=true` at depth 2.
  P2's retained diagnostic trees still expose useful geometry tradeoffs
  (`cutout_r2 + failure_relief_path` improves worst distortion but regresses
  score/foldovers; `failure_relief_path + pareto_guided_dart_wedge` improves
  worst distortion further but also regresses score/foldovers). No composed
  P0/P2 tree can promote because composed trees are not sequentially
  materialized and the measured single-operator families still miss the hard
  score/distortion/foldover gate.

The distinction is intentional: a high-curvature dart candidate is not a cut
mesh boundary and does not promote panel unwrap. It is the next input for the
solver/cutting pass when `cut_topology_receipt.json` reports an untyped open or
branched structure, `seam_graph_not_cut_graph`, or `no_cut_mesh_boundary`.

Archived context refresh on 2026-06-04 clarified the intended dart formalism
from the local canonical chat archive (`~/chat_archive.sqlite`, no web fetch):

- `Repo planning blockers`
  - online UUID: `6a03e573-caa4-83ec-83ed-af05b723ed4c`
  - canonical thread id: `9382ee2cba0c06880b8d351e2055acb49e97a12d`
  - relevant stitched range: `16670-16830`
- `Seam Walker Troubleshooting`
  - online UUID: `698d5e21-6d54-839a-a127-088c1dc21227`
  - canonical thread id: `0eff7f41332ca191629d9246ad3677518461fa55`
- `UV unwrapping explanation`
  - online UUID: `6916c180-1080-8320-ae2d-acc2e3ac3c23`
  - canonical thread id: `3562461ee45a9f6eb3b24f0cbd4a233161a7b60e`

That context sharpens the rule above: darts are not merely future seam edges or
decorative shaping. Darts, gathers, easing, stretch zoning, variable knit,
pleats, gussets, bias orientation, and panel shaping are all typed metric
correction strategies: implementations of an allowed `Delta g` field in the
garment model `(u, Delta g)`. A dart is a discrete curvature operator, not just
an open branch in the seam graph.

Therefore a blocked P3 cut topology should be repaired by classification, not
blind graph pruning. Ordinary panel boundaries must become cuttable disk
boundaries. Branches are correction-tree/operator nodes when they represent
real dart/relief/gusset intent; they must be typed, counted, and carried by
receipts before unwrap can treat them as authorized metric-correction topology.
Only untyped or accidental fragmentation remains a blocker.

Sibling DASHI/Agda check on 2026-06-04:

- `../../dashi_agda/Docs/SeaMeInItROMKernelFormalism.md` records the current
  SeaMeInIt formal surface as theorem-thin and receipt-boundary-only.
- `../../dashi_agda/DASHI/Interop/SeaMeInItROMKernelFormalism.agda` currently
  formalizes `SeamGraph` and `SeamCutPanelization`, with promotion gates for
  topology and panelization, but it does not yet define a `Dart`,
  `MetricCorrection`, `Delta g`, wedge-removal, or curvature-insertion
  operator.
- The adjacent DASHI projection contract
  (`../../dashi_agda/Docs/MeasurementSurfaceProjectionContract.md` and
  `../../dashi_agda/scripts/hepdata_projection_contract.py`) gives the
  contract pattern SeaMeInIt should mirror: Delta-bearing outputs need explicit
  semantics, metric propagation rules, diagnostics, and rejected/degraded
  states before downstream gates consume them.

P3 can use `solver/dart_relief_candidates.json` only as advisory evidence until
the candidate is represented by a promoted `MetricCorrectionReceipt`. The
receipt distinguishes ordinary cut edges from typed correction operators such
as `dart`, `relief_cut`, `gusset`, `ease`, `stretch_zone`, `variable_knit`,
`pleat`, and `bias_orientation`, then exposes a separate metric-correction gate
for unwrap/manufacturing promotion.

The intended receipt shape should stay local to SeaMeInIt while mirroring the
DASHI projection and variational-object vocabulary:

```text
MetricCorrectionResultState =
  correctionOk
  correctionDegraded
  correctionRejected
  correctionAbstained

MetricCorrectionBlocker =
  missingDeltaMetricMeaning
  missingMetricPropagationLaw
  missingShapingIntentReceipt
  missingPanelUnwrapCompatibility
  missingManufacturingReviewReceipt
  correctionNotRequiredForDiskPanel
```

The aggregate `metricCorrectionGate` should be derived from required correction
gates when practical, using the same tri-gate fold discipline as the receipt
spine. If implementation friction requires a stored field, the receipt must
also state how the aggregate was produced. Ordinary disk-like panels do not
need successful corrections; they are either outside the correction stage or
carry an explicit not-required state. Branch or open operator topology may
promote only when its required metric corrections are typed, receipted,
admissible, and consumed by panel unwrap.

The schema guidance should name the operator and fabric-relative evidence
explicitly. A `CorrectionTreeReceipt` should carry fields such as
`correction_tree_id`, `root_panel_label`, `operator_nodes`,
`parent_node_id`, `operator_type`, `branch_degree`,
`delta_metric_meaning`, `metric_propagation_law`, `energy_terms`,
`result_state`, and node-level blockers. A `CorrectionOperatorScoringReceipt`
should carry the deterministic Gate 6a pricing surface: branch id, residual
signature, candidate operator family, estimated residual/fabric violation after
the operator, sewing/manufacturing/style costs, selected operator, and whether
the candidate beats `diagnostic_carry`. A `RealizedCorrectionOperatorReceipt`
then records the artifact-changing interpretation of selected operators. The
first implemented realization is `stretch_zone`: it adds local fabric
admissibility overrides and SVG cut-sheet annotations. A `gusset_corner`
companion can then realize residual relief when stretch alone remains above the
metric residual gate. Both keep an explicit claim boundary that this is not full
polygon deformation or cloth simulation. A
`FabricAwarePanelMetricReceipt` should carry `fabric_receipt_hash`,
`panel_label`, `fabric_profile_id`, `grain_direction`,
`stretch_compliance`, `shear_compliance`, `raw_metric_residual`,
`corrected_metric_residual`, `fabric_relative_threshold`, and
`fabric_metric_gate`. Panel promotion is therefore relative to the declared
fabric carrier, not to a fabric-free geometric residual alone.

The runtime governance analogue is `../animalexic`, not because it solves
garment topology, but because it keeps numeric kernels and promoted canonical
state separate. Animalexic promotes 3D evidence through candidate voxels or
surfels, residual checks, temporal support, neighbor support, and explicit
guards. The SMII equivalent rule is:

```text
seam solver output != promoted garment pattern
dart candidate != promoted metric correction
diagnostic panel UV != manufacturing artifact
manufacturing artifact != finished body/ROM/fabric atlas receipt
```

Animalexic may provide a stronger promoted 3D surface for `BodyCarrier`, but
SMII must still run its own body, basis, ROM, field, topology, correction,
unwrap, and manufacturing gates. Darts are not voxel/surfel geometry; they are
typed variational metric corrections over a panel development problem.

`smii.seams.seam_derivation.FinishedSeamReceipt` is the runtime receipt that
composes promoted stage receipts into the final atlas serialization claim. It
records the body, ROM, fabric, and basis receipt hashes; stage receipt hashes;
selected seam count; panel count; shaping-operator counts; allowance policy;
blocker log; and claim boundary. It does not solve seams by itself and does not
override any upstream receipt gate. `scripts/generate_manufacturing_artifacts.py`
emits `finished_seam_receipt.json` when the upstream receipt paths and hashes
are supplied; `scripts/run_afflec_receipted_demo.py` supplies those paths
automatically in the native `A_v3240` demo chain after cut-topology and
metric-correction gates promote.

## PDF tiling and page size

PDF output tiles panels across multiple pages when a single sheet is too small.
Supported page sizes are `a4` (default), `letter`, and `a0`. The tiling preserves
panel coordinates so downstream assembly remains deterministic; the multi-page
PDF mirrors the pipeline order shown in `CONTEXT.md` lines 800-815.

## Troubleshooting

1. Inspect `metadata.json["patterns"]["panel_warnings"]` and
   `metadata.json["patterns"]["panel_issue_summaries"]` to catch panels that
   exceeded budgets or require review/splitting.
2. Review `metadata.json["patterns"]["measurement_loops"]` to ensure the loop
   coordinates align with the intended fit adjustments.
3. When switching backends, regenerate the `patterns/` directory to keep the
   metadata in sync—older files are not overwritten automatically.
