# Receipt Orchestrator

SMII is moving from a diagnostics-heavy research flow to a gated production
pipeline. The orchestrator must treat each stage as a receipted carrier in a
DAG, not as an informal task list. Geometry may still be inspected in
diagnostic mode, but promoted seam, panel, and manufacturing artifacts must only
consume promoted upstream receipts.

## Receipt DAG

The intended promotion order is:

```text
BodyCarrierReceipt
  -> TransformReceipt / CorrespondenceReceipt
  -> BasisReceipt
  -> ROMFieldReceipt
  -> SeamCostReceipt
  -> SolverPromotionReceipt
  -> PanelUnwrapReceipt
  -> ManufacturingReceipt
```

Every receipt carries:

- stable artifact hashes or provenance references for its inputs and outputs
- residual or quality metrics for its lane
- a ternary promotion state: `-1` rejected, `0` diagnostic or not promoted,
  `1` promoted
- explicit downstream consumer blocks when the receipt is not promoted

An unreceipted object is diagnostic-only. A downstream stage may read it for
inspection, reporting, or comparison, but may not silently promote from it.

## Lane Gates

| Lane | Receipt | Entry gate | Promotion gate |
|---|---|---|---|
| Carrier trust | `BodyCarrierReceipt` | source fit/export artifacts exist | `A_body=+1`, with skull/head plausibility and export-stage hashes |
| Correspondence | `TransformReceipt` / `CorrespondenceReceipt` | source and target mesh hashes exist | `A_T=+1` for transfer-backed claims; collapsed NN maps remain diagnostic |
| Field basis | `BasisReceipt` | promoted physical carrier | `A_field_basis=+1` with stable vertex-aligned basis `B_0` |
| ROM aggregation | `ROMFieldReceipt` | basis plus ROM samples | `A_field=+1` with fields aligned to the seam graph vertex count |
| Topology/solver | `SeamCostReceipt`, `SolverPromotionReceipt` | body and field gates pass | `A_seam_cost=+1`, then `A_seam=+1` only under the solver promotion rule |
| Panel/manufacture | `PanelUnwrapReceipt`, `ManufacturingReceipt` | promoted seam topology | manufacturable artifacts, or explicit non-promotion boundary |

The solver promotion rule is deliberately strict:

```text
A_body = +1
A_field = +1
A_seam_cost = +1
and (A_T = +1 or solve_domain = A_v3240)
```

If that rule is not satisfied, solver outputs are diagnostic-only. Gate 4
therefore enforces the body, field, and solve-domain rule while computing seam
costs, not only at solver invocation time.

Solver promotion adds the final topology check before panel unwrap. A solver
may only consume a promoted `seam_cost_receipt.json`. The solver receipt records
its anchor source, anchor count, connected-component count,
`anchor_fallback_used`, seam artifact hash, total seam cost, panel count, and
whether the produced panels pass the current graph-level disk proxy. Anchor
fallback is auditable rather than silent: it does not by itself block promotion,
but it is recorded for later anchor-policy review.

## Current Gate 0

`BodyCarrierReceipt` is the first enforced receipt. It records the source,
raw-reprojection, refined-pre-repair, and repaired/export hashes; mesh counts;
topology label; landmark residuals; skull rigidity residual; confidence;
promotion state; and blocked downstream consumers.

`generate_undersuit` now accepts a body receipt and blocks before artifact
emission when the receipt is unpromoted or explicitly blocks the
`generate_undersuit` consumer. Non-promoted body receipts default their
`blocked_consumers` to:

```text
generate_undersuit
seam_cost_field
panel_unwrap
```

This makes body trust a property of the receipt itself, not only of one local
pipeline gate.

`smii.app afflec-demo` also emits the first real body receipt from the export
path. Each run writes:

```text
afflec_body_raw_reprojection.npz
afflec_body_refined_pre_repair.npz
afflec_body.npz
body_carrier_receipt.json
```

The receipt is conservative: `bbox` and warning-status runs remain diagnostic,
while only high-trust `PASS` runs with acceptable confidence and crown
eccentricity can promote.

## Minimal Reader

`smii.orchestrator.read_receipt_dag(run_dir)` reads known body,
correspondence, basis, ROM-field, seam-cost, and solver receipt files without running
tasks. It reports lane promotions, the first blocker, and seam-solver
eligibility under the strict rule above. This is intentionally a reader, not a
task scheduler; existing CLIs can use it to decide whether their outputs may
promote.

## Correspondence Policy

The repo currently has correspondence/reprojection tooling, not a proven true
inverse from an internalized ROM domain back to the fitted body. Strategy A
therefore remains the provisional production path: solve on the promoted
physical carrier and project/evaluate fields there. Strategy B can only promote
when a receipted transform or correspondence passes its residual gates.

Full-surface map load/collision metrics and seam-transfer collapse metrics must
remain separate. A dense full-surface map can have unavoidable many-to-one load
when vertex counts differ, while a seam-transfer collapse ratio measures whether
the seam topology survives reprojection. `CorrespondenceReceipt.collision_ratio`
records the full-surface map load ratio when a sampler-native map provides it;
`CorrespondenceReceipt.seam_transfer_collapse` records the seam-local collapse
ratio from `scripts/reproject_seam_report.py`. Both are useful, but they gate
different claims.

`scripts/reproject_seam_report.py` is the first correspondence emitter. When
requested, it writes `correspondence_receipt.json` after quality metrics are
known. The receipt binds source and target mesh hashes, mean/max transfer
distance, full-surface load where available, seam-local collapse, edge retention,
and `A_T`. Collapsed nearest-neighbor transfers are explicit `A_T=-1`
diagnostic-only receipts rather than silently consumable seam topology.

## Field-Basis Policy

`scripts/generate_canonical_basis.py` is the first `BasisReceipt` emitter. It
can write `basis_receipt.json` next to the generated basis artifact when called
with a promoted `body_carrier_receipt.json`. The emitted receipt hashes the
carrier receipt file itself, records the generated basis hash, vertex count,
basis width, construction method, relative reconstruction error on a static
contact-pressure proxy, the 5% relative-error promotion threshold, `A_field_basis`,
and downstream consumer blocks.

Basis generation without a carrier receipt remains available for diagnostics
and legacy plumbing, but it is unreceipted and therefore cannot promote ROM
field aggregation. A non-promoted body carrier is a hard basis-lane blocker.

The current bootstrap basis is the existing sinusoidal-feature QR basis. It is
labelled as such in `construction_method`; replacing it with a cotangent
Laplace-Beltrami eigenbasis remains the next field-basis fidelity upgrade.

## ROM-Field Policy

`examples/rom_aggregate_from_samples.py` is the first `ROMFieldReceipt` emitter.
When called with a promoted `basis_receipt.json`, it writes `rom_fields.npz`
plus `rom_field_receipt.json`. The fields artifact stores per-vertex mean,
peak, and variance arrays for every projected ROM field. The receipt hashes the
basis receipt file, the sample payload, the aggregation summary, and the fields
artifact; records sample counts, pose source, field names, vertex count, peak
pressure diagnostics, and `field_uniformity`; and gates `A_field`.

`field_uniformity` is a flat-field diagnostic: values near `1.0` mean the field
is too uniform to drive meaningful seam differentiation. Promotion requires
`field_uniformity < 0.95`. Synthetic sample payloads remain diagnostic-only by
default and can promote only when the caller passes an explicit synthetic
promotion flag.

## Seam-Cost Policy

`scripts/compute_seam_costs.py` is the first `SeamCostReceipt` emitter. It
loads a promoted `body_carrier_receipt.json` and `rom_field_receipt.json`,
optionally loads a `correspondence_receipt.json` for transfer-backed domains,
and writes `seam_costs.npz` plus `seam_cost_receipt.json`.

Native `A_v3240` solves may promote without a correspondence receipt because
the costs live on the promoted physical carrier. Transfer-backed `B_v9438`
solves must carry `A_T=+1`; rejected or diagnostic correspondence receipts
block cost promotion and force the caller back to native solving or transfer
repair.

`cost_uniformity` is the seam-solver insensitivity diagnostic. Values near
`1.0` mean the edge costs are effectively flat and cannot produce meaningful
solver differentiation. Promotion requires `finite_cost_coverage > 0.99` and
`cost_uniformity < 0.95`.

## Solver-Promotion Policy

`scripts/solve_seams.py` is the first `SolverPromotionReceipt` emitter. It
loads a promoted `seam_cost_receipt.json`, verifies the referenced
`seam_costs.npz` hash, selects anchors from field minima by default, solves a
deterministic low-cost seam edge set, and writes `seam_edges.npz` plus
`solver_promotion_receipt.json`.

Field-minima anchors replace the old geometric-only default for promoted
topology. If the selected anchor subgraph is disconnected, the script falls back
to the largest connected anchor component and records
`anchor_fallback_used=true`; this makes the previous "anchors disconnected"
warning an auditable receipt field instead of a silent solver behavior.

`panels_are_disks` is the hard Gate 6 boundary. The current emitter uses a
graph-level disk proxy until the panel unwrapper owns exact cut-mesh topology:
every post-cut component must be nonempty and have Euler characteristic at
least one. Failed topology leaves the seam artifact diagnostic-only and blocks
`panel_unwrap`.

## Panel-Unwrap Policy

`scripts/unwrap_panels.py` is the first `PanelUnwrapReceipt` emitter. It loads
a promoted `solver_promotion_receipt.json`, verifies that the referenced
`seam_edges.npz` matches `SolverPromotionReceipt.seam_hash`, rejects incomplete
topology before flattening, and writes `panel_uvs.npz` plus
`panel_unwrap_receipt.json`.

The current emitter is a dependency-light bootstrap unwrapper: it accepts the
`lscm`, `abf`, and `arap` method labels for the receipt-facing interface, then
uses deterministic panel-local projection and edge-length distortion checks to
gate promotion. The full conformal exporter backends remain a downstream
integration target for richer production pattern artifacts.

`panels_are_disks=false` is a topology error, not an unwrapper failure. The CLI
must say this explicitly so the fix is to add or repair seam cuts before
flattening. A promoted panel unwrap receipt requires every extracted panel to
stay within the configured distortion threshold after any allowed subdivision
iterations. The receipt records per-panel distortion, worst/mean distortion,
subdivision usage, grain directions, the UV hash, and the solver seam hash.
Non-promoted unwraps block `manufacturing`.

## Scheduling

Carrier trust, correspondence, and field basis can be developed in parallel
because they feed different early receipts. Promotion still proceeds in DAG
order:

1. establish or reject `A_body`
2. establish transfer admissibility or freeze native solve-domain policy
3. build `B_0` and promote a field basis on the promoted carrier
4. aggregate ROM fields from the receipted basis
5. compute promoted seam costs from receipted fields and solve-domain receipts
6. promote solver outputs from promoted seam costs
7. unwrap only promoted topological disk panels
8. manufacture only promoted panel unwrap artifacts

The orchestrator is complete when seam artifacts no longer promote from an
untrusted body, transfer-backed claims are hash- and residual-bound, seam costs
come from receipted fields, and manufacturing artifacts are only emitted from
promoted topology.
