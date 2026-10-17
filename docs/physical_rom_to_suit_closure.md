# Physical ROM-to-suit closure

## Scope

The existing SeaMeInIt receipt chain proves only that a *digital* body/ROM/fabric/seam/panel/manufacturing artifact has passed its declared gates. It does not prove that a fabricated suit realizes the predicted geometry or fields.

This extension makes that boundary explicit:

```text
BodyCarrierReceipt
  -> CorrespondenceReceipt
  -> BasisReceipt
  -> ROMFieldReceipt
  -> SeamCostReceipt
  -> SolverPromotionReceipt
  -> CutTopologyReceipt
  -> MetricCorrectionReceipt
  -> PanelUnwrapReceipt
  -> ManufacturingReceipt
  -> FinishedSeamReceipt
  -> AssemblyReceipt
  -> WearFitReceipt
  -> PhysicalROMValidationReceipt
```

The new physical receipts are implemented in `smii.seams.suit_closure_receipt` and the full run-directory reader is `smii.orchestrator.read_suit_closure_dag`.

## Gate-6 operator-tree materialization

The existing `smii.operator_basis_search.v1` receipt can retain depth-2 operator trees by composing the *measured metric deltas* of single materialized surgeries. Those trees are intentionally diagnostic because metric-delta composition is not the same thing as rematerializing the geometry.

`smii.seams.operator_tree_materialization` adds the next exact rung. A `FacePartitionOperator` carries a measured operator support in the original panel face-index space. `materialize_operator_tree` applies those supports in order to the *current* chart partition:

```text
P0 = all original faces
P(k+1) = refine every chart in Pk by
         (chart intersect operator_support,
          chart minus operator_support)
```

This gives a true sequential face partition with explicit proofs-by-construction of face coverage and pairwise disjointness. Supports for the measured failure-field families are recoverable directly from the existing serializer failure receipt:

```text
failure_relief_path/tree
failure_drain_path/tree
failure_wedge_relief/tree
pareto_guided_dart_wedge/tree
failure_lens_patch
```

The companion CLI is:

```text
scripts/materialize_operator_tree.py
```

and emits `smii.operator_tree_materialization.v1`.

This receipt **always remains diagnostic** (`promotion = 0`) because topology materialization does not supply new LSCM/xatlas/bootstrap distortion, foldover, boundary, or fabric-cone measurements. Its remaining blocker is therefore explicit:

```text
backend_metrics_not_recomputed
```

A materialized tree may remove `operator_tree_not_sequentially_materialized`; it cannot remove Gate 6 until the numerical backends are rerun over the emitted charts and the unchanged acceptance gate passes.

## AssemblyReceipt

`AssemblyReceipt` is evidence that the manufactured parts were actually assembled into the intended garment realization. It binds:

- the exact `finished_seam_receipt.json` hash;
- manufactured-artifact hash;
- measured assembled-geometry hash;
- assembly evidence hash;
- piece and realized seam counts;
- material-lot provenance;
- maximum geometry deviation and its declared threshold.

Missing provenance or incomplete realization is diagnostic-only. A measured geometry deviation above the declared threshold rejects the assembly receipt.

Safe claim:

> this physical assembly realizes the promoted pattern within the declared measured tolerance.

Unsafe claim:

> the garment therefore behaves physically, clinically, or ergonomically as predicted.

## WearFitReceipt

`WearFitReceipt` compares predicted and measured garment/body fields over an explicitly declared ROM/task slice. The receipt binds:

- the exact assembly receipt;
- ROM and task-profile provenance;
- predicted and measured field artifacts;
- raw measurement evidence;
- sample count and task coverage;
- named residual channels and per-channel thresholds.

Typical channels are pressure, shear, support, tension, clearance, or temperature, but the receipt intentionally does not hard-code one sensor stack.

If evidence is missing or task coverage is insufficient, the result remains diagnostic-only. If a measured residual exceeds its declared threshold, the receipt rejects.

## PhysicalROMValidationReceipt

This receipt closes wear evidence back onto the ROM model. It binds:

- the exact wear-fit receipt;
- the source ROM receipt;
- a physical validation-evidence artifact;
- a field-residual artifact;
- closure iteration;
- an optional calibrated ROM receipt;
- an optional/required C3/BT369 transport witness.

Promotion means only that the *declared validation slice* closed under the supplied gates. It is not a universal garment-physics theorem or deployment/safety certificate.

## C3 / BT369 transport

`TriadicTransportWitness` gives C3/BT369 a concrete role in the physical closure instead of treating triadic structure as decorative metadata. It records:

```text
transport_model
address_schema
phase_provenance_hash
orientation_receipt_hash
c3_equivariance_error
c3_equivariance_threshold
```

The corresponding DASHI surface additionally types the exact mathematical contract

```text
T(g . x) = g . T(x)
```

for the chosen C3 phase transport. That equation is a transport-law contract, not empirical evidence that garment mechanics possess exact C3 symmetry. Runtime promotion therefore separately requires measured transport evidence. A transport error above threshold rejects; absent required evidence remains diagnostic-only.

The runtime witness need not be hand-authored. `smii.seams.c3_transport_measurement` fits it from paired observations. In the runtime convention:

```text
C3 phase action:       p -> p + 1 mod 3
BT369 tangent action:  o -> o + 2 mod 6
```

An equivariant offset transport has constant

```text
target_phase - source_phase          mod 3
target_orientation - source_orientation mod 6
```

across the declared correspondence slice. The estimator fits the weighted modal offsets and reports the weighted fraction of observations inconsistent with the joint offset model as `c3_equivariance_error`.

The CLI stage

```text
scripts/emit_suit_closure_receipts.py measure-c3
```

hash-binds the observation set, phase provenance, and orientation receipt and writes the measured transport witness. The subsequent `physical-rom` stage consumes that witness. This removes the need to type an equivariance error into a promotion receipt by hand.

The intended use is to preserve/reopen triadic phase/orientation provenance as ROM hypervoxels, projected fields, fabric directions, seam choices, panel addresses, and measured wear fields are transported between representations.

## Executable emitters

`scripts/emit_suit_closure_receipts.py` exposes four subcommands:

```text
assembly
wear-fit
measure-c3
physical-rom
```

The script hashes the supplied evidence files instead of accepting unbound labels. This keeps physical evidence replayable and makes the post-pattern chain compatible with the existing receipt-orchestrator discipline.

## Current boundary

No physical suit measurement dataset is committed by this change. Therefore the new surfaces do not retroactively promote any existing demo run. Existing P3/Afflec panel distortion remains an upstream Gate-6 blocker where applicable; these physical receipts become reachable only after the digital chain has honestly promoted.

The software-side max-cut now has an exact depth-N topological operator materializer, but the current generated P0/P2 evidence must still be rerun through numerical serialization to determine whether those materialized trees actually improve distortion/foldover/score. The gate is deliberately unchanged.

The next empirical max-cut is therefore:

1. derive the retained P0/P2 operator supports from the measured serializer failure fields;
2. sequentially materialize the retained trees and rerun the numerical unwrap competition over the resulting charts;
3. promote only if the existing Gate-6 score/distortion/fabric criteria clear;
4. fabricate a promoted pattern;
5. capture assembly geometry/tolerance and material-lot evidence;
6. instrument a declared ROM/task slice and produce predicted-vs-measured fields;
7. measure C3/BT369 transport residual where that transport is used;
8. close the residual back onto ROM/basis/fabric parameters and rerun until the physical closure promotes or an honest blocker remains.
