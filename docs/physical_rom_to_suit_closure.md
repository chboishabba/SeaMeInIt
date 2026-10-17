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

The new receipts are implemented in `smii.seams.suit_closure_receipt` and the full run-directory reader is `smii.orchestrator.read_suit_closure_dag`.

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

The intended use is to preserve/reopen triadic phase/orientation provenance as ROM hypervoxels, projected fields, fabric directions, seam choices, panel addresses, and measured wear fields are transported between representations.

## Executable emitters

`scripts/emit_suit_closure_receipts.py` exposes three subcommands:

```text
assembly
wear-fit
physical-rom
```

The script hashes the supplied evidence files instead of accepting unbound labels. This keeps physical evidence replayable and makes the post-pattern chain compatible with the existing receipt-orchestrator discipline.

## Current boundary

No physical suit measurement dataset is committed by this change. Therefore the new surfaces do not retroactively promote any existing demo run. Existing P3/Afflec panel distortion remains an upstream Gate-6 blocker where applicable; these physical receipts become reachable only after the digital chain has honestly promoted.

The next empirical max-cut is therefore:

1. clear the existing panel/metric-correction blocker without loosening its gate;
2. fabricate a promoted pattern;
3. capture assembly geometry/tolerance and material-lot evidence;
4. instrument a declared ROM/task slice and produce predicted-vs-measured fields;
5. measure C3/BT369 transport residual where that transport is used;
6. close the residual back onto ROM/basis/fabric parameters and rerun until the physical closure promotes or an honest blocker remains.
