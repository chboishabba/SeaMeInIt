# SeaMeInIt Roadmap Index

Status date: 2026-06-04

This is the canonical roadmap entrypoint. Older roadmap notes remain useful as
lane history, but priority decisions should start here.

## Current State

The Gate 0-7 receipt chain exists as a constraint system:

```text
BodyCarrierReceipt
  -> TransformReceipt / CorrespondenceReceipt
  -> BasisReceipt
  -> ROMFieldReceipt
  -> SeamCostReceipt
  -> SolverPromotionReceipt
  -> CutTopologyReceipt
  -> MetricCorrectionReceipt
  -> PanelUnwrapReceipt
  -> ManufacturingReceipt
```

The current production blocker is not the receipt order itself. The bundled
production-style Afflec smoke path blocks at Gate 0 because unanchored
measurement refinement can replace a plausible image fit with an implausible
shape candidate and a skull residual above the conservative gate. The curated
seven-image P3 lane is a separate promoted control; it does not authorize the
smoke-path refinement policy.

## Priority Order

### P0 - Restore Reproducible Quality Gates

Status: first patch landed. `dev` and `test` extras now exist, `requirements-dev`
points at those extras, and generated `exports/*` are ignored while the tracked
test fixture remains visible. The sibling venv still needs Ruff installed before
the Ruff gate can run.

Before deeper production work, keep the documented local gates unambiguous:

- document that the intended local runtime is the sibling ITIR venv
  (`../.venv`) or make `pip install -e .[dev,test]` real,
- declare the test/lint/type dependencies that the repo requires,
- keep `../.venv/bin/python -m pytest --maxfail=1 -q` collecting cleanly,
- make Ruff and mypy invocations reproducible,
- decide whether CI lives in this repo or in an external runner.

Reason: an unrunnable quality gate makes every roadmap claim weaker.

### P0 - Resolve Repository Governance Hazards

The repo policy says not to commit binaries, but tracked generated/binary-like
files already exist under `outputs/`, root Afflec images, `exports/`, and docs.
Do not add more binary artifacts. Replace tracked generated outputs with
regeneration commands and clear fixture policy in a dedicated cleanup change.

The license surface also needs a decision: `LICENSE` and README currently imply
different grants and restrictions.

### P1 - Promote Gate 0 Body Trust

Make Gate 0 refinement an explicit candidate-authority boundary before trying
to make the Afflec smoke path promote:

- normalize and hash the effective measurement model actually consumed by the
  solver; remove duplicated authority and honor its configured scale rule,
- constrain measurement refinement around image-derived betas with a declared
  beta domain, prior, anchor, and recomputed residuals,
- emit a hash-linked refinement policy/receipt with `promote`, `abstain`, or
  `reject`; a solver candidate has no authority to replace the canonical body,
- add `BodyCarrierReceipt` v2 lineage: canonical source, selected pre-repair
  checkpoint, repaired export, and refinement-receipt hash,
- separate monotone diagnostic severity from refinement and body authorization:
  warnings remain visible but do not automatically veto every body decision,
- freeze a smoke abstention/non-promotion test and a curated-P3 promotion test
  under the new policy.

Reference policy and P3 handoff: `docs/roadmap/gate0_reference_and_p3_handoff.md`.

### P1 - Strengthen Receipt Consumption

The emitters enforce many local rules, but consumers still trust loaded receipts
too easily. Add machine-readable receipt schemas or equivalent strict validators
and make the DAG reader verify hash/provenance chains, not only promotion
integers.

### P1 - Freeze Solve-Domain Policy

Strategy A is the provisional production path: solve on the promoted physical
carrier (`A_v3240`). Strategy B remains diagnostic until transfer/correspondence
receipts prove `A_T=+1` with acceptable residuals, retention, collapse, and
lineage metrics.

### P2 - Replace Bootstrap Geometry And Field Implementations

Once Gate 0 and receipt trust are stable:

- replace the sinusoidal/QR basis with cotangent Laplace-Beltrami modes,
- replace synthetic ROM fields with real `sampler_real` corpus aggregation,
- validate the new real LSCM Gate 6 backend on P3/Afflec runs and add real
  ABF/ARAP behavior,
- wire Gate 7 into cutter-ready SVG/PDF/DXF exports with explicit geometry.

### P2 - Panel And Manufacturing Quality

Continue the panel layer work after the upstream gates are trustworthy:

- complete seam reconciliation,
- add panel-specific split strategies,
- finish variable seam allowance and cutter-readiness checks,
- test generated reports and exports for escaping, invalid markup, and path
  safety.

### P2 - Materialize Sequential Panel Search Before BT369

`smii.operator_basis_search.v1` only composes independently measured deltas
against the original panel. Its depth-two candidates are unordered,
cross-family, and non-materialized. The historical field
`basis_exhausted_at_depth` therefore does not prove sequential operator-basis
exhaustion.

Build a v2 search over materialized `PanelSearchState` transitions: ordered
operators, recomputed residuals and admissible children after each operation,
backend re-runs, deterministic bounded beam selection, topology/hash
deduplication, and full lineage. Its conclusion must be bounded by the
operator generator, backend set, policy hash, depth, beam width, and
deduplication rule. After the search, emit a multi-label diagnosis of operator
expressivity, serialization-backend, panelization, or physical/policy
infeasibility before choosing a BT369-native garment serializer.

## Known Bad Cases To Track

- `scripts/solve_seams.py` records solver modes that do not yet change the
  actual solve path for all advertised modes.
- `scripts/unwrap_panels.py` now has a real NumPy `lscm` backend, but ABF/ARAP
  remain pending and should not be exposed as fake promotion labels.
- `generate_undersuit()` can still emit without a body receipt unless
  `require_body_receipt=True`.
- Loaded promoted receipts can pass consumer checks even when producer-only
  invariants would have rejected them.
- `exports/` is not ignored even though export commands write binary/vector
  artifacts there.

## Roadmap Surfaces

- Production receipt chain: `docs/smii_production_roadmap.md`
- Receipt policy: `docs/receipt_orchestrator.md`
- Body fit and inverse/back-transfer: `docs/body_fit_and_inverse_roadmap_20260311.md`
- Seam-domain observations: `docs/seam_pipeline_intended_vs_observed.md`
- ROM levels and sprint work: `docs/rom_levels_spec.md`, `docs/sprint_rom_l1.md`
- Historical root roadmap: `ROADMAP.md`
- Current implementation backlog: `TODO.md`
- Audit findings: `docs/roadmap/audit_findings_20260604.md`
- Gate 0 reference and P3 handoff:
  `docs/roadmap/gate0_reference_and_p3_handoff.md`
