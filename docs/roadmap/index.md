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

The current production blocker is not the receipt order itself. Real Afflec
promotion is blocked at Gate 0 because the MediaPipe body-fit path emits
artifacts but measurement refinement pushes final shape betas outside the
plausible range and the skull residual remains above the conservative gate.

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

Make the Afflec body carrier promote for the production-style path, or document
the exact remaining non-promotion reason:

- add explicit reference-set quality warnings for low view diversity and
  long-lens/same-perspective risk,
- tune or constrain measurement refinement,
- keep export-stage mesh checkpoints,
- finish skull/head plausibility gates,
- add an acceptance test for the known Gate 0 non-promotion case.

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
