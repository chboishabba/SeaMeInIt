# Morphology Debugging Architecture

Date: 2026-03-10

## Surfaces

### 1. Canonical context

- `COMPACTIFIED_CONTEXT.md`
- `docs/seam_pipeline_intended_vs_observed.md`
- `docs/solver_kernel_roadmap_note_20260310.md`

These define the meaning of morphology categories and the current roadmap order.

### 2. Run-level morphology ledger

- `scripts/render_run_reference.py`
- `run_reference/run_report_manifest.json`

This is the first user-facing diagnostic surface. It must answer:

- which stage an artifact belongs to,
- whether geometry changed,
- what morphology was observed,
- what morphology was expected,
- what kind of artifact it actually is.

### 3. Per-run override source

- `morphology_observations.json`
- `morphology_audit_overrides.json`

These files allow a run to carry explicit human-judged morphology observations.
The run page should consume them rather than forcing morphology to remain
`unclassified`.

### 4. ROM sample morphology outputs

This surface now exists in a minimal implemented form.

Current implementation:

- `smii.rom.sampler_real --out-rom-samples-dir`
- `rom_samples/rom_sample_manifest.json`
- sampler-native posed sample `.npz` meshes

Current contract:

- representative poses are selected deterministically from sampler statistics
- emitted samples stay on the sampler-native topology
- run/reference and operator-report surfaces display them as morphology-stage artifacts, not as operator-only fields

Remaining extension space:

- richer rendered previews of the sample meshes
- tighter integration with later kernel-comparison pages

### 5. Inverse/back-transfer contract

This is now documented explicitly, even though a true inverse still does not exist.
The architecture must distinguish between:

- exact inverse transform,
- approximate reconstruction,
- correspondence/reprojection transfer.

Current state:

- current capability audit completed
- acceptance contract documented
- future inverse requirements documented

Reference note:

- `docs/rom_sample_morphology_and_transfer_contract.md`

## Dependency Order

1. Morphology ledger exists.
2. Known runs are annotated.
3. ROM sample morphology artifacts are emitted.
4. Inverse/back-transfer contract is clarified.
5. Candidate ROM fields are judged against those artifacts.
6. Seam solver sensitivity is revisited.

## Orchestration Note

For the current morphology-debugging phase, generic repo-audit work is not the
active gating step. The active gate is whether morphology attribution and
inverse/back-transfer semantics are clear enough to support implementation.

Operationally:

- the orchestrator state should not keep routing to `repo-auditor` while the
  phase is already executing against an approved morphology roadmap,
- the next bounded steps should route into implementation-oriented work on
  `M2.1` and `M2b.1` instead.

## Failure Modes

- Labels are trusted without stage evidence.
- Geometry-changing and non-geometry-changing artifacts are mixed.
- ROM aggregate fields are mistaken for transformed morphology.
- Solver outcomes are used as the primary evidence for ROM validity before
  morphology attribution is mature.
