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

### 4. Future ROM sample morphology outputs

These are not implemented yet, but they are the next architectural dependency.
They should emit representative posed/deformed sample meshes or renders so the
pipeline can show where flailing appears.

Required sub-surfaces:

- sample selection policy:
  which poses become "representative" artifacts
- sample artifact contract:
  mesh, render, metadata, and lineage expectations
- run/report integration:
  where those artifacts appear in `run_reference/index.html` and related pages

### 5. Inverse/back-transfer contract

This is also not implemented as a true inverse today, but the architecture now
needs an explicit distinction between:

- exact inverse transform,
- approximate reconstruction,
- correspondence/reprojection transfer.

Required sub-surfaces:

- current capability audit:
  what the codebase can actually do today
- acceptance contract:
  when approximate transfer is considered good enough for interpretation
- future inverse track:
  what would have to be true before calling a transform "invertible"

## Dependency Order

1. Morphology ledger exists.
2. Known runs are annotated.
3. ROM sample morphology artifacts are emitted.
4. Inverse/back-transfer contract is clarified.
5. Candidate ROM fields are judged against those artifacts.
6. Seam solver sensitivity is revisited.

## Failure Modes

- Labels are trusted without stage evidence.
- Geometry-changing and non-geometry-changing artifacts are mixed.
- ROM aggregate fields are mistaken for transformed morphology.
- Solver outcomes are used as the primary evidence for ROM validity before
  morphology attribution is mature.
