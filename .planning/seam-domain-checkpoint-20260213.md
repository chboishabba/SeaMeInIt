# Seam Domain Checkpoint Plan (2026-02-13)

## Phase

Seam pipeline stabilization: topology lineage + solve-domain decision.

## Objective

Select and freeze one canonical seam solve domain policy, then treat any
cross-topology transfer as strictly gated diagnostic output.

## Scope

- In scope:
  - seam pipeline intent/observation alignment,
  - topology lineage clarity (`3240` vs `9438` branches),
  - decision protocol for `solve-on-base` vs `solve-on-ROM`.
- Out of scope:
  - new seam objective terms,
  - new fabric models,
  - UI redesign.

## Constraints

- Do not treat reprojection as semantic equivalence without quality pass.
- Preserve artifact reproducibility (timestamped outputs, provenance fields).
- Keep documentation ahead of implementation.

## Success Criteria

1. Canonical policy chosen (A or B) and documented.
2. Acceptance metrics for transfer quality are explicit and enforced.
3. Every run carries a lineage manifest with mesh/cost/hash/vertex-count chain.
4. A-vs-B comparison protocol produces a deterministic go/no-go decision.

## Assumptions

- Historical artifacts from different topology branches will continue to exist.
- Both branches may remain useful for short-term diagnostics.
- User-observed morphology notes are critical acceptance signals, not optional context.

## Open Questions

1. Is base-domain interpretability more valuable than ROM-domain physical fidelity?
2. Should reprojection ever be allowed for production decisions, or diagnostics only?
3. What threshold values are acceptable for transfer quality in this project?

## Immediate Next Action

Run the A-vs-B protocol with matched solver settings and fixed seeds, then
record a formal decision in `docs/seam_pipeline_intended_vs_observed.md`.

## Checkpoint Exit Rule

Do not proceed to additional seam-semantic tuning until the canonical solve
domain policy is accepted and documented.
