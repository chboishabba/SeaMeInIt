# Morphology Debugging Plan

Date: 2026-03-10

## Phase

Morphology attribution and ROM-stage observability.

## Milestones

### M1. Backfill morphology observations on reference runs

Goal:
- stop the most important run pages from defaulting to `unclassified`

Tasks:
- identify the reference runs we are actively using
- add `morphology_observations.json` into those run roots
- regenerate `run_reference/index.html`

Active reference runs:
- `outputs/comparisons/afflec_raw_vs_refined_20260309/`
- `outputs/comparisons/afflec_same_topology_20260309/`
- `outputs/comparisons/afflec_kernel_diagnostic_raw_20260310/`
- `outputs/assets_bundles/20260309_062711__afflec_raw_refined_verify/`

Done when:
- the main comparison and bundle pages state observed morphology for key artifacts

### M2. Emit explicit ROM sample morphology artifacts

Goal:
- show where flailing occurs directly

Tasks:
- M2.1 choose representative sample selection policy
  - define how many poses to emit
  - define whether selection is by coefficient norm, field energy, task tags, or
    named sweep anchors
- M2.2 define the artifact contract
  - decide mesh/render/metadata outputs
  - decide where lineage and morphology notes live
- M2.3 implement minimal sample artifact emission
  - emit at least one representative posed/deformed sample set into a run root
- M2.4 integrate those artifacts into run pages and operator reports
  - ensure the viewer surfaces show neutral body, samples, aggregate field, and seams separately

Done when:
- a run page can show neutral body, sample poses, aggregate field, and seam outputs separately

### M2b. Clarify inverse-transform status

Goal:
- stop hand-waving about "solve on ogre, invert back to body"

Tasks:
- M2b.1 audit current code paths
  - identify whether any true invertible transform exists in the current codebase
- M2b.2 document current reality
  - state whether back-transfer is only correspondence/reprojection today
- M2b.3 define acceptance for approximation
  - if no true inverse exists, define when approximate transfer is good enough
    to interpret and when it must be rejected
- M2b.4 define future inverse requirements
  - specify what evidence would be needed before claiming an inverse transform

Done when:
- docs and planning state clearly distinguish inverse-transform ambitions from
  the current approximate transfer reality

### M3. Kernel comparison on one topology

Goal:
- judge the current field against alternatives with morphology visible

Tasks:
- compare `seam_sensitivity` against displacement magnitude and derivative magnitude
- keep topology fixed
- record whether the field agrees with observed flailing/sample behavior

Done when:
- we can say whether the current kernel output is unintuitive-but-correct or the wrong design signal

### M4. Solver response after morphology attribution

Goal:
- revisit seam solver sensitivity only after upstream interpretation is clearer

Tasks:
- debug shortest-path anchor/component fallback
- compare shortest-path with `mincut` / `pda` on the same topology/cost pairs
- record whether seam changes finally track kernel differences

Done when:
- solver behavior can be interpreted without morphology ambiguity

## Immediate Next Action

Execute the first bounded step:

- M2.1 define representative ROM sample selection policy
- M2b.1 audit current code paths for any actual inverse-transform candidate

Do not start broader kernel or solver work until those two substeps are written down.
