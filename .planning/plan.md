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
  - define deterministic anchors and fill policy
- M2.2 define the artifact contract
  - emit sampler-native posed meshes plus manifest metadata
  - decide where lineage and morphology notes live
- M2.3 implement minimal sample artifact emission
  - emit at least one representative posed/deformed sample set into a run root
- M2.4 integrate those artifacts into run pages and operator reports
  - ensure the viewer surfaces show neutral body, samples, aggregate field, and seams separately

Done when:
- a run page can show neutral body, sample poses, aggregate field, and seam outputs separately

Status:
- baseline complete on 2026-03-10:
  - deterministic representative-sample policy documented,
  - sampler can emit `rom_samples/rom_sample_manifest.json` plus sampler-native posed sample meshes,
  - run pages and operator reports surface those artifacts.

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

Status:
- complete on 2026-03-10:
  - current code-path audit found no true geometry inverse,
  - repo contract now states that current back-transfer is correspondence/reprojection,
  - acceptance criteria for approximate transfer and requirements for a future inverse are documented.

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

### M5. `B_ogre` forward-object audit

Goal:
- identify the exact historical object behind `B_ogre`

Tasks:
- `M5.1` collect the historical seam/render/mesh artifacts used for `B_ogre`
- `M5.2` identify the exact mesh artifact, topology, and hashes behind `B_ogre`
- `M5.3` decide whether `B_ogre` was real transformed geometry or a render/domain artifact
- `M5.4` record the forward-object decision note

Done when:
- the repo can name the forward object precisely or state that it never had a stable geometry object

Status:
- materially complete on 2026-03-11:
  - historical `B_ogre` was narrowed to a real `9438`-vertex native solve object on `outputs/suits/afflec_body/base_layer.npz`,
  - old `9438 -> 3240` control transfer is confirmed unusable as an inverse/back-transfer path,
  - remaining ambiguity is visual attribution: real geometry vs geometry-plus-render/transfer artifact.

### M6. Afflec crown-fit diagnosis

Goal:
- isolate why the current Afflec fit is producing an egg-shaped / Green-Goblin-like skull crown

Tasks:
- `M6.1` reproduce the issue on the latest current Afflec run
- `M6.2` compare raw reprojection, refined fit, and repaired/exported mesh
- `M6.3` identify which stage introduces the crown distortion
- `M6.4` define a head/skull plausibility gate

Done when:
- one concrete stage is blamed and a corrective implementation target is clear

Status:
- partially complete on 2026-03-11:
  - the issue is reproduced and narrowed away from parameter drift between the calibrated `complete2` and `complete3` runs,
  - likely loci are now late mesh generation / repair / export, or a latent fitted-geometry issue only visible there,
  - missing piece is export-stage mesh checkpointing plus a skull plausibility gate.

### M7. Afflec-facing back-transfer contract

Goal:
- define the required return path from ROM/native solve domain back to the fitted Afflec body

Tasks:
- `M7.1` tie the contract to the exact forward object from M5
- `M7.2` decide true inverse vs strict approximation requirement
- `M7.3` define quality/round-trip/morphology-preservation criteria
- `M7.4` define the implementation track around that exact object

Done when:
- there is an explicit back-transfer spec tied to the real forward object

### M8. Video-input fitting path

Goal:
- specify how video should enter the fitting pipeline to reduce underconstrained shape failures

Tasks:
- `M8.1` define accepted video/frame workflows
- `M8.2` define frame selection and aggregation
- `M8.3` define provenance requirements
- `M8.4` implement later after the contract is settled

Done when:
- the repo has a documented bounded video-fit implementation path

### M9. SMPL-X body-shape coverage audit

Goal:
- document how fitting should behave across body-shape regimes, especially feminine bodies

Tasks:
- `M9.1` audit current provider/config behavior
- `M9.2` document limitations and assumptions
- `M9.3` define intended fit policy for feminine bodies
- `M9.4` add fixtures/tests later

Done when:
- there is an explicit fit policy instead of implicit assumptions

## Immediate Next Action

Execute the next bounded slice:

- finish M6 with export-stage checkpoints and a skull plausibility gate
- start M7 by writing the back-transfer spec against the actual `9438` forward object

Do not jump to video ingestion or inverse implementation before those two audits are written down.
