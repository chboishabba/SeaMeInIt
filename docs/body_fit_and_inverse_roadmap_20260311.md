## Body Fit And Inverse Roadmap

Date: 2026-03-11

### Why This Note Exists

Three threads are now coupled enough that they need one prioritized roadmap:

1. current Afflec fit quality is not yet trustworthy enough for production seam work
2. the historical `B_ogre` branch is still not well-defined as geometry vs render/domain artifact
3. if ROM-domain seam solves are meant to land back on the fitted Afflec body, the repo needs a real back-transfer story

This note prioritizes those threads and breaks them into bounded subtasks.

### Priority Order

#### P1. Audit the exact `B_ogre` forward object

This is the highest priority because it determines whether there is even a real
geometry object to invert or transfer from.

Questions to answer:

- which exact mesh artifact was the source geometry behind historical `B_ogre`
- whether `B_ogre` was a real transformed mesh, or mainly a render/domain artifact
- which topology, hashes, and lineage belong to that forward object

Bounded subtasks:

1. `P1.1` collect the authoritative historical files
   - seam reports
   - render summaries
   - mesh paths
   - decision metrics
2. `P1.2` identify the exact mesh path/hash/topology behind `B_ogre`
3. `P1.3` determine whether the visible ogre-like appearance came from:
   - mesh geometry
   - render-axis/camera behavior
   - reprojection/transfer collapse
   - mixed domain confusion
4. `P1.4` write the forward-object decision note

Done when:

- the repo can point to one exact forward object for `B_ogre`, or explicitly state that no such stable geometry artifact exists

#### P2. Diagnose current Afflec crown/head-shape failure

The current Afflec body is showing an egg-shaped / Green-Goblin-like crown.
This must be isolated before the Afflec body is treated as a trustworthy solve target.

Working hypotheses:

- sparse front-angle imagery is underconstrained for head shape
- current reprojection fit is overfitting frontal observations
- measurement refinement or mesh repair is distorting the skull after fitting

Bounded subtasks:

1. `P2.1` reproduce the crown issue on the latest current Afflec run
2. `P2.2` compare morphology at each stage:
   - raw reprojection fit
   - reprojection + measurement refinement
   - repaired/exported body
3. `P2.3` decide whether the distortion is introduced by:
   - observation scarcity
   - optimization/priors
   - refinement handoff
   - repair/export
4. `P2.4` define an acceptance gate for head/skull plausibility

Done when:

- we know which stage introduces the crown distortion and have a specific corrective target

#### P3. Define a proper Afflec-facing back-transfer requirement

If seams are solved in a ROM/native/internalized domain, they must come back to
the fitted Afflec mesh in a way that preserves the intended design meaning.

This does not yet assume a true inverse exists. It defines what is required.

Bounded subtasks:

1. `P3.1` define the forward object from P1
2. `P3.2` decide whether the return path must be:
   - true inverse
   - approximate transfer with strict guarantees
3. `P3.3` define required quality criteria:
   - topology lineage
   - round-trip checks where possible
   - seam retention/collision limits
   - morphology preservation expectations
4. `P3.4` define the implementation track around that exact object, not generic NN reprojection

Done when:

- the repo has an explicit Afflec-facing back-transfer spec tied to the actual forward object

#### P4. Video-input body fitting path

Video input is likely required to reduce underconstrained head/body fits and
improve side/rear coverage.

Bounded subtasks:

1. `P4.1` specify accepted video workflow
   - single clip
   - multi-view clips
   - extracted frame policy
2. `P4.2` define frame selection and de-duplication rules
3. `P4.3` define observation aggregation across frames/views
4. `P4.4` add provenance contract for video-derived fits
5. `P4.5` implement CLI/path only after the contract is agreed

Done when:

- there is a documented video-ingest fitting contract and a bounded implementation path

#### P5. SMPL-X body-shape coverage audit, especially feminine bodies

This is important, but it comes after P1-P3 because the current ambiguity about
the solve domain and Afflec body trustworthiness is even more blocking.

Questions to answer:

- what gender/body-shape priors are active now
- how neutral/female/male models are chosen
- how fitting behaves on bodies that are not well described by the current defaults

Bounded subtasks:

1. `P5.1` audit current SMPL-X provider/config behavior
2. `P5.2` document current limitations and assumptions
3. `P5.3` define fit-policy expectations for feminine bodies
4. `P5.4` add fixtures/tests later, after the audit is written

Done when:

- the repo has an explicit fit policy instead of implicit assumptions about body type coverage

### Recommended Execution Sequence

1. `P1` audit `B_ogre`
2. `P2` isolate Afflec crown distortion
3. `P3` write the exact back-transfer requirement
4. `P4` specify video-input fitting
5. `P5` audit SMPL-X feminine-body coverage

### Non-Goals For This Slice

- implementing video ingestion immediately
- claiming a true inverse before the forward object is pinned down
- treating current Afflec or `B_ogre` outputs as production-valid by default

### Audit Update

First audit pass completed in:

- `docs/b_ogre_and_afflec_crown_audit_20260311.md`

Current state after that audit:

- `P1` is materially narrowed:
  - there is a real historical `B_ogre` forward solve object on
    `outputs/suits/afflec_body/base_layer.npz` (`9438` vertices),
  - but the old visually ogre-like appearances still cannot yet be attributed
    cleanly to geometry alone rather than geometry plus transfer/render effects.
- `P2` is materially narrowed:
  - the current Afflec crown pathology does not appear to come from parameter
    drift between the calibrated `complete2` and `complete3` runs,
  - it is now most plausibly a late body-generation/export problem or a latent
    fitted-geometry problem only visible at export.

So the next bounded tasks are:

1. add export-stage Afflec body checkpoints
2. define the skull/head plausibility gate
3. write the back-transfer spec against the actual `9438` forward object
