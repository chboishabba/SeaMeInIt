# Devlog

## 2026-03-10

- Established a dedicated morphology-debugging phase in `.planning/`.
- Prioritized the work into four milestones:
  - M1 backfill morphology observations on reference runs,
  - M2 emit explicit ROM sample morphology artifacts,
  - M3 compare candidate ROM fields on one topology,
  - M4 return to solver sensitivity only after morphology attribution is clearer.
- Kept the immediate next action narrow: annotate existing run pages before adding new ROM outputs.
- Consulted the newest local formalism snapshot at `/home/c/Documents/code/ITIR-suite/all_code48.txt`.
- Backfilled `morphology_observations.json` into the four active reference runs so M1 can be evaluated on real pages rather than TODO text alone.
- Advanced planning state to M2 after regenerating the run pages and verifying that override-only artifacts are now included in the morphology audit.
- Recorded the original "solve on internalized morphology, then invert back to fitted body" intent as a distinct roadmap item.
- Clarified from current repo state plus `all_code44.txt` / `all_code48.txt` that this inverse is not presently implemented or justified; today's back-transfer is approximate correspondence/reprojection.
- Reframed the product milestones explicitly as: sewable bodysuit -> thermal routing/cooling -> comfortable systems packaging -> later harder "iron man" modules.
- Broke M2 into four smaller substeps (selection policy, artifact contract, minimal emission, report integration).
- Broke M2b into four smaller substeps (code-path audit, current-reality doc, approximation acceptance, future inverse requirements).
- Set the next bounded orchestration step to: define sample selection policy and audit the codebase for a real inverse candidate.
- Marked generic repo audit complete for this phase so the autonomous orchestrator does not keep routing back to `repo-auditor` instead of the active morphology tasks.
- Wrote the explicit contract for representative ROM sample selection and inverse/back-transfer semantics in `docs/rom_sample_morphology_and_transfer_contract.md`.
- Locked the current implementation target to sampler-native posed sample meshes plus a manifest, with deterministic selection anchors and no fake inverse claims.
- Implemented the baseline M2/M2b slice:
  - `sampler_real` can now emit representative ROM sample meshes and a `rom_sample_manifest.json`,
  - run pages classify those sample meshes as `rom_sample_pose`,
  - operator reports can surface the representative sample manifest,
  - tests cover selection policy, sample artifact emission, operator report wiring, and run-page morphology integration.
- Advanced the orchestrator-facing next step to M3 now that the M2/M2b slice is implemented and passing tests.
- Reprioritized the next phase around fit trustworthiness and the real solve-domain object:
  - first audit the exact historical `B_ogre` forward object,
  - then isolate the current Afflec crown/head-shape failure stage,
  - then define the Afflec-facing back-transfer requirement,
  - only after that specify video-input fitting and SMPL-X feminine-body coverage work.
- Completed the first evidence pass on those two audits:
  - historical `B_ogre` is now pinned to a real `9438` native solve object on `outputs/suits/afflec_body/base_layer.npz`,
  - the old `9438 -> 3240` control transfer is confirmed too lossy to count as an inverse/back-transfer,
  - the current Afflec crown issue is narrowed away from parameter drift and toward late mesh generation / repair / export or a latent fitted-geometry issue revealed there.
- Recorded that audit in `docs/b_ogre_and_afflec_crown_audit_20260311.md`.
- Set the next bounded execution step to: add export-stage Afflec mesh checkpoints, define the skull plausibility gate, and then write the Afflec-facing back-transfer spec against the actual `9438` forward object.
