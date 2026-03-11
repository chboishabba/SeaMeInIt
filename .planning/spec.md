# Morphology Debugging Spec

Date: 2026-03-10

## Goal

Make it possible to say, for each important pipeline stage, whether morphology is:

- base / expected,
- normal-human,
- flailing,
- ogre-like,
- or still unclassified.

This must be attributable by artifact and stage, not inferred from vague run labels.

This phase also needs to answer a deeper design question left over from earlier
project work:

- are we trying to inspect a field/operator on the fitted body,
- or a genuinely internalized deformation domain that later needs an inverse
  map back to the fitted body?

## Problem Statement

Current runs mix together several different things:

- body geometry,
- ROM sample poses,
- aggregate ROM fields on neutral geometry,
- seam solutions,
- reprojection artifacts,
- render-orientation artifacts.

That makes it too easy to mistake:

- a field render for a morphology change,
- a reprojection/render issue for an ogre-like geometry,
- or a real flailing pose output for a bug elsewhere.

## Success Criteria

1. Run pages show a stage-by-stage morphology ledger.
2. Important historical reference runs are backfilled with explicit morphology observations.
3. ROM sampling emits representative posed/deformed sample artifacts so flailing can be inspected directly.
4. Kernel comparisons can use those artifacts to judge whether a field is coherent with the observed morphology.
5. Solver work is evaluated only after morphology attribution is good enough to trust.
6. The repo states clearly whether the current "back to body" step is:
   - a true inverse transform,
   - or only an approximate correspondence/reprojection.
7. Representative ROM sample artifacts are emitted with a deterministic selection policy and explicit lineage.
8. The repo states the current approximation acceptance contract separately from any future inverse ambition.

## Non-Goals

- Declaring `ogre-like` or `flailing` desirable.
- Treating `human` / `ogre` labels as sufficient evidence of morphology.
- Using seam-solver outputs alone to judge ROM correctness.
- Pretending that today's reprojection/correspondence step is already a proven
  inverse of any ROM/internalization transform.

## Current Contract

Reference note:

- `docs/rom_sample_morphology_and_transfer_contract.md`

This phase now treats the following as fixed implementation targets:

- representative ROM sample selection is deterministic and anchored to actual sampler statistics
- representative sample artifacts stay on sampler-native topology
- run pages and operator reports surface sample morphology separately from aggregate fields
- current back-transfer language is approximate correspondence/reprojection, not inverse

## Next Priority Shift (2026-03-11)

Reference note:

- `docs/body_fit_and_inverse_roadmap_20260311.md`

The next phase is no longer only about kernel interpretation. It now also has to
answer:

- what exact forward object `B_ogre` actually was
- why the current Afflec fit is producing an egg-shaped / Green-Goblin-like skull crown
- what back-transfer contract is required if ROM-domain seam solves must land on the fitted Afflec body
- how a future video-input fitting path should be staged
- how SMPL-X fitting policy should handle feminine bodies and other body-shape regimes intentionally rather than implicitly
