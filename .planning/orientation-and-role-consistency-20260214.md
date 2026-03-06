# Orientation and Role Consistency Plan (2026-02-14)

## Phase

Seam pipeline stabilization: orientation invariants + role labeling.

## Objective

Stop “human vs ogre” ambiguity and stop orientation drift between domains so that
any visual comparisons are meaningful.

## Scope

- In scope:
  - define and enforce a shared render orientation policy across domains,
  - ensure artifact filenames encode expected role (`human|ogre`) + topology (`vNNNN`),
  - run the multi-loop solver on the `human` (`v3240`) domain with a stable render.
- Out of scope:
  - changing seam objective terms,
  - adding new mapping/reprojection algorithms (barycentric transfer, ICP),
  - re-fitting SMPL-X or regenerating body meshes.

## Constraints

- Ogre and human renders must always face the same direction in the same bundle.
- No “guessing” roles from filenames alone; roles must be explicit in filenames/manifests.
- Documentation must stay ahead of code changes (docs -> TODO -> code -> changelog).

## Success Criteria

1. Strategy 2 bundles default to a shared orientation policy and record it in manifests.
2. Bundle artifacts include role+topology tags in stems (e.g., `human_v3240__...`).
3. A multi-loop shortest-path run on `human_v3240` has an orbit render that is not face-up.
4. Docs include a single “what is human/what is ogre” definition and the observed mismatches list.

## Open Questions

- Should render “up axis” be fixed (`y`) or inferred deterministically per mesh?
  Current hypothesis: some meshes encode height in `z`, others in `y`; we need an
  `axis_up=auto` policy that is stable on T-poses.

## Next Actions

1. Document the orientation invariant and observed failures.
2. Add a TODO + implement `axis_up=auto` canonicalization option and use it by default in bundles.
3. Re-run a small Strategy 2 smoke bundle to confirm both domains face the same direction.

