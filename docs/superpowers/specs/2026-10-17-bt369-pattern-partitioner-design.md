# BT369-native pattern partition experiment

## Goal

Test whether the remaining Gate-6 failures on Afflec P0/P2 are primarily a **panelization / operator-expressivity** problem before introducing a new UV coordinate solver.

The experiment must not weaken or bypass the current Gate-6 backend competition. A BT369 candidate is only a new way to partition the existing face-backed panel; every resulting chart is still flattened and measured by the registered bootstrap/LSCM/xatlas backends and judged by the same score/distortion/foldover/exportability rules.

## Inputs

For one source panel:

- source 3D vertices and oriented triangle faces;
- the existing serializer failure field (`face_distortions`, `foldover_faces`, `high_distortion_faces`);
- the existing distortion threshold;
- deterministic BT369 partition policy parameters.

## Face-carried BT369 state

Each original face receives a certificate-bearing state:

- `face_index`;
- `residual`: measured per-face serializer distortion;
- `residual_trit ∈ {-1,0,+1}`:
  - `-1` for a measured foldover face;
  - `0` when residual is within the declared threshold;
  - `+1` for a non-foldover face above threshold;
- `orientation_bin ∈ Z/6`, derived deterministically from an oriented face edge in a geometry-attached frame whose `u` axis is the lexicographically tie-broken farthest panel chord and whose `v` axis is built from a canonically signed aggregate face normal;
- `depth`, an MDL-style bounded base-3 refinement depth from `residual / threshold`;
- `triadic_prefix`, from normalized panel-frame face-centroid coordinates;
- `area_weight`;
- measured foldover/high-distortion flags.

This reuses the certificate vocabulary of `smii.unwrap.sphere_bt369` without claiming the garment panel is spherical.

## Partition rule

Build the face adjacency graph. An adjacency is cut when one or more enabled policy predicates fire:

1. foldover isolation: exactly one adjacent face is a measured foldover;
2. residual-trit boundary: the two residual trits differ and the larger residual exceeds threshold;
3. orientation mismatch: circular `Z/6` distance exceeds the configured limit while either side is above threshold;
4. refinement mismatch: depth difference exceeds the configured limit;
5. prefix mismatch: both faces have enough refinement depth and fail the configured shared-prefix requirement.

Connected components after removing cut adjacencies are the candidate chart partition. The receipt must prove by construction/checks that every original face occurs exactly once.

## Policy sweep

The first evaluator performs a small deterministic bounded sweep rather than selecting one hand-tuned policy:

- orientation jump limit: `0, 1, 2`;
- refinement-depth jump limit: `0, 1, 2`;
- shared-prefix requirement: `0, 1`;
- foldover isolation: `true`;
- residual-trit cuts: `true`.

Equivalent topology hashes are deduplicated before backend evaluation.

Each unique partition is converted back into connected `PanelPatch` charts and sent through the existing `serialize_panel` competition. The evaluator records fresh aggregate metrics, unchanged relative-parent acceptance, strict foldover/exportability guard, and profile/Pareto diagnostics.

## Promotion boundary

`bt369_pattern_partition.v1` is a **partition certificate**, not manufacturing authority and not a proof that BT369 is physically privileged.

The evaluator may mark a candidate `promotion = 1` only when freshly measured backend results satisfy the existing parent-surgery rule:

- chart domains backend-serializable;
- aggregate score improves;
- worst distortion does not regress;
- foldovers do not regress;
- result is exportable.

No composed delta estimate is used for promotion.

## Decision after the run

- If a BT369 partition improves/promotes P0 or P2, the next cut is to integrate the successful partition policy into the materialization-aware unwrap search and test it against broader assets.
- If BT369 partitions improve geometry but all registered UV backends still form the limiting Pareto boundary, only then prototype a native BT369 coordinate solver.
- If the bounded BT369 partition family produces no useful frontier movement, diagnose missing geometric/fabric state before increasing search depth.
