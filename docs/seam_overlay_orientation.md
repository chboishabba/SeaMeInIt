# Seam Overlay Orientation

This document defines how seam overlay camera views are generated in
`smii.seams.diagnostics.render_overlay`.

## Canonical Frame

Overlays and orbit renders are rendered in a canonical body frame `(x, y, z)` where:

- `x = width` (arm-span axis)
- `y = depth` (front/back axis)
- `z = up` (head/feet axis)

### PCA-based axis selection (`--axis-up auto`)

Renders must be comparable across domains even when meshes arrive in different
world-coordinate conventions (e.g. one mesh has height in `y`, another in `z`,
or an upstream step rotated the body by 90 degrees).

For `--canonicalize --axis-up auto`, the renderer computes a deterministic
canonical frame using **PCA + robust tail statistics**:

1. Center vertices by subtracting the vertex mean.
2. Compute PCA eigenvectors (principal axes) and project points onto each axis.
3. Compute per-axis robust stats:
   - `qspan = (p99 - p1)`
   - `iqr = (p75 - p25)`
   - `tail_ratio = qspan / iqr`
4. Choose `width` axis as the PCA axis with the largest `tail_ratio` (hands tend
   to create heavy tails).
5. Choose `up` axis using a scored search over axis assignments:
   - evaluate permutations of `(up,width,depth)` among the 3 PCA axes,
   - score favors: large `tail_ratio(width)`, large `qspan(up)`, and a
     "feet-wider-than-head" signal (bottom slice has larger width spread than
     top slice when slicing along the candidate `up` axis),
   - select the best-scoring assignment.
6. `depth` is the remaining axis.

### Sign conventions

PCA axes are only defined up to sign. To keep renders upright and comparable:

- `up` sign is chosen so the **feet end** lies toward negative `z`:
  the bottom slice (low `z`) should have a larger `width` spread than the top
  slice (high `z`) due to two separated feet vs a single head.
- `depth` and `width` signs remain ambiguous for a perfectly symmetric T-pose,
  so **front/back and left/right may flip** between unrelated meshes. This is
  acceptable for seam placement debugging; use explicit rotations if needed for
  human interpretation.

The exact inferred mapping and statistics for each run are written to:

- `overlay_views.json`
- render manifests (for orbit tools)

in the same output directory as seam overlays.

## Camera Views

View presets are fixed and deterministic:

- `front`: `elev=0`, `azim=-90`
- `back`: `elev=0`, `azim=90`
- `front_3q`: `elev=0`, `azim=-120`
- `isometric`: `elev=18`, `azim=-60`

`front_3q` is intentionally a pure yaw from `front` (no pitch), so it should
rotate around the vertical axis without introducing "under" tilt.

## Practical Notes

- If `front`/`back` appear swapped for a specific dataset, that indicates a
  depth-sign ambiguity in the source mesh frame. The run remains internally
  consistent because `back` is always 180 degrees from `front`.
- Use `overlay_views.json` alongside the PNGs whenever reviewing seam placement
  to avoid ambiguity about orientation assumptions.
