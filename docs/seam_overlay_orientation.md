# Seam Overlay Orientation

This document defines how seam overlay camera views are generated in
`smii.seams.diagnostics.render_overlay`.

## Canonical Frame

Overlays are rendered in a canonical body frame `(x, y, z)` where:

- `x = width` (largest non-vertical span)
- `y = depth` (remaining non-vertical axis)
- `z = vertical` (largest global mesh span)

Axis selection is inferred from mesh extents. A vertical sign heuristic is then
applied so positive `z` points toward the narrower body end (typically head).

The exact inferred mapping for each run is written to:

- `overlay_views.json`

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
