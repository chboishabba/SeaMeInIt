# Ogre Artifact Diagnostics

Date: 2026-02-12

## Observed Appearance

The orbit artifacts generated during seam solver sweeps showed an "ogre-like"
silhouette with the following recurring symptoms:

- Head appears oversized and visually absorbs torso/abdomen into a large chin.
- Shoulder/scapula region appears pointy and bulky with a center-spine dip.
- Arms/hands look comparatively normal, but hotspot intensity is strongest in
  shoulders/arms/hands (especially hands).
- Front face often appears very dark while back-of-head is mid-intensity.
- Small "fin" feature appears along spine/top-of-head in some views.

These symptoms were reported primarily in `overlay_orbit.webm` outputs generated
from point-cloud style diagnostics.

## Where The Ogre Occurs In Pipeline

The distortion is currently attributed to visualization stage artifacts rather
than seam geometry deformation:

1. Body mesh remains static (no physically deformed avatar mesh is rendered).
2. ROM values are used as per-vertex color field.
3. Render path draws projected point cloud + seam edges, not a shaded surface.
4. Previous low point size amplified aliasing and silhouette ambiguity.

Additional risk identified:

- Vertex-count mismatch between body mesh and ROM cost field can produce
  misleading hotspot assignment.

Current renderer now raises on mismatch by default to prevent silent corruption.

## Why It Looked Different From Earlier Heatmaps

Earlier heatmaps/PNGs used a different plotting path and camera defaults.
Variant orbit artifacts use a custom lightweight renderer intended for rapid
 batch diagnostics. This changed:

- projection style,
- occlusion behavior,
- point blending profile,
- and file naming/versioning behavior.

So visual style changed independently of solver behavior.

## Artifact Preservation Policy

To avoid data loss from overwrites:

- New renders are emitted to canonical filenames (`overlay_orbit.webm`, etc.)
  and additionally copied to timestamped filenames:
  - `overlay_front_3q_<timestamp>.png`
  - `overlay_orbit_<timestamp>.gif`
  - `overlay_orbit_<timestamp>.webm`
- Existing canonical files are archived before overwrite at:
  - `artifact_history/<timestamp>/...`
- Summary manifests are written both as latest and timestamped:
  - `artifact_summary_all.json|csv`
  - `artifact_summary_<timestamp>.json|csv`

Renderer entrypoint:

- `scripts/render_variant_orbits.py`

## Shortest-Path Walker Notes

The shortest-path solver previously produced overly simple seam walks and could
return infinite cost when anchors were disconnected.

Recent updates:

- disconnected-anchor fallback to largest connected component,
- loop-aware waypoint routing across panel anchor loops,
- local-edge preference by filtering long jumps for seam walking,
- unresolved waypoint warnings propagated to report.

This is intended to improve seam path topology adherence without switching to a
full panel-wide MST/PDA objective.

## Next Debug Steps

1. Compare shortest-path edge sets against seam graph local adjacency per panel.
2. Add quantitative path quality metrics:
   - mean edge length,
   - long-jump ratio,
   - connected component count touched by selected path.
3. Evaluate whether waypoint count should be task/fabric dependent.
4. Add a shaded-mesh artifact mode (non-point-cloud) for qualitative review.
