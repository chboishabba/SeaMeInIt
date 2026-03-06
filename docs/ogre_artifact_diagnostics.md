# Ogre Artifact Diagnostics

Date: 2026-02-12

For baseline mesh lineage (what `afflec_body` and `afflec_canonical_basis`
actually represent), see `docs/mesh_provenance_afflec.md`.

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

Orientation note (2026-02-13):
- User corrected earlier \"face-down\" language; ogre is face-up.
- Some apparent \"face-down vs face-forward\" confusion is consistent with camera yaw choices
  and/or axis canonicalization differences between runs. New render manifests record the
  axis convention (`render_axis.axis_order`) for each artifact so this can be debugged
  from data, not memory.

## Where The Ogre Occurs In Pipeline

Current understanding is mixed and remains under investigation. Two factors can
coexist:

1. ROM/topology transformation effects (domain-level morphology changes).
2. Rendering/diagnostic projection effects (point-cloud aliasing/occlusion).

Prior notes leaned heavily on visualization artifacts; current position is that
this is insufficient by itself for all observed runs.

Additional confirmed contributor (render-axis instability):

- `scripts/render_variant_orbits.py` historically inferred the “up axis” by
  choosing the coordinate axis with the largest span. On T-pose meshes, arm span
  can exceed height, causing a 90-degree axis swap and a consistent “ogre-like”
  silhouette even when the mesh itself is normal. This is a renderer issue, not
  a ROM deformation. Future renders must record or fix the axis convention.

Current known mechanics:

1. Some seam runs are solved on a high-topology branch (`9438`).
2. Base Afflec demo body is a different topology (`3240`).
3. Render path draws projected point cloud + seam edges (not shaded surface).
4. Reprojection between topologies can collapse edges when nearest-neighbor
   distances are large.

Clarification: the seam solver does not operate on a point cloud.
- Solvers consume `vertices` **and** `faces` to build a panel seam graph (mesh-adjacent edges).
- The orbit artifacts are point-cloud renders for debugging; they can make seams
  look visually disconnected even when the underlying edge list is valid.

Additional risk identified:

- Vertex-count mismatch between body mesh and ROM cost field can produce
  misleading hotspot assignment.

Current renderer now raises on mismatch by default to prevent silent corruption.

## Confirmed Root Cause (2026-02-12)

For `shortest_path_knit_4way_light_grain100` we validated the selected seam
edges against true mesh 1-ring connectivity:

- Before fix: `68` unique seam edges, `51` were **not mesh edges**.
- Symptom: long interior spikes and "spiderweb" seam overlays.
- Interpretation: seam solver was traversing ROM edge payload links that are
  not manifold-adjacent in the body mesh.

Fix implemented:

- `build_edge_kernels` now gates candidate edges by panel triangle connectivity
  (mesh-adjacent seam edges only) before solver use.

After fix:

- `shortest_path_knit_4way_light_grain100`: `12` unique seam edges.
- Non-mesh edges: `0`.
- Segment lengths now in local mesh scale (max well below previous spikes).

## Known Failure Modes

This section tracks known failure modes with detection and current status.

### 1) Non-manifold seam edge leakage (index-graph links)

- Symptom:
  - long interior spikes / "spiderweb" seam overlays
  - seams visually cross through body volume
- Detection:
  - compare solver seam edges against mesh 1-ring edge set
  - failure when `non_mesh_edge_count > 0`
- Root cause:
  - solver candidate graph consumed ROM payload edges that were not guaranteed
    to be manifold-adjacent mesh edges
- Status:
  - mitigated
- Mitigation:
  - kernel builder now gates edges by panel triangle connectivity

### 2) Seam appears "missing" in orbit artifacts

- Symptom:
  - seam not visible in mixed point-cloud + seam render
- Detection:
  - seam exists in report edge list but is hard to perceive in overlay
- Root cause:
  - low contrast and clutter from full-body point cloud
- Status:
  - mitigated for debugging
- Mitigation:
  - renderer supports seam-focused views:
    - `--seam-only`
    - `--seam-width`
    - `--body-alpha`

### 3) Selected-run render clobbers global summary manifests

- Symptom:
  - running renderer with `--run ...` overwrote `artifact_summary_all.*`
- Detection:
  - summary manifest unexpectedly contains only selected subset
- Root cause:
  - single-run and full-run flows shared the same summary target filenames
- Status:
  - mitigated
- Mitigation:
  - selected-run mode now writes:
    - `artifact_summary_selected_<timestamp>.json|csv`

### 4) Thumb-local or tiny seam collapse in shortest-path output

- Symptom:
  - seam collapses to very small local path (e.g., thumb-tip looking segment)
- Detection:
  - very low edge count and tiny seam bounding box in report diagnostics
- Root cause:
  - sparse/biased candidate edge graph and anchor selection behavior
- Status:
  - partially mitigated
- Mitigation:
  - shortest-path walker now uses stronger locality checks and trunk-first mode
  - mesh-adjacent kernel gating expanded candidate set over panel mesh edges
- Remaining risk:
  - still sensitive to panel seam set quality and anchor validity

### 5) Panel seam-set corruption / over-broad seam vertex sets

- Symptom:
  - several panels report seam vertex sets covering almost/all body vertices
  - left/right panels become near-identical for some loop bands
- Detection:
  - `len(panel.seam_vertices)` unexpectedly near global vertex count
  - left/right Jaccard overlap near `1.0`
- Root cause:
  - upstream panelization/seam selection logic issue (under investigation)
- Status:
  - open
- Current mitigation:
  - downstream shortest-path and kernel graph now enforce mesh-local adjacency
    so this does not explode into non-manifold seam links

### 6) Body/ROM vertex-count mismatch during rendering

- Symptom:
  - misleading hotspot placement; distorted "ogre" heat interpretation
- Detection:
  - vertex count mismatch between body mesh and ROM cost field arrays
- Root cause:
  - incompatible artifacts used together in renderer
- Status:
  - mitigated
- Mitigation:
  - renderer now raises on mismatch instead of silently padding/truncating

### 7) Seam report/body topology mismatch (out-of-range seam indices)

- Symptom:
  - seam appears partial/missing (only small fragment visible), despite edges in report
- Detection:
  - seam edge indices exceed body vertex count for the render body
- Root cause:
  - seam report generated on a different topology than the render body mesh
- Status:
  - mitigated
- Mitigation:
  - `scripts/render_variant_orbits.py` now fails fast when seam edge indices are
    out of range for the selected body mesh
  - `examples/solve_seams_from_rom.py` now fails early when ROM cost field
    length does not match body vertex count

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
- Renderer now also emits descriptive alias filenames with run/body/cost/render
  settings embedded (for example:
  `orbit__run-<run>__body-<body>__cost-<cost>__...__ts-<timestamp>.webm`).
- Existing canonical files are archived before overwrite at:
  - `artifact_history/<timestamp>/...`
- Summary manifests are written both as latest and timestamped:
  - `artifact_summary_all.json|csv`
  - `artifact_summary_<timestamp>.json|csv`

Renderer entrypoint:

- `scripts/render_variant_orbits.py`

## Required correspondence visualization (vertex-map orbit)

When making claims like “9438→3240 map has high collision / low retention”, the
protocol requires a dedicated orbit that renders:

- target mesh point cloud (colored by map distance),
- optional overlay of the source mesh (neutral gray),
- and an optional subsampled set of correspondence lines (target vertex ->
  mapped source vertex) so gross mismatch is visible.

This is intentionally separate from seam overlays so correspondence quality can
be assessed without seam-solver confounds.

## 2026-02-13 Visual Cross-Check (User)

Run family: `outputs/seams_run/domain_ab_20260213_101158/`

User-observed:
- `B_ogre_to_base_control` looks ogre-like and shows a single scapula-adjacent
  line that does not conform to mesh semantics.
- `B_ogre` looks normal-body-like with mouth/groin seam fragments.
- `A_base` looks ogre-like with a non-loop seam trend (butt-to-head style),
  while still appearing mesh-conforming.

Alignment with diagnostics:
- Non-conforming transferred seam is consistent with reprojection metrics
  (`mesh_edge_valid_ratio=0.0`, high collision ratio).
- Native solves can still be mesh-conforming while morphology differs.
- Run labels (`A_base`, `B_ogre`) should not be treated as guaranteed morphology
  labels; lineage/topology metadata is authoritative.

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

## Solver Behavior Contract (Frozen Baseline + Optional Controls)

To keep runs reproducible while continuing seam semantics work, shortest-path is
treated as a frozen baseline with explicit toggles:

- Baseline (`shortest_path` default):
  - open-path trunk solve between panel anchors;
  - mesh-adjacent edges only;
  - local-edge filtering enabled.
- Optional loop mode (`require_loop=true`):
  - attempt to build a loop-like seam via two distinct anchor-to-anchor paths;
  - if no alternate return path exists, emit an explicit warning.
- Optional symmetry penalty (`symmetry_penalty_weight > 0`):
  - adds mirrored-edge mismatch cost using `ConstraintRegistry` symmetry pairs.
- Optional strict locality (`allow_unfiltered_fallback=false`):
  - disables local-graph disconnection fallback to unfiltered edges.
- Optional reference-mesh diagnostics (`reference_vertices`):
  - computes seam length metrics on a non-warped origin mesh when topology
    indices are unchanged.

These controls are intended as modeling flags, not hardcoded anatomical rules.

## Next Debug Steps

1. Compare shortest-path edge sets against seam graph local adjacency per panel.
2. Add quantitative path quality metrics:
   - mean edge length,
   - long-jump ratio,
   - connected component count touched by selected path.
3. Evaluate whether waypoint count should be task/fabric dependent.
4. Add a shaded-mesh artifact mode (non-point-cloud) for qualitative review.

## Explicit Ogre->Afflec Reprojection

When seams are produced on a different topology and must be inspected on the
afflec base body, use the dedicated reprojection utility:

```bash
PYTHONPATH=src /Whisper-WebUI/venv/bin/python scripts/reproject_seam_report.py \
  --report outputs/seams_run/afflec_real_sp_edges/seam_report.json \
  --source-mesh <mesh used by that seam report> \
  --target-mesh outputs/afflec_demo/afflec_body.npz \
  --vertex-map-file <source_to_target_map.npz> \
  --out outputs/seams_run/reprojected_afflec_real_sp_to_afflec_body/seam_report.json \
  --strict-quality \
  --max-mean-distance 0.05 \
  --max-distance 0.15 \
  --min-edge-retention 0.5 \
  --max-target-collision-ratio 0.2
```

Important:

- Reprojection requires the actual source mesh used by the seam run.
- If nearest-neighbour distances are large and most edges collapse, the source
  mesh is likely wrong or unavailable; treat as invalid transfer.
- New seam reports now include a `provenance` block with body/cost paths and
  vertex counts to reduce this ambiguity.
- `scripts/reproject_seam_report.py` now records `edge_retention_ratio`,
  `unique_target_vertices`, and `target_vertex_collision_ratio` plus quality
  violations. With `--strict-quality`, it exits non-zero on poor transfer quality.
- In this repository, legacy realshape runs are typically `9438`-vertex while
  current `outputs/afflec_demo/afflec_body.npz` is `3240`; this mismatch is
  expected across branches and must be handled explicitly.

## Run Note: 2026-02-13 Variant Matrix `20260213_035227`

Observed from generated artifacts:

- `shortest_path_knit_4way_light_grain100_ogre` rendered as a normal-looking
  body silhouette (not the expected Afflec-shaped output); seam fragments appear
  around mouth/jaw and groin/perineum ("gooch") regions.
- `shortest_path_knit_4way_light_grain100_ogre_to_afflec_body` rendered with an
  ogre-like cloud silhouette and only a single short seam segment near scapula.
- The two seam sets are not meaningfully correlated visually.

Quantitative diagnostics for this run:

- Source seam report topology (`9438` mesh): seam edge indices up to `9432`.
- Reprojected seam report topology (`3240` mesh): edge retention collapsed from
  `38 -> 2` directed panel edges (`17 -> 1` unique undirected edges).
- Reprojection distances were large (`mean_distance=0.3475 m`,
  `max_distance=0.5329 m`), indicating invalid/very weak geometric alignment.

Interpretation:

- This reprojection result should be treated as a topology-transfer failure,
  not a faithful projection of seam semantics onto the base body.
