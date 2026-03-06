# Seam Pipeline: Intended vs Observed (Afflec/Ogre)

Date: 2026-02-13

This note is the explicit alignment document for current seam debugging. It
captures:

- intended pipeline stages and rationale,
- user-observed behavior,
- agent-observed behavior,
- open decisions and next checkpoints.

Use this as the current source of truth when evaluating ogre-related artifacts.

## Scope

This document is about seam path behavior across topology branches, not about
fabric design quality or final garment aesthetics.

## Terms

- `afflec_fixture_fit_body` (working label):
  a fitted body mesh emitted by `smii.app afflec-demo`. Historically this has
  lived at `outputs/afflec_demo/afflec_body.npz` (`3240` verts), but that path is
  not stable provenance (it can be overwritten).
- `suit_base_layer` (derived mesh):
  a watertight “undersuit layer” mesh emitted by `smii.pipelines.generate_undersuit`.
  A historical artifact exists at `outputs/suits/afflec_body/base_layer.npz`
  (`9438` verts). This is not itself proof of image-ingest topology.
- `afflec_canonical_basis`:
  ROM basis artifact at `outputs/rom/afflec_canonical_basis.npz` (`9438` verts
  in the current file), not a direct proof of image-ingest topology by itself.
- `reprojection`:
  seam index transfer from one topology to another via nearest-neighbor mapping.

## Artifact naming policy (required going forward)

To stop repeated morphology/role confusion, **all new artifacts** (reports, orbits,
maps, overlays) must encode:

- `role`: `human` or `ogre`
- `topology`: `v3240`, `v9438`, etc.
- `domain`: `native` or `reprojected_from_<other>`

Examples:
- `human_v3240__mesh_only__orbit__<ts>.webm`
- `ogre_v9438__native_seams__orbit__<ts>.webm`
- `human_v3240__reprojected_seams_from_ogre_v9438__orbit__<ts>.webm`

Role naming note:
- `human`/`ogre` are **provenance labels**, not geometry classifiers.
- Do not infer roles from vertex count or morphology. Require explicit role
  assignment (CLI flags) and record mesh hashes in manifests.
  - For Strategy 2 bundles: `scripts/protocol_strategy2_bundle.py` requires `--base-role` and `--rom-role`.

## Orientation invariant (required going forward)

Within a single comparison bundle, `human` and `ogre` renders must be aligned to
the same facing direction and "up" convention. If one domain is face-up while
the other is face-forward, the comparison is invalid.

Operationally:
- Bundles must render both domains using the same canonicalization policy.
- Any per-domain rotation overrides must be explicit and recorded in manifests.

### Three meanings of "canonical" (keep separated)

- **ROM canonical**:
  the ROM basis / neutral-pose construction. This lives in ROM/model space and
  is about deformation modeling.
- **Domain canonical**:
  which topology / mesh stage is treated as primary for seam solve + downstream
  pattern tools (v3240 vs v9438); this is a workflow decision (Strategy A vs B).
- **Render canonical**:
  rotation normalization for viewable artifacts so different domains can be
  compared visually. This must not be used to infer identity or provenance.

Current policy (2026-02-14):
- `scripts/render_seam_orbit.py` and `scripts/render_vertex_map_orbits.py` default to
  `--canonicalize --axis-up auto`, which uses a PCA-based frame plus robust tail
  statistics to pick width/depth/up (see `docs/seam_overlay_orientation.md`).
- Vertex-map orbits additionally align the source canonical frame into the target
  canonical frame so correspondence lines are not dominated by arbitrary axis
  conventions.

Important naming caveat:
- Run labels like `A_base` and `B_ogre` are workflow labels, not morphology
  guarantees. Current visual observations indicate morphology can appear inverted
  relative to those names in some runs.

## Intended Pipeline (Current Understanding)

1. Image ingest -> fitted body mesh
   - Intent: produce a deterministic base body from Afflec images.
   - Command family: `python -m smii.app afflec-demo`.
   - Logical purpose: establish canonical body for this run.

2. ROM sampling on selected body
   - Intent: produce per-vertex seam costs under pose deformation schedule.
   - Command family: `python -m smii.rom.sampler_real`.
   - Logical purpose: convert motion sensitivity into seam objective weights.

3. Seam solve on the same topology as ROM costs
   - Intent: generate a seam path in the cost field's native mesh domain.
   - Command family: `python examples/solve_seams_from_rom.py`.
   - Logical purpose: avoid index/adjacency mismatch during optimization.

4. Optional transfer (reprojection) to another topology
   - Intent: visualize/compare seams on a different reference body.
   - Command family: `python scripts/reproject_seam_report.py`.
   - Logical purpose: interpretation aid only; not equivalent to solving natively
     on the target mesh.

5. Rendering and diagnostics
   - Intent: inspect seam placement against body + ROM field.
   - Command family: `python scripts/render_variant_orbits.py`.
   - Logical purpose: qualitative verification and artifact persistence.

## What Controls Morphology In Artifacts

Code-level ownership:

- Body shape in overlays comes from the `--body` mesh passed to renderer
  (`scripts/render_variant_orbits.py`), not from seam solver output.
- The seam solver optimizes edges on the provided body graph; it does not deform
  body geometry.
- ROM costs affect coloring/intensity only; they do not change mesh vertices.
- `scripts/render_variant_orbits.py` historically applied `_canonicalize_vertices(...)`
  which inferred “up axis” by max coordinate span. On T-pose meshes, arm span can
  exceed height, so this can rotate the body and create an “ogre-like” silhouette
  even when the underlying mesh is normal. Protocol moving forward: renderer must
  use an explicit axis convention (or record the chosen convention in its manifest)
  to make morphology comparisons meaningful.

Practical implication:

- If visual morphology appears inverted vs run labels (`A_base`, `B_ogre`), the
  authoritative source of truth is:
  1) render command `--body`,
  2) seam report provenance (`body_path`, vertex counts),
  3) topology counts/hashes.

## Observed Behavior (User)

For run family `outputs/seams_run/variant_matrix_20260213_035227/`:

- `.../shortest_path_knit_4way_light_grain100_ogre` appeared as a normal/base
  body with a gut; seam fragments appear at mouth/jaw and groin/perineum.
- `.../shortest_path_knit_4way_light_grain100_ogre_to_afflec_body` appeared
  ogre-like; seam appears as a single short line near scapula and does not
  conform to expected mesh semantics.
- The two seam sets appear uncorrelated.

User position:

- Ogre-mode morphology is currently attributed to ROM-based transformation of a
  normal-looking base mesh.
- Large divergence between ogre and afflec-body views is expected from the
  transformation itself.
- It remains unclear why topology/vertex counts differ across branches.

For run family `outputs/seams_run/domain_ab_20260213_101158/`:

- `B_ogre_to_base_control` appears ogre-like and the seam is a single short line
  between scapula regions; seam does not conform to the target mesh.
- `B_ogre` appears normal-body-like with seam fragments around mouth and groin.
- `A_base` appears ogre-like with older non-loop-looking seam trend
  (butt-to-head style), while still appearing mesh-conforming.

User interpretation:

- Current labels (`A_base`, `B_ogre`) do not match visual morphology expectations.
- Pathing/non-conformance remains the primary issue on transferred seams.

## Observed Behavior (Agent, same run)

- Ogre solve report ran on `9438`-vertex mesh with seam indices up to `9432`.
- Reprojected report on `3240` retained very few edges (`38 -> 2` directed panel
  edges; `17 -> 1` unique undirected).
- Reprojection distances were large (`mean_distance=0.3475m`,
  `max_distance=0.5329m`), indicating a weak transfer.

Interpretation:

- This reprojection output is not reliable for semantic seam comparison.
- The observed non-correlation is expected under a poor transfer map.

## Ogre Morphology Log

Documented morphology (aggregate of prior notes and current run):

- oversized head/chin-like silhouette,
- shoulder/scapula bulk and center-spine dip/fin-like protrusion,
- hot regions concentrated around shoulders/arms/hands in some runs,
- front/back asymmetry in shading intensity,
- possible collapse/compression of torso proportions in point-cloud renders.

See also: `docs/ogre_artifact_diagnostics.md`.

## Critical Open Decision: Solve Domain

Two legitimate strategies remain open.

### Strategy A: Solve on Afflec base first, project to ROM for analysis

- Pros:
  - seam stays on canonical body topology used for downstream pattern tooling.
  - easier interpretation in human/body-reference space.
- Risks:
  - may underrepresent ROM deformation geometry where seam stress emerges.

### Strategy B: Solve on ROM/ogre domain first, reproject to base

- Pros:
  - seam optimization is native to deformation/cost field domain.
  - can capture ROM-induced corridors directly.
- Risks:
  - reprojection can become lossy/non-semantic when geometry mismatch is high.
  - poor transfer can make results appear contradictory.

Current status: unresolved. Both strategies must be benchmarked under explicit
quality gates before selecting a canonical pipeline.

## Why Vertex Counts May Differ (Working Hypothesis)

- Different artifact families were generated at different times with different
  upstream mesh/topology states (base/demo vs legacy realshape/suit branch).
- Some runs are remapped from SMPL template counts to body counts.
- Branch-local artifacts can share similar names (`afflec_body`) while referring
  to different topology lineages.

This is expected historically, but must be explicit per run.

## Current Position (Checkpoint)

- Documentation-first stabilization is in effect.
- We are treating reprojection as diagnostic, not equivalence.
- We require provenance + topology quality checks before interpreting cross-domain seams.
- We have not yet selected Strategy A vs B as the canonical production flow.
- Render artifacts now include per-run input sidecars:
  - `render_input_manifest.json`
  - `render_input_manifest_<timestamp>.json`
  These capture render body path/hash/stats, cost field path/hash/stats, seam
  report provenance, and render parameters.

## A-vs-B Protocol (Runnable)

This protocol is the required checkpoint before freezing canonical policy.

### 1) Prepare timestamped run root

```bash
export TS=$(date -u +%Y%m%d_%H%M%S)
export ROOT=outputs/seams_run/domain_ab_${TS}
mkdir -p "$ROOT"
```

### 2) Strategy A run (solve directly on base Afflec topology)

```bash
PYTHONPATH=src /Whisper-WebUI/venv/bin/python examples/solve_seams_from_rom.py \
  --body outputs/afflec_demo/afflec_body.npz \
  --rom-costs outputs/rom/seam_costs_afflec.npz \
  --solver shortest_path \
  --weights configs/kernel_weights.yaml \
  --mdl configs/mdl_prior.yaml \
  --fabric-dir configs/fabrics \
  --fabric-id knit_4way_light \
  --sp-allow-unfiltered-fallback \
  --sp-max-edge-length-factor 1.8 \
  --out "$ROOT/A_base"

PYTHONPATH=src /Whisper-WebUI/venv/bin/python scripts/render_variant_orbits.py \
  --run-root "$ROOT" \
  --run A_base \
  --body outputs/afflec_demo/afflec_body.npz \
  --cost-default outputs/rom/seam_costs_afflec.npz \
  --cost-smallrom outputs/rom/seam_costs_afflec.npz \
  --point-size 25 --seam-width 8 --body-alpha 140
```

### 3) Strategy B run (solve on ROM/ogre domain, then reproject to base)

```bash
PYTHONPATH=src /Whisper-WebUI/venv/bin/python examples/solve_seams_from_rom.py \
  --body outputs/suits/afflec_body/base_layer.npz \
  --rom-costs outputs/rom/seam_costs_afflec_realshape_edges.npz \
  --solver shortest_path \
  --weights configs/kernel_weights.yaml \
  --mdl configs/mdl_prior.yaml \
  --fabric-dir configs/fabrics \
  --fabric-id knit_4way_light \
  --sp-allow-unfiltered-fallback \
  --sp-max-edge-length-factor 1.8 \
  --out "$ROOT/B_ogre"

PYTHONPATH=src /Whisper-WebUI/venv/bin/python scripts/build_mesh_vertex_map.py \
  --source-mesh outputs/suits/afflec_body/base_layer.npz \
  --target-mesh outputs/afflec_demo/afflec_body.npz \
  --out "$ROOT/ogre_afflec_vertex_map.npz"

# Preferred when available: use sampler-native correspondence exported at ROM stage.
# cp outputs/rom/afflec_rom_correspondence.npz "$ROOT/ogre_afflec_vertex_map.npz"

PYTHONPATH=src /Whisper-WebUI/venv/bin/python scripts/render_variant_orbits.py \
  --run-root "$ROOT" \
  --run B_ogre \
  --body outputs/suits/afflec_body/base_layer.npz \
  --cost-default outputs/rom/seam_costs_afflec_realshape_edges.npz \
  --cost-smallrom outputs/rom/seam_costs_afflec_realshape_edges.npz \
  --point-size 25 --seam-width 8 --body-alpha 140

set +e
PYTHONPATH=src /Whisper-WebUI/venv/bin/python scripts/reproject_seam_report.py \
  --report "$ROOT/B_ogre/seam_report.json" \
  --source-mesh outputs/suits/afflec_body/base_layer.npz \
  --target-mesh outputs/afflec_demo/afflec_body.npz \
  --vertex-map-file "$ROOT/ogre_afflec_vertex_map.npz" \
  --out "$ROOT/B_ogre_to_base/seam_report.json" \
  --strict-quality \
  --max-mean-distance 0.05 \
  --max-distance 0.15 \
  --min-edge-retention 0.5 \
  --max-target-collision-ratio 0.2
echo $? > "$ROOT/B_ogre_to_base/reproject_exit_code.txt"
set -e

PYTHONPATH=src /Whisper-WebUI/venv/bin/python scripts/render_variant_orbits.py \
  --run-root "$ROOT" \
  --run B_ogre_to_base \
  --body outputs/afflec_demo/afflec_body.npz \
  --cost-default outputs/rom/seam_costs_afflec.npz \
  --cost-smallrom outputs/rom/seam_costs_afflec.npz \
  --point-size 25 --seam-width 8 --body-alpha 140
```

### 4) Lineage audit and summary

```bash
PYTHONPATH=src /Whisper-WebUI/venv/bin/python scripts/audit_mesh_lineage.py \
  --body outputs/afflec_demo/afflec_body.npz \
  --rom-costs outputs/rom/seam_costs_afflec.npz \
  --rom-meta outputs/rom/afflec_rom_run.json \
  --seam-ogre "$ROOT/B_ogre/seam_report.json" \
  --seam-reprojected "$ROOT/B_ogre_to_base/seam_report.json" \
  --out-json "$ROOT/lineage_audit.json" \
  --out-csv "$ROOT/lineage_audit.csv"
```

### 5) Decision gates

Gate A1:
- Strategy A must pass native topology consistency (body/cost lengths equal,
  seam indices in range).

Gate B1:
- Strategy B reprojection must pass strict quality.
- `reproject_exit_code.txt` must be `0`.
- `reprojection.quality_ok` must be `true`.

Gate B2:
- No severe seam collapse on transfer.
- `edge_retention_ratio >= 0.5`.
- `target_vertex_collision_ratio <= 0.2` (many-to-one vertex collapse gate).

Gate V1 (manual qualitative check):
- Visual seam/pathing must be interpretable and mesh-conforming in:
  - `A_base/overlay_orbit.webm`,
  - `B_ogre/overlay_orbit.webm` (ROM-native),
  - `B_ogre_to_base/overlay_orbit.webm` (reprojected).

### 6) Freeze policy rule

- If B1 fails, freeze Strategy A temporarily.
- If B1 passes and V1 is acceptable, compare A vs B outputs for seam stability
  and choose canonical policy.
- Record decision in this file under "Decision Record" with date and run root.

## Decision Record

- 2026-02-13 (base-layer strict multi-loop run: `outputs/seams_run/base_layer_multiloop_20260213_111639`):
  - Config:
    - shortest-path on `outputs/suits/afflec_body/base_layer.npz` (`9438`)
    - `--sp-require-loop --sp-loop-strict --sp-loop-count 2 --sp-loop-waypoints 2`
  - Outcome:
    - total cost `9.3334`,
    - strong panel sparsity in loop output:
      - nonempty panels: hip/waist left-right and thigh/hip left only,
      - empty panels: both chest/neck and both waist/chest, plus thigh/hip right.
  - Dominant warnings:
    - `anchors disconnected; using largest connected component anchors`
    - `loop closure unavailable (no alternate return path)`
    - `no path between anchors`
    - `requested 2 loops but only 1 disjoint simple loops found`
  - Interpretation:
    - strict loop mode is functioning (drops invalid loops),
      but upper-body loop feasibility on the `9438` graph remains poor.

- 2026-02-13 (strict multi-loop probe: `outputs/seams_run/looping_strict_probe_20260213_104847`):
  - Configuration:
    - shortest-path with `--sp-require-loop --sp-loop-strict --sp-loop-count 2 --sp-loop-waypoints 2`.
  - `A_v3240_loop2`:
    - produced loop-heavy seams on all 8 panels (nonempty panels: `8/8`).
    - still reports some candidate-loop warnings because diagnostics aggregate all
      attempted anchor-pair closures, including rejected non-simple candidates.
  - `B_v9438_loop2`:
    - nonempty loop seams only on `3/8` panels; remaining panels dropped to no
      seam when no strict-valid loop was available.
    - warnings include `no path between anchors`, `loop closure unavailable`,
      and `requested 2 loops but only 1 disjoint simple loops found`.
  - Outcome:
    - strict mode now avoids non-simple fallback seams,
      but loop feasibility is still topology/panel dependent.

- 2026-02-13 (looping probe run: `outputs/seams_run/looping_probe_20260213_103313`):
  - Solver mode:
    - shortest-path with `--sp-require-loop` enabled on both topologies.
  - `A_v3240_loop`:
    - warnings include `loop closure produced non-simple cycle` and
      `require_loop enabled but selected seam is not a simple cycle`.
    - edge counts: raw `192`, unique `94`.
  - `B_v9438_loop`:
    - warnings include `loop closure unavailable (no alternate return path)` and
      `require_loop enabled but selected seam is not a simple cycle`.
    - edge counts: raw `42`, unique `19`.
  - `B_v9438_loop_to_v3240` transfer (control map):
    - edge retention `0.0476`,
    - collision ratio `0.85`,
    - quality gate failed (same collapse behavior as non-loop runs).
  - Outcome:
    - loop flag increases candidate seam complexity but does not yet guarantee a
      valid simple cycle under current graph/anchor conditions.

- 2026-02-13 (visual cross-check update on run: `outputs/seams_run/domain_ab_20260213_101158`):
  - Alignment with metrics:
    - `B_ogre_to_base_control` non-conformance is consistent with metrics
      (`mesh_edge_valid_ratio=0.0`, `edge_retention_ratio=0.0526`,
      `target_vertex_collision_ratio=0.85`).
    - Native solves still pass mesh adjacency checks
      (`A_base.mesh_edge_valid_ratio=1.0`, `B_ogre.mesh_edge_valid_ratio=1.0`).
  - Misalignment with prior naming assumptions:
    - user-observed morphology suggests `A_base` and `B_ogre` names are not
      reliable indicators of visual body type in this run family.
  - Action:
    - treat topology counts + lineage metadata as authoritative identifiers;
      treat run-name semantics (`base`/`ogre`) as provisional labels only.

- 2026-02-13 (run: `outputs/seams_run/domain_ab_20260213_101158`):
  - Sampler was regenerated successfully (real ROM path) with:
    - `outputs/rom/domain_ab_20260213_101158/seam_costs_afflec.npz`
    - `outputs/rom/domain_ab_20260213_101158/afflec_rom_correspondence.npz`
  - Strategy A (base-native): pass on native topology checks.
  - Strategy B (ROM-native on `9438` ogre mesh): pass on native topology checks.
  - Critical correspondence finding:
    - sampler-native map is `10475 -> 3240` (SMPL template to base body),
    - B_ogre seam source is `9438`,
    - therefore sampler-native map is incompatible for B_ogre->base reprojection.
  - Control reprojection (`9438 -> 3240` mesh NN map) remains poor:
    - `edge_retention_ratio=0.0526`
    - `target_vertex_collision_ratio=0.85`
    - `B_ogre_to_base_control.mesh_edge_valid_ratio=0.0`
  - Conclusion:
    - mismatch point is now explicit: we still lack a transform-native `9438` correspondence artifact.

- 2026-02-13 (run: `outputs/seams_run/domain_ab_20260213_095932`):
  - Strategy A (base-native): pass on native topology checks.
  - Strategy B (ROM-native -> base reprojection): fails strict transfer gate.
  - Reprojection failure metrics:
    - `reproject_exit_code=2`
    - `mean_distance=0.3475`, `max_distance=0.5329`
    - `edge_retention_ratio=0.0526`
    - `target_vertex_collision_ratio=0.85`
  - On-mesh edge validity (from `decision_metrics.json`):
    - `A_base.mesh_edge_valid_ratio=1.0`
    - `B_ogre.mesh_edge_valid_ratio=1.0`
    - `B_ogre_to_base.mesh_edge_valid_ratio=0.0`
  - Runtime note:
    - sampler refresh was blocked in this environment (`ModuleNotFoundError: smplx`);
      run used existing ROM cost artifacts.

- 2026-02-13 (run: `outputs/seams_run/domain_ab_20260213_051532`):
  - Strategy A (base-native): pass on native topology checks.
  - Strategy B (ROM-native -> base reprojection): fails strict transfer gate.
  - Reprojection failure metrics:
    - `reproject_exit_code=2`
    - `mean_distance=0.3475`, `max_distance=0.5329`
    - `edge_retention_ratio=0.0526`
    - `target_vertex_collision_ratio=0.85`
  - Map artifact check:
    - `mapping_mode=explicit_map:source_to_target`
    - persistent map tracking did not improve transfer quality for this pair.
  - On-mesh edge validity (from `decision_metrics.json`):
    - `A_base.mesh_edge_valid_ratio=1.0`
    - `B_ogre.mesh_edge_valid_ratio=1.0`
    - `B_ogre_to_base.mesh_edge_valid_ratio=0.0`
  - Cross-domain agreement on base topology:
    - `edge_jaccard=0.0`
    - `vertex_jaccard=0.0`
  - Reverse-direction check (`A_base -> ogre`) also fails:
    - run path: `A_base_to_ogre` under same root
    - `edge_count_in=102`, `edge_count_out=0`
    - `target_vertex_collision_ratio=0.9808`
  - Policy status:
    - provisional freeze on Strategy A for interpretable outputs,
      Strategy B remains diagnostic until transfer passes strict gates.

## Edge Divergence: When / Why / How

This is the explicit failure mechanism for Strategy B reprojection.

When:
- Divergence happens in reprojection step, not during the native ROM-domain
  solve. Native solve output can still be coherent on `B_ogre`.

How:
1. Source seam vertices are mapped to target vertices by nearest-neighbor.
2. If many source vertices map to the same target vertex (many-to-one), edges
   collapse (`dst_a == dst_b`) and are dropped.
3. Additional remapped edges become duplicates and are dropped.
4. Result: transferred seam can shrink to a small fragment and become
   semantically uncorrelated with source seam.

Why:
- Large geometric/topological mismatch between source and target domains causes
  high NN distances and mapping collisions.
- This is expected when branch topologies differ strongly and are not in a
  tightly registered correspondence.
- Nearest-neighbor transfer is not bijective. Reversing direction does not
  recover information once many-to-one collapse has occurred.
- Persistent full-mesh map tracking helps reproducibility and inverse-direction
  bookkeeping, but does not fix transfer quality unless the correspondence map
  itself is physically/geometrically valid.

## Strategy 2 Artifact Log (ROM Native AND Reprojected)

These are the frozen artifacts used to ground discussion about:
- morphology (\"ogre\" vs \"normal\"),
- seam validity (mesh-adjacent vs non-mesh edges),
- and the practical consequences of high-collision vertex maps.

Important: the run labels (`base`, `rom`) below are *domain labels used by the
Strategy 2 runner*, not morphology guarantees.

### Bundle: Strategy 2 Smoke (timestamp `20260213_125321`)

Bundle root:
- `outputs/assets_bundles/20260213_125321__domain_ab_strategy2_smoke/`

Render set (user observations + machine checks):
- `outputs/assets_bundles/20260213_125321__domain_ab_strategy2_smoke/renders/base__native_base_seams__orbit__20260213_125321.webm`
  - User: ogre morphology; valid non-loop seam.
  - Manifest: `mesh_edge_valid_ratio=1.0` (`51/51` unique seam edges are mesh edges).
- `outputs/assets_bundles/20260213_125321__domain_ab_strategy2_smoke/renders/base__reprojected_rom_seams__orbit__20260213_125321.webm`
  - User: ogre morphology; invalid scapula single-seam.
  - Manifest: `mesh_edge_valid_ratio=0.0` (`0/1` unique seam edges are mesh edges).
- `outputs/assets_bundles/20260213_125321__domain_ab_strategy2_smoke/renders/rom__native_rom_seams__orbit__20260213_125321.webm`
  - User: normal morphology; seams around mouth + groin.
  - Manifest: `mesh_edge_valid_ratio=1.0` (`17/17` unique seam edges are mesh edges).
- `outputs/assets_bundles/20260213_125321__domain_ab_strategy2_smoke/renders/rom__reprojected_base_seams__orbit__20260213_125321.webm`
  - User: normal morphology; no seam visible.
  - Manifest: `unique_edge_count=0` (all edges collapsed/dropped during transfer).
- `outputs/assets_bundles/20260213_125321__domain_ab_strategy2_smoke/renders/base__mesh_only__orbit__20260213_125321.webm`
  - User: ogre morphology; no seam.
  - Orientation note: user originally described some ogre runs as \"face-down\" vs
    face-forward, but later corrected that ogre is face-up. The render manifests
    record the axis convention (`render_axis.axis_order`) used for each orbit.
- `outputs/assets_bundles/20260213_125321__domain_ab_strategy2_smoke/renders/rom__mesh_only__orbit__20260213_125321.webm`
  - User: normal morphology; no seam.

### Bundle: Strategy 2 With Vertex Map (timestamp `20260213_130000`)

Bundle root:
- `outputs/assets_bundles/20260213_125606__domain_ab_strategy2_with_map/`

Vertex-map orbits (user observations):
- `outputs/assets_bundles/20260213_125606__domain_ab_strategy2_with_map/maps/vertex_map__source_to_target__20260213_130000/map_orbit.webm`
  - User: normal-looking target view with yellow points; mapping between ogre:normal fails;
    ogre appears face-up; likely also needs a 90-degree yaw adjustment for intuitive alignment.
- `outputs/assets_bundles/20260213_125606__domain_ab_strategy2_with_map/maps/vertex_map__source_to_target__20260213_130000/map_orbit_20260213_130000.webm`
  - Same content as above (timestamped copy).
- `outputs/assets_bundles/20260213_125606__domain_ab_strategy2_with_map/maps/vertex_map__target_to_source__20260213_130000/map_orbit.webm`
  - User: normal-looking view with pink points.
- `outputs/assets_bundles/20260213_125606__domain_ab_strategy2_with_map/maps/vertex_map__target_to_source__20260213_130000/map_orbit_20260213_130000.webm`
  - Same content as above (timestamped copy).

Additional render confirmations (same failure signature as smoke bundle):
- `outputs/assets_bundles/20260213_125606__domain_ab_strategy2_with_map/renders/base__native_base_seams__orbit__20260213_130000.webm`
  - User: ogre morphology; valid non-loop seam.
- `outputs/assets_bundles/20260213_125606__domain_ab_strategy2_with_map/renders/base__reprojected_rom_seams__orbit__20260213_130000.webm`
  - User: ogre morphology; invalid scapula single-seam.

Map-quality facts (from `map_manifest.json` in the same bundle):
- The correspondence is not close to bijective.
- `source_to_target_collision_ratio=0.9842` (9438 -> 3240 collapses heavily).
- `target_to_source_collision_ratio=0.9997` (3240 -> 9438 collapses nearly entirely).
- Distance saturation: `mean_distance=0.491m` with `clip=0.15m`, so most points hit the
  max colormap intensity (consistent with the \"yellow\" report).

### Bundle: Strategy 2 Aligned Orientation (timestamp `20260214_050137`)

Bundle root:
- `outputs/assets_bundles/20260214_050137__domain_ab_strategy2_aligned/`

Change vs prior bundles:
- Historically, we attempted to force shared render orientation by pinning
  `--axis-width` in `scripts/protocol_strategy2_bundle.py`. This was brittle:
  it depends on both domains sharing a common world-axis convention.

Current policy (2026-02-14 onward):
- Orbit renderers (`scripts/render_seam_orbit.py`, `scripts/render_vertex_map_orbits.py`)
  use PCA-based canonicalization for `--canonicalize --axis-up auto` so that
  "up" and "width" are inferred from the geometry rather than raw axis extents.
- Within a bundle, both domains must use the same canonicalization policy; the
  render manifest records the inferred axes and statistics.

Follow-up (timestamp `20260214_050754`):
- `--axis-width x` produced shared `render_axis.axis_order=['x','z','y']` across both domains
  in bundle `outputs/assets_bundles/20260214_050754__domain_ab_strategy2_aligned_width_x/`.
  This axis choice is generally more plausible when `z` is the thin \"thickness\" axis.

## Why A Could Look Like A Single Loop But Be Flagged Non-Simple

This is a panel-level graph diagnostic mismatch, not necessarily a visual one.

- The warning is computed per panel seam graph using strict graph criteria:
  every seam vertex must have degree `2` in that panel component.
- In the earlier `A_v3240_loop` run, panels such as
  `chest_circumference_to_neck_circumference_{left,right}` had degree-4
  junction vertices (examples: `26`, `3106`, `3194`), so they were not simple
  cycles despite loop-like visual overlays.
- Visual overlays can still look like a clean single loop because:
  - multiple panel seams are merged in one render,
  - occlusion/projection hides branch points,
  - non-simple branches may overlap the dominant loop path.

Observable evidence fields (from reprojection report):
- `edge_count_in`, `edge_count_out`, `edge_retention_ratio`
- `collapsed_edges_dropped`, `duplicate_edges_dropped`
- `mapped_vertex_count`, `unique_target_vertices`,
  `target_vertex_collision_ratio`
- `mean_distance`, `max_distance`

## Next Checkpoints

1. Decide whether to promote provisional Strategy A freeze to canonical policy.
2. Re-run protocol only after solver/mapping changes and append new decision record entry.
3. Freeze run manifests with body/cost/seam lineage fields so ambiguity does not recur.
4. For any Strategy 2 bundle run, record `manifests/seam_compare_metrics.json` and use it as the
   first-pass quantitative check (edge retention/collision, mesh-edge validity, seam length collapse).

## 2026-02-14 Orientation experiments (Strategy 2)

These ad-hoc manual-rotation experiments were used to debug the misrendering
of "face-up" vs "face-forward" bodies. They are retained for provenance, but
the intended stable solution is PCA-based canonicalization (`--axis-up auto`)
as documented in `docs/seam_overlay_orientation.md`.

- Bundle `20260214_053201__domain_ab_strategy2_rotXneg90` (axis_width=x, yaw_offset=0, rotate_x=-90):
  - `rom__mesh_only` stayed face-down.
  - `base__mesh_only` ogre face-forward but upside-down.

- Bundle `20260214_053750__domain_ab_strategy2_rotXpos90` (axis_width=x, yaw_offset=0, rotate_x=+90):
  - `rom__mesh_only` normal body, still facing up.
  - `base__mesh_only` ogre face-forward and upright.

- Bundle `20260214_054244__domain_ab_strategy2_rotXpos90_Z180` (axis_width=x, yaw_offset=0, rotate_x=+90, rotate_z=180):
  - Renders located in `outputs/assets_bundles/20260214_054244__domain_ab_strategy2_rotXpos90_Z180/`.
  - Pending visual check: intended to flip ROM from face-up to face-forward without inverting base; evaluate and choose canonical rotation.

- Bundle `20260214_060744__domain_ab_strategy2_baseX90Z180_rom0` (axis_width=x, yaw_offset=0, base_rotate=(90,0,180), rom_rotate=(0,0,0)):
  - Per-mesh rotation supported; map orbits use source=ROM rotation, target=base rotation.
  - Check if ROM now faces forward while ogre remains upright.

Pending: choose a single canonical rotation that aligns both base/ROM fronts. Next candidates: rotate_x=90 with rotate_z=180, or rotate_x=180.

## 2026-02-14 Base-only seam solve (no ROM/reprojection)

- Mesh: `outputs/afflec_demo/afflec_body.npz` (3240 verts, faces 6476).
- Cost field: `outputs/rom/seam_costs_afflec.npz`.
- Solver: `smii.seams.solver.solve_seams` (mst, edge_mode=mean).
- Output: `outputs/seams_run/base_only_20260214_092147/seam_report.json`
- Orbit renders:
  - `outputs/seams_run/base_only_20260214_092147/base_only__orbit__20260214_092204.webm` (yaw_offset=90, axis_width=x) — body faced up.
  - `outputs/seams_run/base_only_20260214_092147/base_only_faceforward__orbit__20260214_092559.webm` (axis_width=x, rotate_x=90, rotate_z=180) — face-forward orientation fix.
- Notes: solver emitted “No edges intersect seam vertices; edge costs default to empty mapping” warning; panels=8; uses default MDL prior.

## 2026-02-14 Multi-loop seam solve on human (v3240)

Goal: run the multi-loop shortest-path solver on the `human` topology without cross-topology reprojection.

- Body: `outputs/afflec_demo/afflec_body.npz` (`human_v3240`)
- ROM costs: `outputs/rom/seam_costs_afflec.npz` (`vertex_costs.shape == (3240,)`)
- Command:
  - `PYTHONPATH=src:. ../.venv/bin/python examples/solve_seams_from_rom.py --body outputs/afflec_demo/afflec_body.npz --rom-costs outputs/rom/seam_costs_afflec.npz --solver shortest_path --sp-require-loop --sp-loop-count 2 --no-sp-allow-unfiltered-fallback --out outputs/seams_run/human_v3240__shortest_path__loop2__20260214_102129`
- Outputs:
  - `outputs/seams_run/human_v3240__shortest_path__loop2__20260214_102129/seam_report.json`
  - `outputs/seams_run/human_v3240__shortest_path__loop2__20260214_102129/overlay.png`
  - Orbit: `outputs/seams_run/human_v3240__shortest_path__loop2__20260214_102129/human_v3240__shortest_path__loop2__orbit__20260214_102129.webm`

## 2026-02-14 Orientation mismatch observed (role-labeled smoke bundle)

User-observed artifacts (bundle `outputs/assets_bundles/20260214_102426__smoke_roles/`):

- `outputs/seams_run/human_v3240__shortest_path__loop2__20260214_102129/human_v3240__shortest_path__loop2__orbit__20260214_102129.webm`
  - reported: face-up ogre; valid loop seam
- `outputs/assets_bundles/20260214_102426__smoke_roles/renders/human_v3240__native_seams__orbit__20260214_102426.webm`
  - reported: face-up ogre; valid on-mesh non-loop seam
- `outputs/assets_bundles/20260214_102426__smoke_roles/renders/human_v3240__reprojected_seams_from__ogre_v9438__orbit__20260214_102426.webm`
  - reported: face-up ogre; invalid inter-scapula non-conforming seam
- `outputs/assets_bundles/20260214_102426__smoke_roles/renders/ogre_v9438__native_seams__orbit__20260214_102426.webm`
  - reported: forward-facing human; crotch and mouth conforming seams
- `outputs/assets_bundles/20260214_102426__smoke_roles/renders/ogre_v9438__reprojected_seams_from__human_v3240__orbit__20260214_102426.webm`
  - reported: forward-facing human; no seams
- `outputs/assets_bundles/20260214_102426__smoke_roles/renders/human_v3240__mesh_only__orbit__20260214_102426.webm`
  - reported: up-facing ogre; no seams
- `outputs/assets_bundles/20260214_102426__smoke_roles/renders/ogre_v9438__mesh_only__orbit__20260214_102426.webm`
  - reported: forward-facing human; no seams

Interpretation:
- Orientation alignment is still not enforced; per-mesh coordinate frames likely differ (e.g. one mesh encodes height in `z`, another in `y`).
- Role tags must not be derived solely from vertex count; treat them as expected labels, and add deterministic render canonicalization (`axis_up=auto`) to align facing direction.

## Related Docs

- `docs/mesh_provenance_afflec.md`
- `docs/ogre_artifact_diagnostics.md`
- `docs/pipeline_runner.md`
