# Afflec Mesh Provenance and Topology Lineage

Date: 2026-02-13

This document disambiguates the Afflec mesh artifacts that have caused repeated
confusion in seam debugging.

For intended-vs-observed seam behavior and solve-domain decision framing, see
`docs/seam_pipeline_intended_vs_observed.md`.

## Short answer

- `outputs/afflec_demo/afflec_body.npz` is the fitted body mesh emitted by
  `smii.app afflec-demo` from Afflec fixture images and SMPL assets.
- `outputs/rom/afflec_canonical_basis.npz` is a ROM basis artifact generated
  from some input mesh vertices. It is not, by itself, proof that Afflec images
  were used for that specific topology.
- The repo currently contains two active topology families:
  - base/demo branch: `3240` vertices
  - legacy realshape/suit branch: `9438` vertices

Both are valid when intentionally used, but they must not be mixed in a single
solve/render chain without explicit reprojection.

## Role naming (human vs ogre)

For debugging, we use the following **working** role names. These are not value
judgements, just a way to stop ambiguous filenames from derailing analysis.

- `human`:
  the direct image-fit / SMPL-X derived body mesh (currently the `v3240` family).
  Example: `outputs/afflec_demo/afflec_body.npz` (`3240` verts).

- `ogre`:
  the high-topology "internalized" body representation used by some ROM and suit
  pipelines (currently the `v9438` family). It may incorporate remeshing,
  canonicalization, or "realshape" projections and can look visually unlike the
  `human` mesh in renders.
  Examples:
  - `outputs/suits/afflec_body/base_layer.npz` (`9438` verts)
  - `outputs/rom/seam_costs_afflec_realshape_edges.npz` (`9438` vertex costs)

Important: "ogre" here means "the `9438` branch we keep confusing with the
`3240` branch". It does **not** imply that ROM is "deforming" the body; seam
solvers do not change geometry, they only select edges on whatever mesh you pass.

## ROM operator vs `ogre` topology

The archived Sprint R / three-kernel ROM thread is explicit about the math:
ROM is treated as a compressed operator over admissible pose space, represented
by a canonical basis plus pose-indexed coefficient vectors. In repo terms:

- operator-level ROM:
  - canonical basis (`B`)
  - coefficient samples / aggregation semantics
  - schedule/completeness logic
- domain-level ROM projections:
  - seam-cost NPZ sized to a specific mesh topology
  - seam reports solved on that topology
  - render artifacts of whichever body mesh was selected

This means:

- `ogre` is **not** the ROM invariant itself.
- `ogre` is the current working provenance label for the `v9438` topology family.
- `human` is the current working provenance label for the `v3240` topology family.
- a ROM-derived artifact can exist in either domain if it has been projected
  into that topology.

So when asking "where is the ROM invariant?", the answer is not "the ogre
render". The closest current operator-level artifacts are:

- `docs/rom_definition.md` for the mathematical object,
- basis artifacts such as `outputs/rom/afflec_canonical_basis.npz`,
- sampler/certificate metadata describing how pose-space statistics were
  aggregated before projection to seam costs.

Operationally, if an artifact is viewable as a mesh orbit, it is already a
domain-level object and should not be treated as the pure ROM operator.

## Critical note: output paths are not stable provenance

This repo historically overwrote files under fixed paths like
`outputs/afflec_demo/afflec_body.npz`. That means a later run can change the
mesh at the same path (different vertices, different units/scale, different
topology) while older downstream artifacts still reference the old path in their
metadata.

Treat **hash + vertex/face counts** as authoritative provenance, not path
strings.

## Fit provenance and diagnostics (current contract)

The Afflec image-fit stage now emits three distinct artifacts under the body
output root:

- `afflec_observations.json`:
  the 2D keypoints / silhouette-bbox observations used by the reprojection fit.
- `afflec_raw_regression.json`:
  raw image-regressed pose/shape estimates plus `images_used`, `detector`,
  trust status, per-view confidence/measurement summaries, and optional
  optimization metrics.
- `afflec_measurement_fit.json`:
  measurement-model refinement output, now linked back to the raw regression via
  provenance fields and raw measurements.
- `afflec_fit_diagnostics.json`:
  canonical audit artifact for deciding whether the run is merely
  "photo-driven" or numerically credible.

The diagnostics status should be interpreted as:

- `PASS`: numerically plausible and detector path is not obviously degraded.
- `WARN`: usable for debugging, but not trustworthy as a calibrated baseline.
- `FAIL`: do not treat the emitted body as a credible anthropometric fit.

Important:
- `bbox` is retained as a coarse fallback and is expected to emit at least a
  `WARN` status in many cases.
- `mediapipe` remains the preferred path when a higher-trust body estimate is
  required.
- `fit_mode=auto` now prefers the reprojection optimizer and only falls back to
  the older heuristic path when reprojection cannot run.

## Historical scale anomaly (afflec-demo)

As of 2026-02-13, the local `outputs/afflec_demo/afflec_smplx_params.json`
contains `scale ≈ 0.0073`, which produces a **centimeter-scale** mesh in
`outputs/afflec_demo/afflec_body.npz`.

This is consistent with the fixture/demo nature of the pipeline, but it means:

- do not assume `outputs/afflec_demo/afflec_body.npz` is already in real-world
  meters unless validated per-run, and
- do not use the filename `afflec_body` as evidence it represents a physically
  sized “Ben baseline”.

Newer runs should be judged from `afflec_fit_diagnostics.json` first, not by
comparing against that historical anomaly in isolation.

## Provenance map (current local artifacts)

### Key artifacts (paths -> role/topology)

| Path | Role tag | Topology | Notes |
| --- | --- | --- | --- |
| `outputs/afflec_demo/afflec_body.npz` | `human` | `v3240` | Image-fit / SMPL-X derived mesh (fixture/demo scale may vary). |
| `outputs/rom/seam_costs_afflec.npz` | `human` | `v3240` | ROM vertex costs in the same topology as `afflec_body.npz`. |
| `outputs/suits/afflec_body/base_layer.npz` | `ogre` | `v9438` | Undersuit base layer derived from `afflec_body` (different topology). |
| `outputs/rom/seam_costs_afflec_realshape_edges.npz` | `ogre` | `v9438` | ROM-derived costs projected into the `v9438` domain (includes edges). |

1. Image ingest and fit
   - command: `python -m smii.app afflec-demo ...`
   - artifact: `outputs/afflec_demo/afflec_body.npz`
   - observed count: `3240` vertices

2. ROM sampling on fitted body
   - command: `python -m smii.rom.sampler_real --body outputs/afflec_demo/afflec_body.npz ...`
   - artifacts:
     - `outputs/rom/seam_costs_afflec.npz` (`3240` vertex costs)
     - `outputs/rom/afflec_rom_run.json` (`meta.vertex_count=3240`)
   - sampler metadata also records source-template remapping
     (example: `source_vertex_count=10475 -> target_vertex_count=3240`).

3. Legacy high-topology branch (historical runs)
   - artifacts:
     - `outputs/rom/seam_costs_afflec_realshape*.npz` (`9438` vertex costs)
     - `outputs/seams_run/afflec_real_*_edges/seam_report.json` (indices near
       `9432`, no provenance block in older reports)
     - `outputs/suits/afflec_body/base_layer.npz` (`9438` vertices)
   - these runs are a different topology branch and need explicit transfer for
     base-body visual inspection.

### Are we applying ROM to the human mesh, or is ogre the ROM internalization?

Both exist in the repo:

- ROM costs in the `human` domain (`v3240`):
  - `outputs/rom/seam_costs_afflec.npz` has `vertex_costs.shape == (3240,)`.
  - This is "ROM findings applied to the human mesh" (same vertex count).

- ROM costs in the `ogre` domain (`v9438`):
  - `outputs/rom/seam_costs_afflec_realshape_edges.npz` has `vertex_costs.shape == (9438,)`.
  - This implies some upstream step has produced a `v9438` representation of the
    body and projected the ROM-derived quantities into that topology.

Operationally:
- If you solve seams using `seam_costs_afflec.npz`, you are solving on `human` (`v3240`).
- If you solve seams using `seam_costs_afflec_realshape_edges.npz`, you are solving on `ogre` (`v9438`).

That still does **not** mean the `v9438` branch is "the ROM invariant". It only
means ROM-derived quantities were projected into that topology before solving.

## Historical inverse-transform intent (important)

Part of the earlier project intent was stronger than today's repo contract:

1. fit a normal SMPL-X-style human body from photos,
2. apply a ROM-related internalization/deformation into a different solve domain
   (historically described as the `ogre` morphology),
3. solve seams on that internalized morphology because it was assumed to encode
   movement-respecting geometry,
4. then bring the solved seam back through an inverse transform to the normal
   fitted body.

That inverse step is **not implemented or validated** in the current repo.

Current reality:

- seam solving on a different topology/domain is implemented,
- nearest-neighbour / correspondence-based reprojection is implemented,
- strict quality diagnostics for reprojection are implemented,
- but a true inverse of the ROM/internalization transform is not available.

Why this matters:

- if the internalization behaves more like a projection, collapse, or lossy
  topology/domain transfer than an invertible deformation, then there may be no
  exact inverse to recover a seam on the original fitted body,
- the Dashi-side formalism in `all_code44.txt` and `all_code48.txt` reinforces
  this caution: projection-like operators are not generally invertible, and
  "kernel" plus "admissibility lens" should not be confused with a reversible
  geometric transform.

Operational consequence:

- today, any "back to body mesh" step should be treated as correspondence-based
  transfer / approximation, not as a mathematically justified inverse ROM
  application.
- if the project requires a true inverse, that must become an explicit design
  track with its own contract and acceptance tests.

4. Reprojection bridge
   - commands:
     - `python scripts/build_mesh_vertex_map.py --source-mesh ... --target-mesh ... --out ...`
     - `python scripts/reproject_seam_report.py --report ... --source-mesh ... --target-mesh ... --vertex-map-file ...`
   - only valid when source mesh truly matches the seam report topology.
   - large nearest-neighbour distances or collapsed edges indicate wrong source.

## Why the ambiguity happened

- Some older artifacts were generated before strict provenance fields were added
  to seam reports and basis metadata.
- `outputs/suits/afflec_body/*` predates the current `outputs/afflec_demo/afflec_body.npz`
  and uses a different vertex topology.
- Path names alone (`afflec_body`) are not sufficient provenance; vertex counts
  and hashes are required.

## Operational invariants

- Solve invariant: `len(body.vertices) == len(rom.vertex_costs)` must hold.
- Render invariant: all seam edge indices must be `< len(render_body.vertices)`.
- Transfer invariant: reprojection requires known source mesh; check reported
  mean/max mapping distances before trusting transferred seams.

## Recommended run chain (single topology, no ambiguity)

```bash
export TS=$(date -u +%Y%m%d_%H%M%S)
export BODY_DIR=outputs/bodies/afflec_fixture/${TS}
export ROM_DIR=outputs/rom/afflec_fixture/${TS}

PYTHONPATH=src /Whisper-WebUI/venv/bin/python -m smii.app afflec-demo \
  --images tests/fixtures/afflec \
  --output "${BODY_DIR}" \
  --detector bbox

PYTHONPATH=src /Whisper-WebUI/venv/bin/python -m smii.rom.sampler_real \
  --body "${BODY_DIR}/afflec_body.npz" \
  --poses data/rom/afflec_sweep.json \
  --weights data/rom/joint_weights.json \
  --out-correspondence "${ROM_DIR}/rom_correspondence.npz" \
  --out-costs "${ROM_DIR}/seam_costs.npz" \
  --out-meta "${ROM_DIR}/rom_run.json" \
  --mode diagonal
```

Then keep seam solving and rendering on that same body/cost pair.

The correspondence NPZ above is generated at the ogre/ROM stage itself and is
the preferred mapping artifact for any downstream cross-topology seam transfer.

## Lineage audit helper

Use the audit helper to capture provenance snapshots with automated mismatch
checks:

```bash
PYTHONPATH=src /Whisper-WebUI/venv/bin/python scripts/audit_mesh_lineage.py \
  --body outputs/afflec_demo/afflec_body.npz \
  --basis outputs/rom/afflec_canonical_basis.npz \
  --rom-costs outputs/rom/seam_costs_afflec.npz \
  --rom-meta outputs/rom/afflec_rom_run.json \
  --seam-ogre outputs/seams_run/<ogre_run>/seam_report.json \
  --seam-reprojected outputs/seams_run/<reprojected_run>/seam_report.json
```

The command emits timestamped JSON/CSV under `outputs/seams_run/` unless
`--out-json` / `--out-csv` is provided.

For seam transfer, prefer strict reprojection quality checks:

```bash
PYTHONPATH=src /Whisper-WebUI/venv/bin/python scripts/reproject_seam_report.py \
  --report <source seam_report.json> \
  --source-mesh <source mesh npz> \
  --target-mesh outputs/afflec_demo/afflec_body.npz \
  --vertex-map-file <vertex_map.npz> \
  --out <reprojected seam_report.json> \
  --strict-quality --max-mean-distance 0.05 --max-distance 0.15 \
  --min-edge-retention 0.5 --max-target-collision-ratio 0.2
```
