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

## Critical note: output paths are not stable provenance

This repo historically overwrote files under fixed paths like
`outputs/afflec_demo/afflec_body.npz`. That means a later run can change the
mesh at the same path (different vertices, different units/scale, different
topology) while older downstream artifacts still reference the old path in their
metadata.

Treat **hash + vertex/face counts** as authoritative provenance, not path
strings.

## Current scale anomaly (afflec-demo)

As of 2026-02-13, the local `outputs/afflec_demo/afflec_smplx_params.json`
contains `scale ≈ 0.0073`, which produces a **centimeter-scale** mesh in
`outputs/afflec_demo/afflec_body.npz`.

This is consistent with the fixture/demo nature of the pipeline, but it means:

- do not assume `outputs/afflec_demo/afflec_body.npz` is already in real-world
  meters unless validated per-run, and
- do not use the filename `afflec_body` as evidence it represents a physically
  sized “Ben baseline”.

## Provenance map (current local artifacts)

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
