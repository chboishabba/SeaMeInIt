# Body Lineage Protocol (v1)

Date: 2026-02-13

This document defines the intended protocol for generating and labeling body
meshes so seam/ROM debugging does not depend on ambiguous filenames like
`afflec_body.npz` or legacy folders like `outputs/suits/afflec_body/`.

## Principles

- **Single entrypoint for baseline body**: the baseline body for a subject comes
  from images (fixture or real), fit into SMPL-X parameters, then exported as a
  mesh. This is the only stage that should be described as “the subject body”.
- **Derived meshes are labeled as derived**: undersuit layers (base/insulation/
  comfort) are downstream products of the baseline body. They must never be
  named or described as “the body”.
- **ROM produces costs, not a new neutral mesh**: the real ROM sampler emits a
  cost field sized to the mesh it was sampled on. ROM does not mutate the mesh
  file; it changes the objective field.
- **Cross-projection is diagnostic**: if you render a seam on a different mesh
  topology than it was solved on, the output must be labeled as transferred and
  must carry quality gates (distance, collision, retention).

## Canonical stages (and what they mean)

1. **Body fit (baseline)**
   - inputs: images (+ optional measurement priors)
   - output: body mesh NPZ + parameter payload JSON
   - label: `stage=fit_body`

2. **Derived undersuit layers**
   - inputs: baseline body mesh
   - output: `base_layer.npz` etc. + metadata
   - label: `stage=undersuit_base_layer` (or insulation/comfort)

3. **ROM sampling**
   - inputs: a specific mesh topology (baseline body OR derived base layer)
   - output: seam cost field NPZ + provenance JSON + optional correspondence NPZ
   - label: `stage=rom_costs`

4. **Seam solving**
   - inputs: mesh + ROM costs (or uniform costs for “base-only” solve)
   - output: `seam_report.json`
   - label: `stage=seam_solve`
   - required metric: seam edges must be validated against mesh adjacency.

5. **Transfer (optional)**
   - inputs: `seam_report.json` + source mesh + target mesh + vertex map
   - output: transferred `seam_report.json`
   - label: `stage=seam_transfer`

6. **Rendering**
   - inputs: render body + seam report + cost field
   - output: orbit/still artifacts + manifest
   - label: `stage=render`

## Output directory convention (recommended)

Avoid fixed, overwrite-prone paths like `outputs/afflec_demo`.

Use timestamped run roots and keep the *exact* mesh used by downstream stages:

```bash
export TS=$(date -u +%Y%m%d_%H%M%S)
export SUBJECT=afflec_fixture
export BODY_DIR=outputs/bodies/${SUBJECT}/${TS}
export SUIT_DIR=outputs/suits/${SUBJECT}/${TS}
export ROM_DIR=outputs/rom/${SUBJECT}/${TS}
export SEAMS_DIR=outputs/seams_run/${SUBJECT}/${TS}
```

## Labeling convention (recommended)

Every stage should record and display the following stable identifiers:

- `subject`: string (fixture or real subject id)
- `stage`: one of `fit_body`, `undersuit_base_layer`, `rom_costs`, `seam_solve`,
  `seam_transfer`, `render`
- `topology`: `v{vertex_count}_f{face_count}`
- `body_hash`: sha256 of vertices array (or file sha256)

Optional:

- `units_guess`: based on bbox height (for diagnosing scale anomalies)
- `axis_convention`: which axis is treated as up/right/depth in render

## “Solve on base” vs “solve on ROM” (both, but explicit)

The protocol supports solving in two modes that should ultimately agree when the
pipeline is coherent:

- Base solve: uniform costs (debug topology/pathing without ROM).
- ROM solve: seam costs from ROM sampler (optimize for motion sensitivity).

If both are rendered on the same mesh topology, no transfer is needed. Transfer
is only for “compare on a different reference mesh” and must be gated.

