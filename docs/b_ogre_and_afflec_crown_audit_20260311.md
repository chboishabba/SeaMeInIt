## `B_ogre` Forward Object And Afflec Crown Audit

Date: 2026-03-11

### Scope

This note closes the first audit pass for:

1. the exact historical forward object behind `B_ogre`
2. the current Afflec crown / egg-skull failure

It is intentionally evidence-first and does not claim a fix.

### `B_ogre` Findings

#### Exact forward object identified

The strongest currently-audited historical `B_ogre` object is a real native
solve domain on the `9438`-vertex suit/body mesh:

- mesh: `outputs/suits/afflec_body/base_layer.npz`
- mesh SHA256: `b122dc2cf8b075a5a5bcc0c124a075247268332203df7873c36de65e4027695c`
- topology: `9438` vertices / `18872` faces
- ROM costs: `outputs/rom/seam_costs_afflec_realshape_edges.npz`
- ROM costs SHA256: `750e0472648fff6a4f324cd4b34e78648dd8c878a1b2acbd85a0c3a3c57f50d8`
- ROM cost topology: `9438` vertex costs

Evidence:

- `outputs/seams_run/domain_ab_20260213_101158/lineage_audit.json`
- `outputs/seams_run/domain_ab_20260213_051532/decision_metrics.json`

The native `B_ogre` seam report in the `20260213` domain-A/B runs has:

- `body_path = outputs/suits/afflec_body/base_layer.npz`
- `body_vertex_count = 9438`
- `rom_costs_path = outputs/rom/seam_costs_afflec_realshape_edges.npz`
- `rom_vertex_count = 9438`
- native seam `mesh_edge_valid_ratio = 1.0`

So `B_ogre` was not merely a label pasted onto a `3240` Afflec solve. There was
an actual `9438`-topology forward solve object in those runs.

#### What remains unresolved

This audit does **not** prove that the old visually ogre-like appearance came
from real transformed geometry alone.

The old ogre-like visuals are still concentrated in overlay/orbit artifacts such
as:

- `outputs/seams_run/domain_ab_20260213_101158/B_ogre_to_base_control/overlay_front_3q.png`
- `outputs/seams_run/domain_ab_20260213_101158/B_ogre_to_base_control/overlay_orbit.webm`
- `outputs/seams_run/variant_matrix_20260213_035227/shortest_path_knit_4way_light_grain100_ogre_to_afflec_body/overlay_front_3q.png`

At the same time, historical notes already say some native `B_ogre` views looked
normal/base-like while transferred control views looked ogre-like. That means
the visible pathology can still be a mixture of:

- real domain geometry differences
- reprojection collapse
- render-axis / camera / overlay artifact
- stage-label confusion

#### Back-transfer conclusion from this audit

The historical `9438 -> 3240` return path is not a usable production back-transfer.

In `outputs/seams_run/domain_ab_20260213_101158/B_ogre_to_base_control/seam_report.json`:

- `mapped_vertex_count = 20`
- `unique_target_vertices = 3`
- `target_vertex_collision_ratio = 0.85`
- `mean_distance = 0.3475`
- `max_distance = 0.5329`
- `edge_retention_ratio = 0.0526`
- `quality_ok = false`

So the old control transfer did not preserve enough seam structure to count as
an inverse or even a good approximate transfer.

### Afflec Crown Findings

#### What is visibly wrong

Current exported Afflec bodies can show a strongly elongated top-of-skull /
egg-shaped crown. This is most obvious in:

- `outputs/bodies/afflec_fixture/20260309_reprojection_mediapipe_calibrated_complete3/afflec_body.npz`

Using a crude top-5%-of-height crown XY eccentricity proxy, that exported body
looks substantially more anisotropic than the other current Afflec outputs.

#### What the audit narrowed down

The calibrated `mediapipe` runs `complete2` and `complete3` have matching:

- raw reprojection diagnostics
- refined SMPL-X parameter payloads
- scale
- beta magnitudes

But only `complete3` has the final exported `afflec_body.npz`.

This means the current visible crown pathology is **not** explained by a change
in fitted parameters between those two runs. The most likely remaining loci are:

1. final mesh generation from the fitted parameters
2. post-fit repair / component filtering / mouth capping / export
3. a distortion already latent in the fitted geometry that only becomes visible
   in the final exported mesh

So the crown issue is now narrowed to the late body-generation/export path, not
to a last-minute parameter drift between the calibrated runs.

#### What remains unresolved

This audit does **not** yet isolate the exact line where the crown gets worse,
because the repo does not currently persist enough pre-export intermediate head
meshes to compare:

- raw reprojection mesh
- refined pre-repair mesh
- post-repair pre-save mesh

That instrumentation gap is now the blocking item for M6.

### Operational Conclusions

1. `B_ogre` had a real `9438`-topology forward solve object.
2. The historical Afflec-facing return path was catastrophically lossy and
   cannot be treated as a valid inverse.
3. Current Afflec crown distortion is most likely a late-stage body export /
   repair issue, or a latent fitted-geometry issue revealed there.
4. Before implementing a new back-transfer, the repo should:
   - tie the spec to the actual `9438` forward object,
   - add export-stage morphology checkpoints for the Afflec body pipeline.

### Immediate Next Steps

1. Add export-stage body checkpoints for Afflec runs:
   - raw reprojection mesh
   - refined pre-repair mesh
   - repaired/export-ready mesh
2. Define a skull plausibility gate from those checkpoints.
3. Write the back-transfer spec against the actual `9438` forward object rather
   than generic nearest-neighbor reprojection.
