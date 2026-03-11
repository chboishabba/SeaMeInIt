## ROM Sample Morphology And Transfer Contract

Date: 2026-03-10

### Purpose

This note closes the ambiguity around two related questions:

- how representative ROM sample morphologies are chosen and emitted
- whether current "back to body" behavior is a true inverse transform

It is a contract for the current repo state, not a statement of ideal future theory.

### M2.1 Representative Sample Selection Policy

Representative ROM sample artifacts are selected from the actual per-pose sampler run.
They are not synthetic showcase poses and they are not chosen by hand after the fact.

Current deterministic selection anchors are:

1. `max_field_l2_norm`
   The pose with the largest L2 norm of the sampled `seam_sensitivity` field.
2. `max_displacement_mean_norm`
   The pose with the largest mean vertex displacement magnitude from neutral.
3. `median_field_l2_norm`
   The pose nearest the median field L2 norm across the run.
4. `max_weight`
   The highest-weight sample from the sweep/schedule.

Deduplication rules:

- duplicate poses selected by multiple anchors are emitted once
- remaining slots are filled by descending `field_l2_norm`
- tie-break is deterministic: larger metric first, then `pose_id`

Default count:

- emit up to `4` representative samples unless the caller requests another count

Rationale:

- `max_field_l2_norm` shows the strongest operator response
- `max_displacement_mean_norm` shows the strongest visible morphology change candidate
- `median_field_l2_norm` shows a non-extreme interior sample
- `max_weight` preserves the schedule's intended emphasis

### M2.2 Sample Artifact Contract

When sample emission is enabled, the sampler writes a dedicated directory containing:

- `rom_sample_manifest.json`
- one `.npz` mesh per selected pose

Mesh contract:

- vertices are the transform-native posed vertices produced directly by the ROM sampler backend
- faces are included when the backend exposes a native face array
- these samples stay on the sampler-native topology; they are not geometry-remapped to the fitted body topology

Manifest contract:

- source lineage:
  - body path/hash
  - params path
  - sweep/schedule path
  - source vertex count
  - target vertex count
- selection policy:
  - anchor names
  - fill policy
  - requested sample count
- inverse/transfer status:
  - `true_inverse_available`
  - `current_transfer_mode`
  - acceptance notes
- per-sample entries:
  - `pose_id`
  - `selection_reason`
  - `weight`
  - `field_l2_norm`
  - `field_max`
  - `displacement_mean_norm`
  - `displacement_max_norm`
  - `topology`
  - `mesh_path`
  - `geometry_sha256`

Interpretation rule:

- these artifacts exist to show posed/deformed morphology directly
- they do not replace the aggregate operator field
- they do not imply that a valid inverse back to the fitted body exists

### M2b.1 Current Inverse Audit

Current code audit result:

- `smii.rom.basis.KernelProjector.encode()` is only an inverse projection inside the same orthonormal basis/field space
- `scripts/build_mesh_vertex_map.py` builds nearest-neighbor correspondence maps, not invertible geometry transforms
- `scripts/reproject_seam_report.py` reprojects seam edge indices across topologies using nearest-neighbor or map lookup, not inverse geometry reconstruction
- `smii.rom.sampler_real --out-correspondence` exports nearest-neighbor source/target correspondence with quality stats, not an inverse deformation map

So today there is no true inverse transform from an internalized/ROM-native morphology domain back to the fitted body mesh.

### M2b.2 Current Reality

Current "back-transfer" language must mean:

- correspondence
- reprojection
- approximate transfer with explicit quality diagnostics

It must not mean:

- invertible transform
- exact reconstruction
- round-trip geometry equivalence

### M2b.3 Acceptance Criteria For Approximate Transfer

Approximate transfer is only acceptable for interpretation when all of the following hold:

1. topology lineage is explicit in manifests
2. transfer quality metrics are recorded
3. collision and edge-retention are within configured gates
4. the transferred artifact is described as transferred/reprojected, not native
5. no claim of exact inverse is made

If those conditions fail, the transfer is diagnostic only and must not be treated as production-valid seam evidence.

### M2b.4 Requirements For A Future True Inverse

Before the repo can claim an inverse transform, it needs all of:

1. a defined forward transform that maps one geometry/domain into another
2. a defined inverse or pseudo-inverse algorithm for that same transform
3. round-trip error measurement on geometry and relevant seam structures
4. bounded reconstruction error thresholds
5. evidence that the inverse preserves the design semantics needed for seam interpretation

Until then, the safe language is:

- operator-native sample morphology
- correspondence/reprojection transfer
- approximate back-transfer
