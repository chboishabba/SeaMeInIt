# Layer Separation Contract (Human/Ogre, ROM, Rendering)

This document exists to prevent repeated ambiguity caused by overloading the
word "canonical" and conflating provenance/identity with geometry/rendering.

## Terms (Do Not Mix)

### 1) Identity Layer (Provenance)

Role labels such as `human` and `ogre` are **provenance assertions** about where
an artifact came from in the pipeline (image-ingest fit, undersuit/internalized
branch, etc.).

- Roles must not be inferred from vertex count, topology, or morphology.
- Roles must be explicit in CLI flags and recorded in manifests (hashes).

### 2) Geometry Layer (Topology / Domain)

Topology tags such as `v3240`, `v9438` refer to **mesh connectivity** and vertex
indexing. They decide what operations are valid:

- Seam solving requires a consistent adjacency graph.
- Seam reprojection across topologies is lossy unless a true correspondence is
  available.

The "canonical domain" decision is the Strategy A vs Strategy B question:

- Strategy A: solve seams on the base (human) domain first, then evaluate/project to ROM.
- Strategy B: solve seams on the ROM/ogre domain first, then reproject to base.

### 3) Render Layer (Rotation Normalization)

Orbit renders and overlay PNGs require a stable view frame so comparisons are
meaningful. Render normalization is a **visualization contract only**:

- It must not be used to infer identity.
- It must not be described as ROM-basis canonicalization.

The render tools use `--canonicalize --axis-up auto` to normalize rotation via
PCA + robust tail statistics (see `docs/seam_overlay_orientation.md`).

## Contracts (What Must Always Be True)

- Artifact stems must include `role` + `vNNNN` tags (see
  `docs/seam_pipeline_intended_vs_observed.md`).
- Within a Strategy 2 bundle, base+ROM renders must share a comparable render
  frame (ROM aligned to base for viewability).
- Manifests must record mesh hashes so "identity" is defensible.

