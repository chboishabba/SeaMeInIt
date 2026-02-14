# Body Lineage Protocol v1 (2026-02-13)

## Phase

Pipeline stabilization: single body entrypoint + unambiguous mesh labeling + correspondence visualization.

## Objective

Make it impossible to confuse:

- who/what a mesh represents (Ben fixture vs generic vs derived undersuit layer),
- which topology it lives on (vertex/face counts),
- which space it is in (baseline body vs derived layer vs ROM cost domain),
- and whether a seam is native-solved vs transferred.

## Constraints

- One canonical body entrypoint for “Afflec fixture subject”: images -> SMPL-X fit -> body mesh.
- Downstream stages must not silently overwrite or “reuse” ambiguous filenames without provenance.
- Do not commit binaries; provide CLI commands to regenerate artifacts.
- All cross-topology transfer is diagnostic unless it passes explicit quality gates.

## Deliverables

1. **Docs**: a single, explicit protocol for body mesh generation and naming/labeling.
2. **Audit artifacts**: a “mesh registry” JSON with hashes/stats for all meshes used in a run.
3. **Renderer fixes**: remove unstable axis auto-inference that can create “ogre-like” silhouettes.
4. **Vertex-map orbit**: render a mesh-only (and optional mapping-line) orbit to visualize NN map quality.
5. **Seam report metrics**: always record mesh-edge conformance for native solves and transferred solves.

## Success Criteria

- A run folder can be understood from its manifests alone (no guessing from filenames).
- “Ogre morphology” is explainable and intentionally reproducible (explicit body path + renderer axis settings).
- We can produce a dedicated orbit that demonstrates correspondence quality (collision/retention) visually.
- Every seam report includes a pass/fail metric for “walks on mesh edges”.

## Open Questions

- Which mesh is the canonical “Ben baseline”: fitted SMPL-X body, or derived undersuit base layer?
- Are units/scale in the current `afflec-demo` output correct (meters) or a known fixture artifact?
- Do we keep the legacy `9438` topology branch, or treat it as deprecated once a single topology pipeline is stable?

## Next Checkpoint

After implementing v1:

- run `afflec-demo` to a timestamped output dir,
- generate undersuit layers from that exact body,
- run ROM sampler on the same topology,
- solve seams on the same topology,
- optionally transfer seams to the other topology and render both (native + transferred) with explicit labels,
- produce a mesh-registry + vertex-map orbit bundle.

