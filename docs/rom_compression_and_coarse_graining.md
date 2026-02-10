# ROM Compression, Coarse-Graining, and Fine-Graining (PDA/MDL framing)

This note captures the compression and refinement story that drives ROM → seam inference. It complements `docs/rom_levels_spec.md` (schedules, certificates) by explaining *why* ROM sampling compresses and *when* we refine.

## Scope

- Formalise compression in this repo’s PDA/MDL language.
- Define hypervoxels/hypersheets (coarse states) and how seams relate.
- Clarify when fine-graining is triggered and why it does not lose accuracy.
- Align compression with ROM levels (L0–L3) and sampler outputs (`--out-envelope`, `--out-certificate`, `--rom-meta`).

## Core definitions

- **Admissibility:** A pose’s constraint stack (collisions, limits, strain gates). Two poses are equivalent if they activate the same stack.
- **Hypervoxel:** A region of pose space whose poses share the *same* admissibility stack. Inside one hypervoxel, all ROM/seam decisions are identical.
- **Hypersheet:** A continuous family of hypervoxels with the same constraint transitions. Hypersheets align with seam loci and panel boundaries.
- **Compression (MDL sense):** Replace enumerated poses with a small set of structural explanations (hypervoxels/sheets) that keep admissibility decisions unchanged.
- **Fine-graining:** Splitting a hypervoxel *only* when an admissibility boundary is approached (constraint stack ambiguity).

## Why compression is correct (not lossy)

Coarse states are *exact equivalence classes*, not averages. Within a hypervoxel, every pose would yield the same seam/constraint outcome, so no information relevant to seams is lost. Compression ends when adding structure no longer changes:

- ROM envelopes/extrema,
- seam cost rank ordering,
- MDL mass distribution (certificate deltas go to zero).

## When fine-graining happens

Fine-graining is conditional:

- Trigger: approaching an admissibility boundary where the PDA stack could change (new collision, limit activation, strain gate, legality rejection).
- Effect: split the hypervoxel locally; do **not** increase resolution globally.
- Stop: once the stack is stable again, refinement halts. Interiors stay coarse and exact.

This avoids the “refine everywhere” trap; only boundaries (hypersheets) receive more detail.

## Compression across ROM levels

| Level | What varies | What compresses | Expected outcome |
| --- | --- | --- | --- |
| L0 | Single joints (axis-local) | Basis sanity; almost no hypervoxels | Flex heatmaps reflect joint density; seams have little incentive |
| L1 | Joint pairs / short chains | First real hypervoxels (chain strain, collisions) | Hips/shoulders/pelvis start to “light up”; seam incentives emerge |
| L2 | Task manifolds (procedural + MoCap weighting) | Stable hypersheets across realistic motion | Seam placement begins to stabilise across tasks |
| L3 | Residual-driven closure | No new hypersheets; envelopes/cost ranks converge | ROM certificate says “complete”; seams become invariant |

Sampler hooks:

- `--schedule … --filter-illegal` produces levelled samples + legality metadata.
- `--out-envelope` (L0 extrema), `--out-certificate` (L3 completeness, rank deltas).
- `--out-meta/--rom-meta` carries per-pose metadata used by diagnostics (e.g., flex heatmaps).

## Interpreting current overlays/heatmaps

- Flex heatmaps today visualize a **joint-density / displacement proxy** from L0 sweeps. Rigid regions (skull) look “good” because motion is low; distal hands are hot because DOF density is high. This is expected at L0 and is *not* a seam recommendation.
- To surface hips/pelvis/knees, add fields that activate admissibility boundaries (collision/overlap, chain coupling, strain gates). Those create hypersheets that seams must respect.

## Practical next fields to add (minimal, no MoCap)

1. **Collision/legality energy:** joint spheres/capsules; reject or down-weight samples with penetration. Immediately forces hips/torso/shoulder regions into the decision surface.
2. **Chain amplification:** pair/chain sweeps (hip+knee, pelvis+spine, shoulder+elbow) so displacement/strain is measured under coupled motion.
3. **Strain/gradient proxy:** replace pure displacement with gradient/strain energy to penalise high curvature change more than rigid translation.

These three turn the current displacement proxy into admissibility-aware fields that the PDA can compress into stable seam hypersheets.

## Operational checkpoints (tie to certificates)

- **Envelope stability:** Few/no extrema updates between rounds (`rom_L0_envelope` + `rom_L3_certificate` deltas).
- **Seam rank stability:** Spearman ≥ 0.98 between successive seam cost vectors.
- **Rejection stability:** Legality/collision rejection rate stops changing; no new failure modes reported.
- **MDL plateau:** Adding samples no longer reduces description length (panel/seam count stops changing).

When all hold for ≥2 rounds, declare ROM complete for the schedule/task mix.

## References in this repo

- Schedules and stop rules: `docs/rom_levels_spec.md`
- Sampler CLI: `src/smii/rom/sampler_real.py`
- Legality/collision scoring: `src/smii/rom/pose_legality.py`
- Flex heatmaps (diagnostic only): `smii/rom/flex_heatmap.py` and `examples/solve_seams_from_rom.py`
