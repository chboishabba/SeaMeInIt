# Sprint ROM-L1 — Chain-Aware Admissibility & Hypervoxel Emergence

This sprint specification captures the next narrow step after L0: introduce chain-aware admissibility so the PDA can begin collapsing pose space into hypervoxels. It is deliberately limited (no MoCap, no learning, no final seams) and aims to create the first non-trivial compression signal.

## Sprint name

**ROM-L1: Chain-aware admissibility fields (collision + coupling)**

## Motivation (formal)

At L0, ROM sampling measures joint-local displacement density; admissibility stacks are trivial, so compression is minimal. L1 adds short-range dependency closure (kinematic chains + legality), enabling:

- multiple poses → one admissibility state,
- emergence of hypervoxels,
- first meaningful seam incentives.

## Scope (strict)

In scope:
- Collision/legality field that turns on admissibility boundaries.
- Chain-aware sweep schedules (paired joints/short chains).
- Chain-sensitive displacement proxy (coupled motion costs more than isolated).
- Wire these fields into seam costs for diagnostics only.

Out of scope:
- MoCap or learned priors.
- Global posture realism or balance.
- Final seam topology decisions (belongs to L2/L3).

## Work plan (what changes)

1) **Collision / legality field (minimal, deterministic)**
- Joint-centered spheres or capsules.
- Non-adjacent joint–joint and limb–torso checks.
- Scalar legality score per pose (penetration depth/count) with reject/weight option.
- Persist rejection reasons in sampler metadata; deterministic across seeds/schedules.

2) **Chain-aware sweep schedules (L1)**
- Add structured pairs/chain sweeps: hip+knee, pelvis+spine, shoulder+elbow, optional bilateral asymmetry.
- Small grids (e.g., 5×5 or Latin hypercube) mirrored L/R.
- Deterministic expansion; no random walks.

3) **Chain-sensitive proxy (displacement is not enough)**
- Add one: displacement gradient along bones, chain amplification factor, or root-distance weighting so coupled motion costs more than isolated motion.
- Ensure hip+spine or shoulder+elbow excites more field mass than single-joint motion.

4) **Integrate into seam cost (diagnostic)**
- Include legality/chain fields in seam cost accumulation.
- Expect rigid cranial regions to stop dominating; hips/shoulders/pelvis should register.
- Seam rankings may improve but are not final; this sprint is about admissibility signals.

## Compression expectations

- Fewer distinct admissibility stacks than poses.
- Repeated collision patterns across many poses.
- Early hypervoxel formation (same constraint stack for many samples).
- Seam incentives persist across pose subsets (local rank stability).

## Deliverables

Code:
- `pose_legality.py` (or equivalent) with spheres/capsules and scoring.
- L1 schedule definitions (YAML) with chain sweeps.
- Sampler metadata extended with legality fields.

Diagnostics:
- Updated overlays showing new energy around hips/shoulders/pelvis.
- CSV/JSON: rejection rates, common collision pairs.

Docs:
- Short addendum to `docs/rom_levels_spec.md` explaining what L1 adds over L0.

## Done-when (hard gates)

1. **Admissibility boundaries exist**: poses are rejected/down-weighted; rejection reasons repeat.
2. **Chain joints light up**: hips/pelvis/shoulders show increased field mass (not just hands).
3. **Compression begins**: unique admissibility stacks ≪ samples; adding L1 samples yields few new stacks.
4. **Seam rankings stabilize locally**: rank correlation vs L0 improves; global stability not required.

## Guardrails (non-goals)

- No “realistic posture” claims.
- No MoCap ingestion.
- No final seam count decisions.
- No global balance/task semantics.

## Pre-spec for Sprint L2 (task manifolds → hypersheets)

Objective: sample deterministic task manifolds (procedural trajectories) to test which admissibility boundaries persist across coherent motion families and force PDA stack compression across trajectories (hypersheets). No MoCap or learning; focus on task-coherent sampling, stack trajectory logging, and seam rank stability across tasks.

Done-when for L2 (preview):
- Same boundaries appear across ≥2 tasks.
- Seam rank correlation ≥0.95 across task manifolds.
- Unique stacks grow slowly; adding task samples yields few new stacks.
- Boundaries/seam loci sharpen around hips/shoulders/side torso.
