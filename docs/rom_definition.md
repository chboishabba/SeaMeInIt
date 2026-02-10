# Range of Motion (ROM) as a compressed operator

This note formalizes how ROM relates to SMPL-X and how the sampler represents
the compressed ROM object used by the aggregator and seam-cost pipeline.

## SMPL-X provides the coordinate system, not ROM itself

SMPL-X defines a differentiable mapping from shape and pose to a vertex set:

```
Phi(beta, theta, psi) -> V in R^{N x 3}
```

* `beta`: shape parameters
* `theta`: joint angles (pose)
* `psi`: facial / expression parameters

SMPL-X is instantaneous: it maps one pose to one mesh. ROM is about variation
over the admissible pose space, not a single evaluation of `Phi`.

## ROM as a pushforward measure over pose space

Let the admissible pose space be `Theta_adm ⊂ Θ` (joint limits, gates, etc.).
For each pose `theta ∈ Theta_adm`, define a per-vertex field `f_theta(v)` (e.g.,
strain or curvature proxy). ROM is the aggregated response across poses:

```
R(v) = ∫_{Theta_adm} f_theta(v) dμ(theta)
```

with `μ(theta)` a pose-space measure (uniform, task-weighted, etc.). If joint
structure is retained, ROM is a supertensor `R_{v, j1, j2, ...}` capturing
cross-joint coupling, which is too large to store explicitly.

## Compression via canonical basis and coefficients

To make ROM tractable, fields are projected onto a canonical basis `B ∈ R^{V x K}`
(orthonormal in this codebase):

```
f_theta(v) ≈ Σ_{k=1..K} c_k(theta) * B_{v,k}
```

ROM is then represented as the distribution of coefficient vectors
`c(theta) ∈ R^K` over admissible poses. This compressed object is what the ROM
sampler emits and what the aggregator consumes.

**Key invariant:** coefficient length `K` must match the basis component count
and the seam graph’s vertex set (via the basis vertex count).

## What the sampler actually does

A ROM sampler maps pose → coefficients:

```
theta -> c(theta)
```

Real samplers derive `theta` from motion/pose sources; procedural sweeps are a
deterministic quadrature over `Theta_adm` and are therefore a valid finite
approximation of the ROM integral (not random noise). Outputs are valid when
they:

* are deterministic functions of pose
* respect joint limits/gates
* use a basis sized to the target mesh

## Synthetic vs real samplers

* **Real sampler:** coefficients come from motion-derived poses; set
  `meta.synthetic=false` and capture provenance (`pose_source`, `basis`, `body`).
* **Synthetic sampler (plumbing):** use `scripts/generate_synthetic_rom_sampler.py`
  to emit deterministic coefficients sized to the afflec mesh. The payload is
  marked `meta.synthetic=true` and lives at `outputs/rom/afflec_sampler.json`
  (gitignored). Swap in a real sampler JSON at the same path to run the exact
  same pipeline without code changes.

## Usage invariants (seam-cost path)

1. Build basis on the target mesh with component count `K` matching the sampler.
2. Aggregate sampler → seam costs; lengths should equal the mesh vertex count.
3. Run undersuit with `--seam-costs` pointing to the aggregated NPZ.

Violating K/vertex alignment should fail loudly rather than broadcast/truncate.

## Canonical ROM sources in this repo

You can plug three ROM sources into the same pipeline; swapping them never
changes solver logic.

1) **Real ROM (SMPL-X finite differences)** — source of truth. Uses
`smii.rom.sampler_real` to evaluate SMPL-X, perturb joints, stream FD Jacobians,
and emit coefficient samples with provenance hashes. See `docs/rom_real_sampler.md`
for the full CLI; a typical call is:
`python -m smii.rom.sampler_real --body outputs/afflec_demo/afflec_body.npz --poses data/rom/afflec_sweep.json --weights data/rom/joint_weights.json --fd-step 1e-3 --out-costs outputs/rom/seam_costs_afflec.npz`.

2) **Task-weighted ROM aggregation** — same samples, different measure. Task
profiles (`configs/tasks/*.yaml`) define μ_task; use
`smii.seams.task_profiles.aggregate_rom_for_task(samples, task_profile, basis, aggregation="mean")`
to reweight existing ROM samples without recomputing physics. This is the Sprint
S3 “ROM for what?” feature.

3) **Synthetic/procedural ROM** — plumbing-only fallback. Deterministically
generates coefficient vectors sized to the mesh via
`scripts/generate_synthetic_rom_sampler.py ... --components <K> --samples <N>`
and marks `meta.synthetic=true`. Use only for CI/debugging until real sampler
outputs exist.

### ROM flow (conceptual)

```
SMPL-X poses → sampler_real → ROM samples (coeffs)
           → aggregate_rom_for_task (task-weighted measure)
           → SeamCostField → seam solvers (kernel/PDA/MDL)
```

## ROM is computed, not premade: progressive admissibility levels

SMPL/SMPL-X supplies pose coordinates and geometry, not ROM sweeps. Admissible
structure is constructed in layers:

- **Level 0 — single-joint admissibility.** Sweep each joint independently;
  measure displacement/strain/Jacobian norms to identify limits and sensitivities
  (SMPL does *not* ship trustworthy limits).
- **Level 1 — joint pairs.** 2D sweeps to capture non-additive coupling (e.g.,
  shoulder–elbow, hip–knee, spine–hip) as a pairwise ROM kernel; no library
  provides this.
- **Level 2 — latent coupling.** Compress pairwise responses into a latent basis,
  keep active coupling modes, discard additive directions. This is where the
  canonical basis + coefficients architecture fits.
- **Level 3 — higher-order couplings.** Explore only combinations that survive
  lower levels; build a hierarchical admissibility lattice (PDA/witness stability
  prunes most combinations) instead of brute forcing (J^n).

## Empirical datasets: clip and weight ROM, not define it

MoCap datasets (AMASS/CMU/H36M, contact sports, dance/acrobatics, clinical ROM)
are streaming inputs to validate and weight procedural ROM:

1. Procedural sweep defines the hypothesis space (Levels 0–3).
2. Stream MoCap poses → project to ROM coefficients → update running stats
   (min/max, quantiles, coarse coupling occupancy) without storing all poses.
3. Optionally probe boundary poses with the FD kernel to flag mechanically
   hostile regimes.
4. Export envelopes/density per ROM dimension and coupling; label rare vs common
   vs contact-only regimes (“illegality certificates” when outside procedural ROM
   or causing blowups).

## Dynamics/impact belong in MDL (beyond static ROM)

ROM is static; garments fail under dynamics. Add MDL terms for:

- vertex/region velocity and acceleration magnitudes,
- effective inertia (downstream mass × lever arm),
- impact tolerance per region (injury/pain curves; automotive/sports med sources),
- optional contact probability (from contact-sport clips).

A practical MDL term: `mdl_impact(v) = log(1 + D(v)/tau(v))` where `D(v)` is
peak acceleration × inertia from sampled motion and `tau(v)` is tolerable
impulse/acceleration for the region. This steers seams/fabric away from
high-ROM × high-accel × low-tolerance zones while keeping ROM kernels unchanged.

## What SMPL / SMPL-X does **not** provide

- No premade ROM dataset or sweep; you must define admissible poses and gates.
- No joint-limit registry you should trust; limits and coupling semantics live
  in this codebase (see `data/rom/` / `configs/tasks/`).
- Pose priors (e.g., VPoser) encode likelihood, not admissibility or task
  relevance; they are optional inputs to pose generation, not ROM themselves.
- No precomputed joint-pair or higher-order coupling maps; those emerge from the
  finite-difference sampler and task-weighted aggregation above.
