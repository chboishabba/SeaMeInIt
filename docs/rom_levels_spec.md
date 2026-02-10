# Sprint R — ROM Levels, Sweep Schedule, Bootstrapping, Completeness

This spec defines ROM levels (L0–L3), a minimal sweep schedule, bootstrapping with MoCap + procedural sweeps, and stop rules for completeness. Output targets seam costs, kernel/MDL/PDA seam optimization, and fabric regime assignment.

## Goal and Outputs

Turn ROM from an implicit side effect of the sampler into an explicit, schedulable, auditable operator with levels, artifacts, and stop criteria. This sprint does not add seam solvers or fabric logic.

Produce a repeatable ROM operator with artifacts:

- pose schedule
- ROM samples (L0/L1/L2)
- ROM envelopes/certificates (L0/L3)
- completeness report

## 1) ROM Level Definitions

ROM operator: `R : Θ -> R^K`, `theta -> c(theta)` with canonical basis `B ∈ R^{V×K}` and admissible pose space `Θ`.

- **L0 — Per-joint marginal ROM.** Single-joint variation, others neutral. Outputs min/max and per-vertex extrema of ROM response. Artifact: `rom_L0_envelope.json`.
- **L1 — Pairwise coupled ROM.** Joint pairs sampled on curated 2D grids. Outputs conditional envelopes and coefficient-space summaries (no `J×J` tensor materialization). Artifact: `rom_samples_L1.json`.
- **L2 — Task-conditioned ROM.** Task distributions from procedural controllers (MoCap is augmentation only). Artifact: `rom_samples_L2.json` with task profile weights.
- **L3 — Adaptive ROM completeness.** ROM is complete when additional samples do not change envelopes, seam cost ranking, or MDL mass distribution. Artifact: `rom_L3_certificate.json`.

**Note on L1 additions (chain-aware admissibility):** See `docs/sprint_rom_l1.md` for the next sprint’s scope. L1 will introduce collision/legality fields and paired chain sweeps so hips/pelvis/shoulders start affecting admissibility stacks; it is still synthetic (no MoCap) and focused on early hypervoxel emergence.

## 2) Minimal Practical Sweep Schedule

### L0 sweep (mandatory, cheap)

Use a light schedule to avoid combinatorial explosion. Example:

```yaml
level: L0
joints:
  shoulder:
    axes: [flexion, abduction, rotation]
    steps: 7
  elbow:
    axes: [flexion]
    steps: 5
  hip:
    axes: [flexion, abduction, rotation]
    steps: 7
```

Expected cost: ~100–150 poses.

### L1 sweep (curated joint pairs only)

```yaml
level: L1
pairs:
  - [shoulder, elbow]
  - [hip, knee]
  - [spine_twist, shoulder]
steps: 5
```

Expected cost: ~75–100 poses. Only anatomically meaningful pairs.

### L2 sweep (procedural controllers)

```yaml
level: L2
tasks:
  overhead_reach:
    controller: reach
    samples: 25
  squat:
    controller: squat
    samples: 20
  twist_reach:
    controller: twist_reach
    samples: 20
```

Controllers emit pose trajectories (not grids).

### CLI hook (sampler)

Use the schedule directly with the real sampler:

```bash
PYTHONPATH=src python -m smii.rom.sampler_real \
  --body outputs/afflec_demo/afflec_body.npz \
  --weights data/rom/joint_weights.json \
  --schedule data/rom/sweep_schedule.yaml \
  --filter-illegal \
  --out-envelope outputs/rom/afflec_envelope.json \
  --out-certificate outputs/rom/afflec_rom_certificate.json \
  --out-costs outputs/rom/seam_costs_afflec.npz \
  --out-meta outputs/rom/afflec_rom_run.json
```

## 3) Bootstrapping Higher-Order ROM (procedural → MoCap)

### Phase A — Procedural only (this sprint)

- Run L0/L1/L2 sweeps.
- Produce:
  - `rom_samples_L0.json`
  - `rom_samples_L1.json`
  - `rom_samples_L2.json`
- Aggregate → seam costs → diagnostics.

### Phase B — MoCap augmentation (next sprint)

MoCap does not define ROM limits. It only augments density:

1. Load MoCap poses and project to SMPL-X θ.
2. Reject illegal poses (joint limit violations).
3. Treat remaining poses as density evidence for L2 → L3 (frequency + extremity).

## 4) ROM Completeness Metrics (stop rules)

Declare completion when all hold for ≥2 consecutive rounds:

- **Envelope convergence:** 99% of vertices change below ε.
- **Seam cost rank stability:** Spearman correlation > 0.98 across rounds.
- **MDL mass saturation:** incremental MDL contribution per vertex trends to zero.

## Sprint R Deliverables

- Spec: `docs/rom_levels_spec.md` (this file, pinned).
- Configs: `data/rom/sweep_schedule.yaml`; `data/rom/task_profiles/*.yaml`.
- Samples: `outputs/rom/rom_samples_L0.json`, `rom_samples_L1.json`, `rom_samples_L2.json`.
- Certificate: `outputs/rom/rom_L3_certificate.json` (counts, envelope deltas, rank correlations, stop reason).

## Acceptance Tests

- **Spec compliance:** schedule YAML validates against schema; required outputs materialize.
- **Reproducibility:** fixed seed → identical pose IDs and coefficient hashes.
- **Completeness logic:** on a fixed synthetic dataset, completeness triggers in ≤N rounds.
- **Downstream stability:** seam cost rank correlation improves monotonically across rounds.

## Next options

- Add YAML schemas for `sweep_schedule.yaml` and `task_profile.yaml` aligned with existing config style (`kernel_weights.yaml`, `mdl_prior.yaml`).
- Wire the MoCap streaming “ROM certificate” generator to emit envelopes/density and boundary tags.
