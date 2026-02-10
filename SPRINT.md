# SeamInit Sprint Status

Source goals are the SeamInit × DASHI-ROM sprint phases in `CONTEXT.md` (e.g., sprint definitions at CONTEXT.md:350-671). This file tracks where we stand relative to those goals.

## Current Position

- Real ROM operator path is live: `sampler_real` streams finite-difference sensitivities over SMPL-X, enforces vertex-count invariants (10,475 → 9,438), remaps deterministically, and emits provenance hashes alongside seam costs consumed by `generate_undersuit`.
- Neutral-pose nearest-neighbour remapping is recorded with mapping statistics; seam costs now align to the afflec body mesh and carry samples-used metadata.
- NaN/empty seam handling is in place: zero-fill when no mapped vertices/edges are found, emit warnings, and persist per-panel diagnostics.
- Vertex mapping policy is explicit and configurable (`--vertex-map nearest|error`, `--max-map-distance <m>`); mapping stats are persisted in provenance JSON.
- Sprint 2 is closed; focus shifts to seam solvers (Sprint 4 / Sprint S1) with ROM-derived seam costs as inputs.
- Sprint R (ROM formalisation) is captured in `docs/rom_levels_spec.md`; the initial sweep schedule lives at `data/rom/sweep_schedule.yaml` and will drive L0/L1/L2 sampling once wired, plus an L3 completeness certificate.
- Seam solvers (MST, shortest path, mincut, PDA) currently pass all tests (`pytest tests/seams -q`); no failing evidence reproduced in this run.

## Sprint Tracker

- **Sprint 0 — Alignment & Contracts** (CONTEXT.md:352-391) — *In progress*. Constraint registry skeletons and schema drafts are added (`data/constraints/`, `schemas/basis_readme.md`, `schemas/rom_samples.yaml`); the object map and coupling registry still need to be formalized and populated.
- **Sprint 1 — Kernel–Body Projection** (CONTEXT.md:394-425) — *Mostly complete*. Basis loader/projector and canonical generator (`scripts/generate_canonical_basis.py`) are validated; assets remain generated-not-versioned and the validation notebook still needs real payloads.
- **Sprint 2 — ROM Aggregation & Field Statistics** (CONTEXT.md:428-462) — *Complete*. ROM sampler streams real pose-derived costs with provenance hashes and vertex-count invariance; NaN-safe seam aggregation and vertex-map policy flags are landed.
- **Sprint 3 — Seam Cost Field & Constraints** (CONTEXT.md:465-500) — *In progress*. Seam costs map onto undersuit generation with metadata; constraint registry and visual diagnostics remain open.
- **Sprint 4 — Seam Solvers** (CONTEXT.md:503-540) — *Not started*. No shortest-path or min-cut seam solvers tied to the ROM cost field. Next: implement the two solvers and export labeled panels.
- **Sprint 5 — Seam Feedback Operator** (CONTEXT.md:544-575) — *Not started*. No seam operator or iterative stability loop in place. Next: add the operator, rerun costs, and track drift metrics.
- **Sprint 6 — Seam Couplings & PDA Integration** (CONTEXT.md:579-608) — *Not started*. No seam obligations/couplings in the PDA gate. Next: formalize seam cocycles and extend the acceptance gate with blocking logs.
- **Sprint 7 — Optimization & Regime Selection** (CONTEXT.md:612-648) — *Not started*. No multi-candidate optimizer or Pareto front surfaced. Next: design the objective and runner, then expose recommendations.
- **Sprint 8 — Productization & Handoff** (CONTEXT.md:651-669) — *Not started*. No CLI/API for the ROM-aware seam stack. Next: package the pipeline with CAD-friendly exports, QA tools, and docs.

## Suggested Immediate Steps

1. Start Sprint S2 with diagnostics: implement visual overlays (ROM heatmap + seam paths) and numeric reports (kernel/MDL breakdown, stability).
2. Add solver variants (shortest-path or min-cut) reusing the kernel+MDL cost; keep MST as baseline for comparison.
3. Populate constraint manifests with real vertex/edge ids and formalize the coupling/object map so gates can use shared ids.
4. Refresh the canonical basis from a production mesh and exercise the validation notebook with real payloads (artifacts stay in `outputs/rom/`).

## Sprint R2 — Real ROM Operator & Seam Cost Integration (Closed)

**Objective**: Implement a real, pose-derived ROM operator using SMPL-X, integrate it end-to-end into seam cost generation and undersuit synthesis with strict dimensional invariants and provenance.

**Delivered**

- ROM operator as a pushforward over admissible pose space with finite-difference Jacobians; streaming contraction avoids materializing full tensors.
- Deterministic real ROM sampler with pose sweep, finite-difference sensitivity accumulation, and seam cost outputs aligned to target body meshes.
- Vertex alignment guarantees with neutral-pose nearest-neighbour remapping, mapping statistics in provenance, configurable policy (`nearest` vs `error`) and distance thresholds.
- NaN-safe seam aggregation: zero-fill for empty seams/panels, warnings emitted, per-panel diagnostics recorded.
- Provenance: hashes for body, weights, pose sweep, call counts, FD step size, synthetic vs real sampler markers.
- Tests and documentation: pytest coverage for sampler and seam costs; CLI flags/behaviours documented; ROM formalism recorded.

**Outcome**: Seam costs are computed from a real ROM operator derived from admissible motion with dimensional consistency and auditable provenance.

**Status**: ✅ Complete.

## Sprint R — ROM Formalisation, Scheduling & Completeness (Planned)

**Intent**: Turn ROM from an implicit side effect of the sampler into an explicit, schedulable, auditable operator with levels, artifacts, and stop criteria — without changing seam solvers or fabric logic.

### ROM Levels

- **L0**: per-joint marginal envelopes; single-DOF sweeps with neutral pose elsewhere.
- **L1**: curated joint-pair sweeps (no full `J×J` tensor).
- **L2**: task-conditioned ROM from procedural controllers (MoCap augments later).
- **L3**: completeness certificate when envelopes, seam ranks, and MDL mass stabilize.

### Minimal Practical Schedule

Artifacts: `data/rom/sweep_schedule.yaml`, `data/rom/task_profiles/*.yaml`.

- L0: ~100–150 poses (e.g., shoulder/hip 7 steps, elbow 5).
- L1: curated pairs only, ~75–100 poses.
- L2: controller-based trajectories (reach/squat/twist).

### Deliverables

- Code: `smii/rom/pose_schedule.py`, `smii/rom/completeness.py`, `sampler_real --schedule`.
- Outputs: `outputs/rom/rom_samples_L0.json`, `rom_samples_L1.json`, `rom_samples_L2.json`, `outputs/rom/rom_L3_certificate.json`.

### Completeness Metrics

- Envelope convergence (99% below ε).
- Seam cost rank stability (Spearman > 0.98).
- MDL mass saturation (incremental MDL contribution trends to zero).

### Out of Scope (Sprint R)

- No ML/learned priors.
- No full MoCap ingestion.
- No fabric or seam solver changes.

## Sprint S1 — ROM-Driven Seam Optimization (Spec)

**Theme**: Turn ROM seam cost fields into concrete, optimal seam placements. Duration: 1–2 weeks. Depends on Sprint R2 (frozen: ROM sampler, mapping policy, NaN handling).

**Objective**: Given a body mesh, seam graph, and ROM-derived seam cost field, compute optimal seam placements that minimize ROM stress while respecting garment topology and manufacturing constraints.

**Inputs (available)**: `SeamCostField` (vertex_costs, optional edge_costs, metadata including samples_used, mapping_info, empty flags); seam graph (vertex↔seam memberships, seam edges, panel membership); constraints (forbidden regions, symmetry pairs, panel limits).

**Formal problem**: Select seams (S⊆E) minimizing
\[
\mathcal C(S)=\sum_{v\in S} c_v + \alpha\sum_{e\in S} c_e + \beta\,\mathcal P_{\text{fabric}}(S) + \gamma\,\mathcal P_{\text{topology}}(S)
\]
subject to seam continuity, panel validity, symmetry constraints, and forbidden-region exclusion. Deterministic optimization only (no learning).

**Work Breakdown**

- **S1.1 Edge cost construction** (Geometry/ROM): define edge costs from vertex costs (mean, max, length-weighted integral), respect empty_vertices/empty_edges, add unit tests. Deliverable: `edge_costs = build_edge_costs(cost_field, seam_graph, mode="mean")`.
- **S1.2 Baseline seam solver** (Algorithms): implement one deterministic solver (shortest path or min-cut or DP on seam graph). Deliverable: `solve_seams(seam_graph, vertex_costs, edge_costs, constraints) -> SeamSolution`.
- **S1.3 Constraint integration** (Systems): enforce forbidden regions (hard fail), symmetry (soft penalty/mirroring), panel continuity; emit reasoned errors on conflicts. Deliverable: constraint-aware solver wrapper with clear failure modes.
- **S1.4 Diagnostics & explainability** (Tooling): visual overlays (seam paths on body, ROM heatmap vs seam placement); cost breakdown (per-term contributions, top-N avoided high-cost regions). Deliverables: PNG/SVG overlays and JSON cost attribution report.

**Non-goals**: learning/ML, fabric physics simulation, multi-garment optimization, UI polish.

**Deliverables/Artifacts**: `smii/seams/solver.py`, `smii/seams/edge_costs.py`, `tests/seams/test_solver.py`, `examples/solve_seams_from_rom.py`; `solve_seams` API; JSON cost breakdown; visual debug assets.

**Definition of Done**: Deterministic seam solutions on identical inputs; ROM-driven seams reduce total ROM cost vs baseline; forbidden seams hard-error; empty panels warn visibly; outputs include seam list, cost breakdown, and visual debug artifacts.

**Sprint exit statement**: “Seams are now placed by optimizing against a real, motion-derived ROM operator under explicit garment constraints.”

## Sprint S3 — Fabric-Aware & Task-Weighted Seam Optimization

**Theme**: Move from ROM-safe seams to material-optimal seams for a task.

**Objective**: Co-optimise seam placement with fabric anisotropy and task-weighted ROM so seams are materially and functionally justified.

**Scope**

- Frozen: PDA controller, MDL structure, solver APIs, constraint semantics (bugfix-only).
- In scope: fabric tensor kernels, task-weighted ROM aggregation, fabric regime selection, material-aware diagnostics.

**Work Breakdown**

- **S3.1 Fabric kernel integration** (Geometry/Materials): add fabric-aware kernel terms (fabric stretch mismatch, shear penalty, curvature) to EdgeKernel; tests: zero penalty when grain aligned, monotonic under misalignment.
- **S3.2 Task-weighted ROM aggregation** (ROM/Systems): task profiles as weighted pose mixtures; `aggregate_rom(samples, task_profile) -> SeamCostField`; tests: different tasks change seam costs; deterministic per task.
- **S3.3 Fabric regime decision layer** (Architecture): discrete regimes (knit/woven/composite), grain orientation per panel, seam allowance class; PDA state includes `(seams, fabric assignments)` with manufacturability constraints and MDL modifiers.
- **S3.4 Diagnostics: “why this fabric here”** (Tooling): JSON rationale per panel/seam; overlays showing fabric regime + seam paths and avoided regions; stability/witness extended to fabric choices.
- **S3.5 End-to-end example** (Integration): task/fabric-aware CLI run (e.g., `solve_seams_from_rom --task reach --fabric knit --solver pda-mincut`) producing seams, fabric assignments, diagnostics, and ROM-only comparison.

**Acceptance Criteria**

1. Changing task profile changes seam placement meaningfully.
2. Changing fabric regime changes seam topology/routing.
3. ROM-only vs ROM+fabric solutions are comparable and explainable.
4. PDA converges without oscillating between regimes.

**Artifacts**

```
smii/seams/fabric_kernels.py
smii/seams/task_profiles.py
smii/seams/regime_solver.py
configs/tasks/*.yaml
configs/fabrics/*.yaml
examples/task_fabric_seam_demo.py
```

**Sprint exit statement**: “Seams are now placed as a function of motion, material, and task — not geometry alone.”

## Sprint S2 — ROM-Driven Seam Solvers (Production + Diagnostics)

**Theme**: Harden the PDA seam optimizer into a production-usable, inspectable solver. Status entering sprint: kernel/MDL/PDA stack present, deterministic solver behaviour with tests.

**Objective**: Deliver a ROM-driven seam solver that can run end-to-end, be inspected visually and numerically, compared against baselines, and trusted for downstream fabrication decisions.

**Scope**

- Frozen: ROM operator, kernel semantics, MDL formulation, PDA acceptance logic (bugfix-only).
- In scope: diagnostics, visualisation, solver variants, CLI/example drivers, weight configuration, comparative evaluation.

**Work Breakdown**

- **S2.1 Solver variants** (Algorithms): add at least one additional solver (shortest-path seam routing or min-cut/graph-cut) sharing the kernel+MDL objective. Deliverables: `solve_seams_shortest_path(...)` / `solve_seams_mincut(...)`; solutions comparable on common metrics.
- **S2.2 Diagnostics & explainability** (Tooling): visuals (body overlay with ROM heatmap, seam paths, panel boundaries, avoided high-cost regions); numeric reports (per-seam kernel + MDL breakdown, stability/witness results). Deliverables: PNG/SVG exports and JSON schema.
- **S2.3 Comparative evaluation** (Systems): run MST baseline, PDA-MST, PDA-SP/mincut; report total cost, seam length, panel count, max ROM cost intersected, perturbation stability. Deliverables: table/JSON summary and example script/notebook.
- **S2.4 CLI & example driver** (Integration): `examples/solve_seams_from_rom.py` running end-to-end with configurable solver/weights/prior and emitting seams + diagnostics. Example invocation:
  `python examples/solve_seams_from_rom.py --body afflec_body.npz --rom-costs seam_costs_afflec.npz --solver pda-mincut --weights configs/kernel_weights.yaml --mdl configs/mdl_prior.yaml --out outputs/seams_afflec/`
- **S2.5 Weight configuration & tuning hooks** (Architecture): externalise kernel weights and MDL priors via YAML schemas with defaults reproducing S1 behaviour (no learning).

**Acceptance Criteria**

1. One command yields seam solution + visuals + reports.
2. PDA solver beats MST baseline on ROM cost without exploding seam count.
3. Diagnostics explain why seams moved and which costs dominate.
4. ±ROM noise reruns preserve topology or flag instability.

**Artifacts**

```
examples/solve_seams_from_rom.py
smii/seams/solvers_sp.py
smii/seams/solvers_mincut.py
smii/seams/diagnostics.py
configs/kernel_weights.yaml
configs/mdl_prior.yaml
```

**Sprint exit statement**: “We can compute, explain, and compare ROM-driven seam layouts in a production-ready, reproducible way.”
