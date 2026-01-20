# SeamInit Sprint Status

Source goals are the SeamInit × DASHI-ROM sprint phases in `CONTEXT.md` (e.g., sprint definitions at CONTEXT.md:350-671). This file tracks where we stand relative to those goals.

## Current Position

- Undersuit panel regularization remains the active shipped path (ROADMAP.md:16-19), but Sprint 0/1/2 scaffolding now exists: constraint manifests in `data/constraints/`, schema stubs in `schemas/`, ROM helpers in `src/smii/rom/`, and a validated canonical-basis generator (`scripts/generate_canonical_basis.py`) that writes ignored assets to `outputs/rom/`.
- Sprint 3+ remains untouched: no seam costs, solvers, feedback operator, or PDA coupling beyond stubs, though seam cost interfaces now exist for wiring.

## Sprint Tracker

- **Sprint 0 — Alignment & Contracts** (CONTEXT.md:352-391) — *In progress*. Constraint registry skeletons and schema drafts are added (`data/constraints/`, `schemas/basis_readme.md`, `schemas/rom_samples.yaml`), but the object map and coupling registry still need to be formalized and populated.
- **Sprint 1 — Kernel–Body Projection** (CONTEXT.md:394-425) — *In progress*. Kernel basis loader/projector now exist (`smii.rom.basis`) with tests; canonical basis generation is validated locally (`outputs/rom/canonical_basis.npz` is ignored) but assets remain generated-not-versioned, and the validation notebook still needs real payloads.
- **Sprint 2 — ROM Aggregation & Field Statistics** (CONTEXT.md:428-462) — *In progress*. Aggregation handles required/optional/observed fields with gate-aware rejection summaries, per-edge statistics, and hotspot diagnostics (see `examples/rom_hotspot_diagnostic.py`); still need ROM sampler wiring and higher-fidelity visual overlays.
- **Sprint 3 — Seam Cost Field & Constraints** (CONTEXT.md:465-500) — *Not started*. No ROM-informed seam cost generator or constraint filter present. Next: define forbidden-edge registry and implement the per-edge cost computation/visualizer.
- **Sprint 4 — Seam Solvers** (CONTEXT.md:503-540) — *Not started*. There are no shortest-path or min-cut seam solvers tied to the ROM cost field. Next: implement the two solvers and export labeled panels.
- **Sprint 5 — Seam Feedback Operator** (CONTEXT.md:544-575) — *Not started*. No seam operator or iterative stability loop in place. Next: add the operator, rerun costs, and track drift metrics.
- **Sprint 6 — Seam Couplings & PDA Integration** (CONTEXT.md:579-608) — *Not started*. No seam obligations/couplings in the PDA gate. Next: formalize seam cocycles and extend the acceptance gate with blocking logs.
- **Sprint 7 — Optimization & Regime Selection** (CONTEXT.md:612-648) — *Not started*. No multi-candidate optimizer or Pareto front surfaced. Next: design the objective and runner, then expose recommendations.
- **Sprint 8 — Productization & Handoff** (CONTEXT.md:651-669) — *Not started*. No CLI/API for the ROM-aware seam stack. Next: package the pipeline with CAD-friendly exports, QA tools, and docs.

## Suggested Immediate Steps

1. Populate constraint manifests with real vertex/edge ids and formalize the coupling/object map so gates can use shared ids.
2. Refresh the canonical basis from a production mesh and exercise the validation notebook with real payloads (artifacts stay in `outputs/rom/`).
3. Wire ROM sampler input to `smii.rom.aggregation` and extend visual diagnostics beyond the demo script to satisfy Sprint 2 exit criteria.
