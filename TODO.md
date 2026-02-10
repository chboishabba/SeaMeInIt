# TODO

- Extend auto-split strategies (multi-cut, seam-aware) and propagate child-specific issues.
- Expose outline cleanup parameters (outlier threshold, smoothing iterations, simplify tolerance)
  in the pattern export CLI so garment makers can tune output fidelity.
- Wire PDA coupling manifest + gate decisions into ROM sampler/export paths with rejection logging (aggregation now emits structured gate ids/reasons).
- Expand ROM aggregation diagnostics (visuals, hotspot overlays) and connect to seam validators; demo stubs live at `examples/rom_hotspot_diagnostic.py` and `examples/rom_aggregate_from_samples.py`.
- Add output no-op verification for ROM/heatmap runs: log mtimes + content hashes for `afflec_body.npz`, `seam_costs`, heatmaps, and fitted params; warn when outputs are unchanged.
- Replace PGM-only measurement fixtures with explicit `tests/fixtures/afflec/measurements.yaml` ingestion (keep PGM support behind deprecation if needed).
- Add runtime performance attribution: detect GPU vs CPU heavy compute paths and flag when a claimed GPU-assisted run is actually CPU-bound (log + metrics).
- Regenerate canonical ROM basis via `python scripts/generate_canonical_basis.py --vertices <production mesh npy/npz> --components <K> --harmonics 5 --output outputs/rom/canonical_basis.npz`
  (do not commit the resulting NPZ; keep outputs/rom/ ignored), then run the sampler aggregator with real payloads:
  `PYTHONPATH=src python examples/rom_aggregate_from_samples.py --samples outputs/rom/afflec_sampler.json --basis outputs/rom/canonical_basis.npz --save-costs outputs/rom/seam_costs.npz`
  and pass `--seam-costs outputs/rom/seam_costs.npz` into `generate_undersuit` to annotate seams. When no real sampler is available, generate a plumbing-only one via `scripts/generate_synthetic_rom_sampler.py --body outputs/afflec_demo/afflec_body.npz --components <K> --samples 8 --out outputs/rom/afflec_sampler.json` (meta.synthetic=true) and swap in a real sampler at the same path when ready.
- Standardize per-body ROM dataset naming/versioning and add a caching policy/CLI so `sampler_real` outputs can be reused without reruns; document the refresh path alongside the cache location.
- Add a streaming MoCap ROM envelope pass (AMASS/contact/dance/clinical) that projects poses to ROM coefficients, emits per-dimension envelopes/density, tags rare/contact-only regimes, and optionally probes boundary poses with the FD kernel to flag mechanically hostile regions; keep outputs as JSON “ROM certificate” artifacts.
- Extend MDL with dynamic terms (velocity/acceleration/inertia, impact tolerance atlas, optional contact probability) and surface them in seam/fabric decisions without altering ROM kernels; document the injury/pain data sources used.
- Implement Sprint R spec (`docs/rom_levels_spec.md`): wire `data/rom/sweep_schedule.yaml` + `data/rom/task_profiles/*.yaml`, add `smii/rom/pose_schedule.py` + `smii/rom/completeness.py`, extend `sampler_real --schedule`, generate L0/L1/L2 sample artifacts, and emit `outputs/rom/rom_L3_certificate.json` with envelope deltas + rank correlations; seed acceptance tests for reproducibility and downstream stability checks.
- Execute Sprint ROM-L1 (`docs/sprint_rom_l1.md`): add legality/collision scoring, chain-aware sweep schedules, a chain-sensitive displacement proxy, and integrate these fields into seam cost diagnostics; publish L1 addendum in `docs/rom_levels_spec.md`.
- Sprint S1 (ROM-driven seam optimization):
  - Edge cost construction from vertex costs (mean/max/length-weighted integral) with unit tests — landed in `smii/seams/edge_costs.py`.
  - Deterministic seam solver (MST baseline) with `SeamSolution` API in `smii/seams/solver.py`; extend with shortest-path/min-cut variants as needed.
  - Constraint integration baseline (forbidden regions hard-fail, symmetry penalty, panel connectivity warnings) shipped; refine policies and error surfaces.
  - Diagnostics and explainability: overlays (PNG/SVG) of seams vs ROM heatmaps, JSON cost attribution (per-term and top-N avoided high-cost regions), example driver `examples/solve_seams_from_rom.py`.
  - PDA seam optimizer stack (kernels, MDL prior, moves, PDA controller) shipped in `smii/seams/{kernels,mdl,moves,pda}.py`; next: tune weights, add visual/debug outputs, and wire CLI/driver.
- Sprint S2 (production + diagnostics):
  - Add solver variants (`solve_seams_shortest_path`, `solve_seams_mincut`) reusing kernel+MDL objective and compare to MST baseline — implemented; extend benchmarks.
  - Ship diagnostics: ROM heatmap + seam overlays, avoided high-cost region highlights, per-seam kernel/MDL breakdown, stability/witness report; output PNG/SVG + JSON — scaffold added (`smii/seams/diagnostics.py` with threshold highlighting), still need richer visuals/witness.
  - Comparative evaluation script/notebook covering MST, PDA-MST, PDA-SP/mincut with metrics (total cost, seam length, panel count, max ROM cost intersected, perturbation stability) — initial script `examples/compare_seam_solvers.py` added; add notebook + perturbation metrics.
  - CLI/example driver `examples/solve_seams_from_rom.py` with config-driven weights/MDL (`configs/kernel_weights.yaml`, `configs/mdl_prior.yaml`) and solver selection emitting seams + diagnostics — defaults added; still need sample inputs/assets.
- Sprint S3 (fabric-aware, task-weighted):
  - Fabric kernels: incorporate stretch/shear mismatch and grain alignment into `EdgeKernel`; add fabric YAML loader (`configs/fabrics/*.yaml`) and penalties/tests.
  - Task profiles: load task mixtures (`configs/tasks/*.yaml`) and feed task-weighted ROM aggregation (`aggregate_rom(samples, task_profile)`).
  - Regime layer: extend PDA state with fabric assignments and grain rotations; add moves (`switch_fabric`, `rotate_grain`), manufacturability/MDL modifiers.
  - Diagnostics: per-panel rationale (fabric vs ROM), overlays showing fabric regimes + seams, stability under ROM and grain jitter; add task/fabric-aware example driver `examples/task_fabric_seam_demo.py`.
