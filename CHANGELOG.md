## Unreleased

- Updated ROM formalisation docs (Sprint R levels/schedule/completeness), sprint status, and compact context snapshot; aligned TODOs to new deliverables and operational checks.
- Added R6-lite spline fitting with curvature-bound fallback and split gating in boundary regularization.
- Added opt-in auto-split (`--auto-split`) and split helpers, plus structured issue metadata in patterns output.
- Expanded SVG annotations with severity styling, legend, and annotation levels.
- Added stress and spline tests for boundary regularization; fixed seam loop indexing for canine axis tests.
- Made undersuit pipeline test mesh watertight to satisfy generator requirements.
- Fixed suit_hard attachment validation import for test collection.
- Added seam length reconciliation with `SEAM_MISMATCH` issues based on seam partner metadata.
- Propagated seam-aware split metadata (`seam_avoid_ranges`, `seam_midpoint_index`) into exporter panels.
- Captured the missing afflec regression warning explicitly in tests.
- Documented TODO hygiene guidance link from README to `CONTEXT.md`.
- Added seam partner metadata normalization to derive seam-aware split ranges and documented the schema.
- Added multi-page PDF tiling with selectable page sizes for pattern exports.
- Added panel validation gate aggregation to surface ok/warning/error status in pattern metadata.
- Added seam cost NaN-safe aggregation (zero-fill with warnings) and vertex mapping policy flags (`--vertex-map`, `--max-map-distance`) to the ROM sampler provenance path.
- Added ROM-driven seam optimization scaffolding: edge cost derivation modes, deterministic MST seam solver with constraint enforcement, PDA kernel/MDL stack, and regression tests.
- Added solver variants (shortest-path, min-cut) sharing kernel+MDL objective, diagnostics/report scaffold, and example CLI `examples/solve_seams_from_rom.py` for end-to-end runs.
- Added default kernel/MDL config YAMLs, richer diagnostics overlay highlighting high-cost seams, and comparison script `examples/compare_seam_solvers.py` for benchmarking MST/PDA/SP/mincut.
