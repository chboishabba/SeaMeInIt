## Unreleased

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
