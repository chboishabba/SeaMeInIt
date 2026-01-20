# TODO

- Extend auto-split strategies (multi-cut, seam-aware) and propagate child-specific issues.
- Expose outline cleanup parameters (outlier threshold, smoothing iterations, simplify tolerance)
  in the pattern export CLI so garment makers can tune output fidelity.
- Wire PDA coupling manifest + gate decisions into ROM sampler/export paths with rejection logging (aggregation now emits structured gate ids/reasons).
- Expand ROM aggregation diagnostics (visuals, hotspot overlays) and connect to seam validators; demo stub lives at `examples/rom_hotspot_diagnostic.py`.
- Regenerate canonical ROM basis via `python scripts/generate_canonical_basis.py --vertices <path> --components 64`
  (do not commit the resulting NPZ; keep outputs/rom/ ignored); smoke-tested locally with sample vertices.
