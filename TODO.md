# TODO

- Extend auto-split strategies (multi-cut, seam-aware) and propagate child-specific issues.
- Expose outline cleanup parameters (outlier threshold, smoothing iterations, simplify tolerance)
  in the pattern export CLI so garment makers can tune output fidelity.
- Wire PDA coupling manifest + gate decisions into ROM sampler/export paths with rejection logging (aggregation now emits structured gate ids/reasons).
- Expand ROM aggregation diagnostics (visuals, hotspot overlays) and connect to seam validators; demo stubs live at `examples/rom_hotspot_diagnostic.py` and `examples/rom_aggregate_from_samples.py`.
- Regenerate canonical ROM basis via `python scripts/generate_canonical_basis.py --vertices <production mesh npy/npz> --components 128 --harmonics 3 --output outputs/rom/canonical_basis.npz`
  (do not commit the resulting NPZ; keep outputs/rom/ ignored), then run the sampler aggregator with real payloads:
  `PYTHONPATH=src python examples/rom_aggregate_from_samples.py --samples <sampler.json> --basis outputs/rom/canonical_basis.npz --save-costs outputs/rom/seam_costs.npz`
  and pass `--seam-costs outputs/rom/seam_costs.npz` into `generate_undersuit` to annotate seams.
