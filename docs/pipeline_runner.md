# Afflec end-to-end runner (mesh-first)

This is the enforced, mesh-based pipeline for the Afflec fixture. It ensures every stage shares the same SMPL-X mesh and measurements derived from that mesh (no pseudo paths).

## Prereqs
- SMPL/SMPL-X assets in `assets/smplx` (see `tools/download_smplx.py`).
- Afflec fixtures in `tests/fixtures/afflec/` (pgm + jpg/avif).
- Install extras: `pip install -e .[vision]`.

## Steps (deterministic)

1) Fit Afflec images → mesh
```bash
PYTHONPATH=src python -m smii.app afflec-demo \
  --images tests/fixtures/afflec \
  --output outputs/afflec_demo \
  --clean-output    # optional: delete previous run to avoid reuse
  --detector bbox   # default; use --detector mediapipe if you prefer the full model
# add --force to override existing files without deleting the directory
```
Outputs: `outputs/afflec_demo/afflec_body.npz`, `afflec_smplx_params.json`, measurement report.

2) Sample ROM + seam costs on that mesh
```bash
PYTHONPATH=src python -m smii.rom.sampler_real \
  --body outputs/afflec_demo/afflec_body.npz \
  --poses data/rom/afflec_sweep.json \
  --weights data/rom/joint_weights.json \
  --fd-step 1e-3 \
  --vertex-map nearest --max-map-distance 0.03 \
  --out-costs outputs/rom/seam_costs_afflec.npz \
  --out-meta outputs/rom/afflec_rom_run.json \
  --mode diagonal
```
Provenance captures body hash, sweep hash, mapping stats.

3) Solve seams + flex heatmap
```bash
PYTHONPATH=src python examples/solve_seams_from_rom.py \
  --body outputs/afflec_demo/afflec_body.npz \
  --rom-costs outputs/rom/seam_costs_afflec.npz \
  --rom-meta outputs/rom/afflec_rom_run.json \
  --weights configs/kernel_weights.yaml \
  --mdl configs/mdl_prior.yaml \
  --solver pda \
  --out outputs/seams_run \
  --flex-heatmap
```
Outputs: `outputs/seams_run/seam_report.json`, `overlay.png`, flex heatmaps (`flex_heatmap*.png`, `flex_stats.csv`).

## Guarantees
- Measurements are derived from the fitted mesh (see `src/smii/measurements/from_mesh.py`).
- Afflec photos are ingested; PGMs are deprecated and ignored by default. The text measurement record is `tests/fixtures/afflec/measurements.yaml`.
- Downstream ROM/seam diagnostics consume the same mesh to avoid drift.

## Latest Afflec CLI hardening (bbox default)
- CLI now logs exact images and supports `--force` / `--clean-output` / `--detector`; default detector is the fast bbox heuristic (use `--detector mediapipe` for the full model). Entry point wired via `__main__`.
- Image expansion ignores `.pgm` by default; legacy PGM support remains only for back-compat. Detector plumbed through `fit_from_images` and measurement inference.
- Fixtures/docs refreshed: `tests/fixtures/afflec/measurements.yaml` added; README updated; this runner doc reflects the photo-first flow.
- Clean rerun (bbox detector) produced:
  - `PYTHONPATH=src python -m smii.app afflec-demo --images tests/fixtures/afflec --output outputs/afflec_demo --clean-output --force`
  - `PYTHONPATH=src python -m smii.rom.sampler_real … --out-costs outputs/rom/seam_costs_afflec.npz --out-meta outputs/rom/afflec_rom_run.json`
  - `PYTHONPATH=src python examples/solve_seams_from_rom.py … --out outputs/seams_run --flex-heatmap`
  - Hashes: `afflec_body.npz` verts `e77036334064e2f9bd36caf6c943f543c1aa53ab46a8495bf9d13ea82e6b0f56`, faces `4ee8c5bb9a47e5fec7e2cd14c64f8c566d522f8ff0b12fefaed62bdd23c690b3`; `seam_costs_afflec.npz` `120f4aa1d6f7b25d0249b9fdd22b523a1555ebe27262ced2cc4abca6abf5e0a2`; `afflec_rom_run.json` `ef68dc9c8227df7dda851e7c6a151ef2d51772fa8c1a6092f59bd7e6bdcd432d`; `flex_heatmap_with_seams.png` `a6201535be5c3a05ce9b6410f6479374d70be01bb002fe54a963ae42917bdab4`.
- Tests: `pytest tests/pipelines/test_measurements_from_mesh.py tests/pipelines/test_fit_from_images.py -q` (pass).
- Dependencies added to support this run: `trimesh`, `pymeshfix` (pulls `vtk/pyvista`) to repair non-watertight meshes.
