# Afflec end-to-end runner (mesh-first)

This is the enforced, mesh-based pipeline for the Afflec fixture. It ensures every stage shares the same SMPL-X mesh and measurements derived from that mesh (no pseudo paths).

For mesh lineage and topology-family disambiguation (3240 vs 9438 historical
branches), see `docs/mesh_provenance_afflec.md`.
For seam solve-domain intent vs observed behavior, see
`docs/seam_pipeline_intended_vs_observed.md`.

## Prereqs
- SMPL/SMPL-X assets in `assets/smplx` (see `tools/download_smplx.py`).
- Afflec fixtures in `tests/fixtures/afflec/` (pgm + jpg/avif).
- Install extras: `pip install -e .[vision]`.

## Steps (deterministic)

1) Fit Afflec images → mesh
```bash
export TS=$(date -u +%Y%m%d_%H%M%S)
export BODY_DIR=outputs/bodies/afflec_fixture/${TS}
export ROM_DIR=outputs/rom/afflec_fixture/${TS}
export SEAMS_DIR=outputs/seams_run/afflec_fixture/${TS}

PYTHONPATH=src python -m smii.app afflec-demo \
  --images tests/fixtures/afflec \
  --output "${BODY_DIR}" \
  --detector bbox   # default; use --detector mediapipe if you prefer the full model
# add --force to override existing files without deleting the directory
```
Outputs: `${BODY_DIR}/afflec_body.npz`, `${BODY_DIR}/afflec_smplx_params.json`, measurement report.

2) Sample ROM + seam costs on that mesh
```bash
PYTHONPATH=src python -m smii.rom.sampler_real \
  --body "${BODY_DIR}/afflec_body.npz" \
  --poses data/rom/afflec_sweep.json \
  --weights data/rom/joint_weights.json \
  --fd-step 1e-3 \
  --vertex-map nearest --max-map-distance 0.03 \
  --out-correspondence "${ROM_DIR}/rom_correspondence.npz" \
  --out-costs "${ROM_DIR}/seam_costs.npz" \
  --out-meta "${ROM_DIR}/rom_run.json" \
  --mode diagonal
```
Provenance captures body hash, sweep hash, mapping stats.

3) Solve seams + flex heatmap
```bash
PYTHONPATH=src python examples/solve_seams_from_rom.py \
  --body "${BODY_DIR}/afflec_body.npz" \
  --rom-costs "${ROM_DIR}/seam_costs.npz" \
  --rom-meta "${ROM_DIR}/rom_run.json" \
  --weights configs/kernel_weights.yaml \
  --mdl configs/mdl_prior.yaml \
  --solver pda \
  --out "${SEAMS_DIR}" \
  --flex-heatmap
```
Outputs: `${SEAMS_DIR}/seam_report.json`, `overlay.png`, flex heatmaps (`flex_heatmap*.png`, `flex_stats.csv`).

## Guarantees
- Measurements are derived from the fitted mesh (see `src/smii/measurements/from_mesh.py`).
- Afflec photos are ingested; PGMs are deprecated and ignored by default. The text measurement record is `tests/fixtures/afflec/measurements.yaml`.
- Downstream ROM/seam diagnostics consume the same mesh to avoid drift.

## Guardrails

- Do not reuse fixed output paths (like `outputs/afflec_demo`) for debugging runs.
  Use timestamped run roots so older derived artifacts do not appear to “come
  from” newly overwritten meshes.
- Do not rely on filenames like `afflec_body` to mean “Ben baseline”. Always
  record vertex/face counts and hashes (see `docs/mesh_provenance_afflec.md`).

## Latest Afflec CLI hardening (bbox default)
- CLI now logs exact images and supports `--force` / `--clean-output` / `--detector`; default detector is the fast bbox heuristic (use `--detector mediapipe` for the full model). Entry point wired via `__main__`.
- Image expansion ignores `.pgm` by default; legacy PGM support remains only for back-compat. Detector plumbed through `fit_from_images` and measurement inference.
- Fixtures/docs refreshed: `tests/fixtures/afflec/measurements.yaml` added; README updated; this runner doc reflects the photo-first flow.
- Clean rerun (bbox detector) produced:
  - `PYTHONPATH=src python -m smii.app afflec-demo --images tests/fixtures/afflec --output outputs/afflec_demo --clean-output --force`
  - `PYTHONPATH=src python -m smii.rom.sampler_real … --out-costs outputs/rom/seam_costs_afflec.npz --out-meta outputs/rom/afflec_rom_run.json`
  - `PYTHONPATH=src python examples/solve_seams_from_rom.py … --out outputs/seams_run --flex-heatmap`
  - Record run-local hashes instead of hardcoding historical values:
    - `sha256sum outputs/afflec_demo/afflec_body.npz outputs/rom/seam_costs_afflec.npz outputs/rom/afflec_rom_run.json`
- Tests: `pytest tests/pipelines/test_measurements_from_mesh.py tests/pipelines/test_fit_from_images.py -q` (pass).
- Dependencies added to support this run: `trimesh`, `pymeshfix` (pulls `vtk/pyvista`) to repair non-watertight meshes.
