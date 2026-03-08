# Afflec Image Pre-processing

Our so-called "Afflec" captures are nothing more than demo stills of Ben Afflec
that we annotated to exercise the image-fitting pipeline. There is no dedicated
Afflec model hiding behind the scenes.

The repo now supports two image-fit tiers:

- `fit-mode=heuristic`: legacy measurement/landmark heuristic.
- `fit-mode=reprojection` (or `auto`): keypoint-driven SMPL-X reprojection fit
  with iterative optimization plus optional measurement refinement.

PGM header fixtures are still supported as a lightweight measurement fixture
path, but they are no longer the only intended input path.

## Workflow

1. For the full reprojection path, provide RGB stills (`jpg/png/avif`) and run:
   `python -m smii.app fit-from-images --images <paths...> --fit-mode auto`
2. For fixture-style measurement tests, write measurement metadata into PGM
   headers using `# measurement:<name>=<value>`.
3. Save the processed assets to disk and call the relevant CLI:
   - reprojection fit: `python -m smii.app fit-from-images ...`
   - measurement fixture fit: `python -m smii.pipelines.fit_from_measurements --images <paths...>`

The extractor aggregates measurements from all provided images. When duplicate
measurement names appear across files the numeric values must match; otherwise
execution fails with a `MeasurementExtractionError`.

## Generating Sample Assets

Sample fixtures live in `tests/fixtures/afflec/`. They illustrate how to
annotate images without distributing large binaries. You can regenerate them by
running:

```bash
python - <<'PY'
from pathlib import Path

TEMPLATE = """P2
# measurement:height=170.0
# measurement:chest_circumference=96.5
2 2
255
0 0
0 0
"""

Path("tests/fixtures/afflec/sample_front.pgm").write_text(TEMPLATE, encoding="utf-8")
Path("tests/fixtures/afflec/sample_side.pgm").write_text(
    TEMPLATE.replace("chest_circumference=96.5", "waist_circumference=82.3"),
    encoding="utf-8",
)
PY
```

These files contain enough metadata for automated tests while keeping the
repository binary-free. Replace the measurement names and values with the
outputs from your own pre-processing routine to drive production fits—just
remember that the "Afflec" fixtures are playful stand-ins, not a production
dataset.

## Reprojection artifacts

When the reprojection path is used, the body output root includes:

- `*_observations.json`: 2D keypoints and silhouette-bbox observations used by
  the optimizer.
- `*_regression_raw.json`: canonical shared-shape plus per-image pose fit.
- `*_fit_diagnostics.json`: optimization metrics, reprojection RMSE, trust
  level, and consistency flags.

If the detector is `bbox`, those artifacts are still considered coarse inputs.
Use `mediapipe` when available for a higher-trust initialization/observation
path.
