# Undersuit Generation Pipeline

The undersuit generator derives watertight bodysuit meshes from fitted SMPL-X body
records and can optionally add insulating and comfort-liner layers. The
implementation lives in [`src/suit/undersuit_generator.py`](../src/suit/undersuit_generator.py)
and is exposed through the pipeline entry point
[`src/smii/pipelines/generate_undersuit.py`](../src/smii/pipelines/generate_undersuit.py).

## Workflow Overview

1. Load a fitted body mesh (JSON or NPZ) produced by one of the fitting
   pipelines.
2. Optionally supply a measurement JSON file to control ease and proportional
   scaling.
3. Run the undersuit pipeline to emit meshes and metadata under
   `outputs/suits/<body-record-name>/`.

### Example CLI Invocation

```bash
python -m smii.pipelines.generate_undersuit \
  outputs/meshes/manual_measurement_fit.json \
  --measurements path/to/measurement_overrides.json \
  --base-thickness 0.0015 \
  --insulation-thickness 0.003 \
  --comfort-thickness 0.001 \
  --ease-percent 0.04
```

This command writes the following artefacts:

- `base_layer.npz`: watertight outer skin of the bodysuit.
- `insulation_layer.npz`: optional insulating shell expanded from the base
  layer.
- `comfort_layer.npz`: optional liner mesh derived from the insulation layer.
- `metadata.json`: serialised sizing metadata including layer thicknesses,
  surface areas, seam continuity metrics, scaling factors and paths to the
  generated artefacts.

## Layering Options

Layer toggles and thicknesses are configured via CLI switches or the
`UnderSuitOptions` dataclass when integrating programmatically.

- `--no-insulation` / `include_insulation=False`: skip the insulation layer.
- `--no-comfort` / `include_comfort_liner=False`: skip the comfort liner.
- `--base-thickness`, `--insulation-thickness`, `--comfort-thickness` adjust
  layer offsets (in metres).
- `--ease-percent` applies a global proportional ease to accommodate movement.
- `--weight chest_circumference=2.0`: emphasise individual measurements when
  computing scaling.

By default all layers are generated using the fitted mesh normals to maintain
seam continuity. Metadata includes a `seam_max_deviation` metric that should
remain near zero for watertight inputs.

## Programmatic Usage

```python
from pathlib import Path

from suit import UnderSuitGenerator, UnderSuitOptions
from smii.pipelines.generate_undersuit import load_body_record

body = load_body_record(Path("outputs/meshes/manual_measurement_fit.json"))
measurements = {"chest_circumference": 100.0, "waist_circumference": 82.0}
options = UnderSuitOptions(base_thickness=0.002, include_comfort_liner=False)

generator = UnderSuitGenerator()
result = generator.generate(body, options=options, measurements=measurements)

# Persist meshes or inspect metadata
print(result.metadata["layers"])
```

When integrating into broader pipelines ensure the source mesh is watertight;
the generator validates this before producing layers.

## Seam-Minimal Panel Extraction

The undersuit pipeline now follows a **metric-guided minimal-seam** strategy to
prevent starburst artefacts, reduce manual clean-up, and support non-human
topologies such as dogs. The process enforces four rules that combine geometric
developability with biomechanical constraints.

### 1. Detect flattenable regions

- Evaluate Gaussian and mean curvature per vertex (or use triangle angular
  defect) to measure local developability.
- Treat near-zero curvature zones as candidates for large, seam-free panels and
  flag high-curvature ridges as mandatory seam locations.

### 2. Optimise for the fewest seams

- Thresholded curvature yields a graph where nodes are surface regions and edges
  mark potential cuts.
- Solve for the minimum contiguous cut set that converts the surface into
  topological disks while avoiding over-segmentation.
- Humans typically resolve to 3–4 panels (centre-back, inner-arm, inner-leg,
  neck ring). Dogs generally require 2–4 panels (belly, optional dorsal, shoulder
  loops).

### 3. Align seams with tension maps

- Use measurement inputs (circumferences, limb lengths) to estimate tension
  direction and magnitude.
- Prohibit seams across high-tension axes (e.g., human chest, canine shoulder
  saddle) and prefer placements where loads are minimal or visibility is low.

### 4. Supervise UV unwrapping

- Cut the mesh along the approved seam graph and unwrap each panel with LSCM or
  ABF.
- Reject panels that exceed distortion thresholds and iteratively subdivide
  until developability limits are satisfied.

These steps keep seam counts low, eliminate radial distortion, and extend the
pipeline to quadruped body plans without special cases—only the curvature field
and measurement priors change between species.
