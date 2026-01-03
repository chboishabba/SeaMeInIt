# Flattened Panel Export

The undersuit pipeline now produces flattened panels through the
`SeamGenerator` → `PatternExporter` flow defined in
`src/suit/seam_generator.py` and `src/exporters/patterns.py`. Panels are
segmented by measurement loops, annotated with seam metadata, and then unwrapped
with either the lightweight plane projection backend or the conformal LSCM
solver.

## Automatic export via `generate_undersuit`

Running the standard undersuit CLI now emits ready-to-cut pattern files
alongside the meshes. Use the `--pattern-backend` flag to select the flattening
method:

```bash
python -m smii.pipelines.generate_undersuit \
  outputs/afflec_demo/afflec_body.npz \
  --measurements data/measurements/sample.json \
  --pattern-backend lscm
```

Outputs land under `outputs/suits/<record>/`:

- `patterns/undersuit_pattern.(svg|dxf|pdf)` – flattened panels in vector
  formats.
- `metadata.json["patterns"]` – records the backend, panel names, warning
  flags, measurement loops, and seam attributes used during flattening.
- `panel_payload.json` / `seams_payload.json` – JSON sources that can be
  re-flattened later or re-meshed with different tolerances.

Every run recomputes measurement loops, seams, and panel slices from the base
layer, so the new method stays in sync with updated body metrics without manual
editing.

## Re-flattening or iterating on seam allowances

Use the dedicated pattern exporter CLI when you need to tweak seam allowances,
swap backends, or regenerate files without rerunning the full undersuit solve:

```bash
python -m smii.pipelines.export_patterns \
  --mesh outputs/suits/afflec_bodysuit/panel_payload.json \
  --seams outputs/suits/afflec_bodysuit/seams_payload.json \
  --backend lscm \
  --scale 1.0 \
  --seam-allowance 0.01 \
  --output exports/patterns/afflec_bodysuit
```

This command accepts the serialized panel payload and seam overrides produced by
the pipeline. Any panel missing an explicit entry in the seams JSON inherits the
`--seam-allowance` default provided on the command line.

## Payload schema cheatsheet

- `panel_payload.json` contains `{"panels": [{"name": "...", "vertices": [...],
  "faces": [...]}]}`. Vertices are 3D coordinates in metres; faces form a
  triangle fan that defines the developable patch passed to the backend.
- `seams_payload.json` maps panel names to metadata. At minimum set
  `"seam_allowance": <metres>`; additional keys are preserved inside exported
  metadata so downstream tooling can distinguish stitch types or reinforcements.

You can hand author these files for custom garments as long as they keep the
same structure.

## Demo output model

The current demo output model is the panel payload emitted by
`src/smii/pipelines/generate_undersuit.py` and exercised in
`examples/undersuit_pattern_export.py`. It matches the `PanelPayload` dataclass
in `src/suit/panel_payload.py` and the exporter `Panel3D` shape, even before the
full `Panel` schema (see `CONTEXT.md` lines 564-604) is wired through.

A recent run example is `outputs/suits/afflec_body/metadata.json`, where
`patterns.panels` lists the exported panel names alongside file outputs.

## Outline cleanup and annotations

Flattened outlines are post-processed to keep sewing patterns clean and usable:

- consecutive duplicate vertices are removed
- extreme outlier edges are dropped when they exceed 3x the median edge length
- boundaries are Laplacian smoothed with an interior constraint to avoid drift
- polylines are simplified with Douglas-Peucker to reduce jagged nodes

Annotation metadata can include `grainline`, `notches`, `folds`, and `label`
entries. The exporter renders these as dedicated layers in SVG/DXF/PDF output,
with `panel-outline` drawn from the seam outline and explicit `seam-outline` and
`cut-outline` layers showing seam vs. seam-allowance geometry.

## Backend, scaling, and warnings

- `simple` backend performs a PCA-based projection and is fast enough for smoke
  tests.
- `lscm` backend requires NumPy and python-igl (libigl bindings) but keeps
  angular distortion bounded and should be used for production patterns.
- `scale` globally scales panel outlines before writing SVG/DXF/PDF.
- Per-panel seam allowances override the exporter default; mismatches are
  surfaced via `metadata["panel_warnings"]`.

The exported PDF includes a textual summary of all panels, seam allowances, and
the backend used. SVG and DXF files embed seam allowance values as attributes so
local CAD or nesting software can pick them up automatically.

## Troubleshooting

1. Inspect `metadata.json["patterns"]["panel_warnings"]` to catch panels that
   exceeded the flattening distortion threshold. The seam generator will note
   whether a panel should be subdivided.
2. Review `metadata.json["patterns"]["measurement_loops"]` to ensure the loop
   coordinates align with the intended fit adjustments.
3. When switching backends, regenerate the `patterns/` directory to keep the
   metadata in sync—older files are not overwritten automatically.
