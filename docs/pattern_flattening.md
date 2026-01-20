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
  --pdf-page-size a4 \
  --annotate-level summary \
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
  `"seam_allowance": <metres>`. Optional keys include `seam_partner` and
  `seam_length_tolerance` for seam length reconciliation. Use `seam_partners`
  to author explicit edge-to-edge intent; each entry captures edge index ranges
  on the ordered outline and the partner panel/edge. If `seam_partners` are
  present, the pipeline derives `seam_avoid_ranges` for seam-aware splitting
  unless you set `seam_avoid_ranges` explicitly. `seam_midpoint_index` can still
  override split placement when a seam edge is meant to be the cut axis.
  Additional keys are preserved inside exported metadata so downstream tooling
  can distinguish stitch types or reinforcements.

Example seam partner entry (edge indices refer to the ordered outline used by
the flattening backend):

```json
{
  "seam_partners": [
    {
      "edge": [12, 20],
      "partner_panel": "panel_B",
      "partner_edge": [4, 12],
      "role": "primary",
      "zone": "side_torso"
    }
  ]
}
```

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
- Regularization issues are exported as `metadata["panel_issues"]` with severity
  and per-issue fields, plus `metadata["panel_issue_summaries"]` for rollups.
- When auto-splitting is enabled, `metadata["auto_split"]` records the split
  strategy and resulting panel count.
- Seam length mismatches emit `SEAM_MISMATCH` issues when `seam_partner` metadata
  is present.
Auto-split guidance follows `CONTEXT.md` lines 4555-4675 for pipeline gating.

The exported PDF includes a textual summary of all panels, seam allowances, and
the backend used. SVG and DXF files embed seam allowance values as attributes so
local CAD or nesting software can pick them up automatically.

## PDF tiling and page size

PDF output tiles panels across multiple pages when a single sheet is too small.
Supported page sizes are `a4` (default), `letter`, and `a0`. The tiling preserves
panel coordinates so downstream assembly remains deterministic; the multi-page
PDF mirrors the pipeline order shown in `CONTEXT.md` lines 800-815.

## Troubleshooting

1. Inspect `metadata.json["patterns"]["panel_warnings"]` and
   `metadata.json["patterns"]["panel_issue_summaries"]` to catch panels that
   exceeded budgets or require review/splitting.
2. Review `metadata.json["patterns"]["measurement_loops"]` to ensure the loop
   coordinates align with the intended fit adjustments.
3. When switching backends, regenerate the `patterns/` directory to keep the
   metadata in sync—older files are not overwritten automatically.
