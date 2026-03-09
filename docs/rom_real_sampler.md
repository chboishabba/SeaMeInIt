# Real ROM sampler (finite differences, streaming contraction)

This note documents the finite-difference ROM sampler that derives real seam
cost fields directly from SMPL-X pose sweeps. It avoids analytic Jacobians and
never materialises the `(V×J×J)` supertensor; all contractions stream per pose
with diagonal joint weights.

## Inputs and invariants

- Body mesh NPZ (`vertices`, `faces`) plus SMPL-X parameter payload for shape
  and scale (sidecar `*_smplx_params.json` when using the afflec demo).
- Pose sweep spec (`data/rom/afflec_sweep.json`): deterministic list of pose
  overrides using SMPL-X body joint names; axis-angle radians; optional per
  pose weight; missing joints default to zero (neutral).
- Joint weight spec (`data/rom/joint_weights.json`): diagonal weights per pose
  coordinate; scalars broadcast to the full axis-angle triplet for a joint;
  overrides by joint name are supported for readability.
- FD step `h` in radians (default `1e-3`) and a small `epsilon` added to the
  displacement norm when forming sensitivities.
- Outputs must be deterministic: identical inputs → identical NPZ bytes.

## Pose sweep format

```
{
  "meta": {"units": "radians", "joint_layout": "smplx_body_v1"},
  "neutral": {"global_orient": [0, 0, 0]},
  "poses": [
    {"id": "neutral"},
    {
      "id": "arms_up",
      "weight": 1.0,
      "body_pose": {
        "left_shoulder": [0.8, 0.0, 0.0],
        "right_shoulder": [-0.8, 0.0, 0.0]
      },
      "notes": "Symmetric shoulder flexion"
    }
  ]
}
```

- Joint keys must come from `SMPLX_BODY_JOINTS` (21 joints × 3 = 63 body pose
  values). `global_orient` accepts a 3-vector axis-angle. Additional pose keys
  (jaw/eyes/hands/transl) mirror the SMPL-X parameter names when needed.
- Pose order is preserved; an optional `--pose-limit` CLI flag truncates the
  list deterministically for fast smoke runs.

## Weight format (diagonal `W`)

```
{
  "meta": {"joint_layout": "smplx_body_v1", "notes": "Diagonal weights"},
  "weights": {
    "global_orient": 0.0,
    "body_pose": {
      "default": 1.0,
      "overrides": {"left_shoulder": 1.5, "right_shoulder": 1.5}
    },
    "jaw_pose": 0.0,
    "left_hand_pose": 0.0,
    "right_hand_pose": 0.0,
    "transl": 0.0
  }
}
```

- Scalars broadcast across a parameter; sequences must match the parameter
  length. `overrides` apply per joint (all three axes) when the layout is
  `smplx_body_v1`.
- `--joint-subset` can narrow evaluation to named parameter blocks for faster
  runs, but shape alignment is still validated.

## Algorithm sketch (diagonal mode)

1. Load neutral mesh `V₀ = Φβ(θ₀)` from the SMPL-X body model using the provided
   parameters and scale once. Reuse the same model for all poses.
2. For each pose `θ` with measure weight `Δμ(θ)`:
   - Evaluate `V(θ)`.
   - For each active pose coordinate `j` with weight `w_j`:
     - `V⁺ = Φβ(θ + h e_j)`, `V⁻ = Φβ(θ - h e_j)`.
     - `Ṽ_j = (V⁺ - V⁻) / (2h)`.
     - `d = V(θ) - V₀`; `S_{vj} = | (d_v / (||d_v||₂ + ε)) · Ṽ_{vj} |`.
     - Accumulate `q_v += w_j * S_{vj}²` without storing `(V×J×J)` tensors.
   - Add `q_v` into the seam cost accumulator `c_v ← c_v + Δμ(θ) q_v`.
3. Save vertex costs via `save_seam_cost_field`; edges remain empty in this
   sprint. Metadata captures body hash, sweep hash, weights hash, FD step, git
   commit, call counts, and `meta.synthetic=false`.

## CLI usage (afflec target)

```
PYTHONPATH=src python -m smii.rom.sampler_real \
  --body outputs/afflec_demo/afflec_body.npz \
  --poses data/rom/afflec_sweep.json \
  --weights data/rom/joint_weights.json \
  --fd-step 1e-3 \
  --vertex-map nearest \
  --max-map-distance 0.03 \
  --out-correspondence outputs/rom/afflec_rom_correspondence.npz \
  --out-costs outputs/rom/seam_costs_afflec.npz \
  --out-meta outputs/rom/afflec_rom_run.json \
  --mode diagonal

PYTHONPATH=src python -m smii.pipelines.generate_undersuit \
  outputs/afflec_demo/afflec_body.npz \
  --seam-costs outputs/rom/seam_costs_afflec.npz \
  --output outputs/undersuit_demo \
  --pdf-page-size a4
```

Outputs are deterministic and sized to the body vertex count (for example,
`3240` on the current `outputs/afflec_demo/afflec_body.npz`; legacy branches in
this repo also include `9438`-vertex artifacts).
`--out` can optionally emit an audit sampler JSON summarising the pose sweep and
call counts alongside the direct seam costs.

## Operator inspection outputs

The sampler can now emit operator-level coefficient artifacts in addition to the
topology-bound seam-cost NPZ:

```bash
PYTHONPATH=src python -m smii.rom.sampler_real \
  --body outputs/afflec_demo/afflec_body.npz \
  --weights data/rom/joint_weights.json \
  --schedule data/rom/sweep_schedule.yaml \
  --basis outputs/rom/afflec_canonical_basis.npz \
  --out-coeff-samples outputs/rom/afflec_coeff_samples.json \
  --out-envelope outputs/rom/afflec_envelope.json \
  --out-certificate outputs/rom/afflec_rom_certificate.json \
  --out-costs outputs/rom/seam_costs_afflec.npz \
  --out-meta outputs/rom/afflec_rom_run.json
```

Notes:

- `--basis` is required when using `--out-coeff-samples`.
- The current exported coefficient field is `seam_sensitivity`.
- Coefficients are derived from the same per-pose finite-difference sensitivity
  field used to accumulate seam costs, encoded against the orthonormal basis.
- `afflec_coeff_samples.json` is an operator-level artifact; the seam-cost NPZ
  remains a topology-level projection.

If the SMPL-X template vertex count differs from the body mesh (e.g., 10,475 →
3,240), the sampler deterministically remaps costs to the body vertices via a
nearest-neighbour map in the neutral pose and records the mapping statistics in
the provenance JSON.

When `--out-correspondence` is provided, the same remap stage also persists a
bidirectional correspondence artifact (`source_to_target` and
`target_to_source` index arrays with distances). This is the transform-native
map emitted during ogre generation, so downstream reprojection can reuse it
instead of re-deriving NN maps from scratch.

To inspect the ROM object directly, use the standalone report generator:

```bash
PYTHONPATH=src python scripts/render_rom_operator_report.py \
  --basis outputs/rom/afflec_canonical_basis.npz \
  --rom-meta outputs/rom/afflec_rom_run.json \
  --coeff-samples outputs/rom/afflec_coeff_samples.json \
  --envelope outputs/rom/afflec_envelope.json \
  --certificate outputs/rom/afflec_rom_certificate.json \
  --costs outputs/rom/seam_costs_afflec.npz \
  --body outputs/afflec_demo/afflec_body.npz \
  --out-dir outputs/rom/operator_report
```

This produces a static HTML report plus adjacent JSON/PNG artifacts and is the
preferred answer to "show me the ROM invariant" in the current repo.

Report contract update:

- coefficient/operator charts should render directly in the HTML as DOM-native
  visual elements rather than being emitted as separate PNG files;
- existing topology-level media artifacts such as seam overlays, flex heatmaps,
  map orbits, GIFs, and WebMs should be embedded in the report page when their
  paths are supplied, so the report becomes the canonical viewing surface;
- JSON sidecars like `report_manifest.json` and `coeff_summary.json` remain the
  machine-readable outputs.

This operator report is now a specialized subpage. The canonical user-facing
surface for a whole run should be a run-level reference page that embeds body,
ROM, seam, heatmap, overlay, and orbit/map media together and links back to the
operator page for operator-only inspection.

The report now performs an explicit consistency check across:

- basis vertex count,
- ROM meta vertex count,
- optional body mesh vertex count,
- optional seam-cost vertex count.

If these disagree, the report marks `consistency_status = WARN` and records
human-readable mismatch flags in `report_manifest.json`. It does not fail hard:
the goal is to keep historical artifact inspection possible while making the
mismatch impossible to miss.

## Mapping policy and seam graph gaps

- Vertex mapping is now explicit:
  - `--vertex-map nearest|error` selects deterministic nearest-neighbour remap or a hard failure when counts differ.
  - `--max-map-distance <meters>` raises a warning (or error in `error` mode) when any mapped vertex exceeds the threshold; default 0.03 m (3 cm).
  - `--out-correspondence <path.npz>` exports full bidirectional map arrays and
    collision metrics from this exact sampler run.
- Seam graph aggregation handles missing coverage: if a panel has no mapped vertices or edges, the exported seam costs are zero-filled and a warning lists the affected seams so you can inspect topology.
- Seam cost metadata now includes per-panel counts (`vertex_count`, `edge_count`) and empty flags so reports/exports can highlight zero-filled panels.
