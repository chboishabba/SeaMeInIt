# SMII Receipt Chain Production Roadmap

Generated: 2026-05-14

Status: Gates 0-7 are implemented as a constraint system. The follow-up work
below upgrades bootstrap implementations into production-grade geometry,
field, and manufacturing outputs without changing the receipt order.

## Receipt Chain

```text
Gate 0  BodyCarrierReceipt       emitted by afflec-demo
Gate 1  CorrespondenceReceipt    emitted by reproject_seam_report.py
Gate 2  BasisReceipt             emitted by generate_canonical_basis.py
Gate 3  ROMFieldReceipt          emitted by rom_aggregate_from_samples.py
Gate 4  SeamCostReceipt          emitted by compute_seam_costs.py
Gate 5  SolverPromotionReceipt   emitted by solve_seams.py
Gate 6  PanelUnwrapReceipt       emitted by unwrap_panels.py
Gate 7  ManufacturingReceipt     emitted by generate_manufacturing_artifacts.py
```

The dependency order is:

```text
carrier_receipt -> A_body -> B_0 -> ROM/fabric fields -> seam costs
  -> seam topology -> panel unwrap -> manufacture
```

Do not invert this order. Geometry can still be inspected diagnostically, but
promoted downstream artifacts must consume promoted upstream receipts.

## Current Demo Flow

The current Afflec demo path is a native `A_v3240` solve. It does not require a
promoted correspondence receipt because it does not transfer seam topology
across domains. The correspondence lane remains available for reprojection
audits and transfer-backed claims.

The single-command runner is:

```bash
PYTHONPATH=src python scripts/run_afflec_receipted_demo.py \
  --output outputs/demo/afflec_receipted \
  --detector mediapipe \
  --require-high-trust-detector \
  --allow-synthetic-promotion
```

The runner defaults Gate 0 to `--detector mediapipe` because the receipted
demo is a production-style path. `bbox` remains available via
`--detector bbox` for coarse diagnostics and CI smoke checks, but bbox-derived
body receipts are expected to remain non-promoted unless later fit diagnostics
prove otherwise. `--require-high-trust-detector` can be used to fail hard if
the detector path falls back to a coarse source.

Current Afflec fixture status: a bounded MediaPipe Gate 0 run completes and
emits artifacts, so the earlier no-receipt run is not a receipt-chain failure.
The emitted receipt remains diagnostic-only: raw MediaPipe regression is
plausible (`beta_max_abs=1.95`, high detector trust), but measurement refinement
pushes final betas to `max_abs=11.84` and the skull residual remains marginally
above the Gate 0 threshold (`0.3602 > 0.35`). The next production blocker is
measurement-refinement calibration, not detector wiring.

Inspectable outputs for a fully promoted run are:

- `run_manifest.json`
- `body/body_carrier_receipt.json`
- `basis/basis_receipt.json`
- `rom/rom_field_receipt.json`
- `seams/seam_cost_receipt.json`
- `solver/solver_promotion_receipt.json`
- `panels/panel_unwrap_receipt.json`
- `manufacturing/manufacturing_receipt.json`
- `manufacturing/cutting_layout.svg`
- `manufacturing/seam_allowance.npz`

The expected first blocker on real Afflec data is Gate 0. If
`body_carrier_receipt.json` has `promotion: 0`, the chain stops. That is correct
behavior: the skull/head plausibility threshold is conservative by design.

## Implementation Upgrade Steps

### Step 1 - Afflec Receipted Demo Runner

Status: implemented in `scripts/run_afflec_receipted_demo.py`.

`scripts/run_afflec_receipted_demo.py` is the single-command runner for Gates
0-7.

Requirements:

- execute gates in dependency order
- pass the selected Gate 0 detector through to `smii.app afflec-demo`, with
  `mediapipe` as the default production-style detector
- expose `--require-high-trust-detector` for runs that should hard-fail coarse
  detector fallback
- stop at the first non-promoted receipt
- write `run_manifest.json` with gate promotions, receipt paths, hashes, and
  timestamps
- support `--dry-run` without writing artifacts
- return exit code `0` for all promoted, `1` for expected non-promotion, and
  `2` for hard gate violations or missing required artifacts

Exit criteria: promoted and blocked runs both produce correct manifest state;
dry runs print the planned command chain without creating run artifacts.

### Step 2 - Real LSCM/ABF/ARAP Unwrap

Replace the Gate 6 bootstrap panel-local projection with real conformal
flattening backends.

Requirements:

- unwrap each topological disk panel with LSCM, ABF, or ARAP
- preserve the existing `PanelUnwrapReceipt` schema and promotion gate
- keep `panels_are_disks=false` as a topology error, not an unwrapper error
- derive grain directions from principal curvature or another documented
  fabrication-relevant direction field

Exit criteria: per-panel distortion is lower than the bootstrap projection and
borderline panels promote with a clear distortion margin.

### Step 3 - Pattern Exporter Wiring

Wire Gate 7 into `src/exporters/patterns.py`.

Requirements:

- produce stitch-line and cut-line geometry rather than only SVG metadata
- offset panel boundaries by variable seam allowance `a_s(v)`
- emit SVG, PDF, DXF, `panel_manifest.json`, `seam_allowance.npz`, and
  `manufacturing_receipt.json`
- label seam partners, grain directions, notches, and scale calibration

Exit criteria: the DXF opens at the correct scale in a CAD tool; cut and stitch
lines are geometrically distinct; seam partner labels match across panels.

### Step 4 - Real ROM Fields

Fix the environment needed by `smii.rom.sampler_real` and replace synthetic Gate
3 fields with corpus-aggregated fields.

Requirements:

- resolve missing SMPL-X runtime extras
- emit `pose_source="rom_corpus_aggregated"`
- retain the `ROMFieldReceipt` hash chain and field-uniformity gate

Exit criteria: `field_uniformity < 0.85`; pressure peaks are visibly elevated
at expected support regions; seam-cost uniformity improves.

### Step 5 - Cotangent Laplace-Beltrami Basis

Replace the Gate 2 sinusoidal/QR basis with a cotangent-weight
Laplace-Beltrami basis plus anatomical atlas weighting.

Requirements:

- construct vertex-aligned eigenmodes on the promoted carrier
- record atlas/mode provenance in `BasisReceipt`
- keep the receipt hash chain from basis to body carrier

Exit criteria: relative reconstruction error drops below `0.03` on the test
field and modes are smooth and anatomically sensible.

### Step 6 - Production Manufacturing Package And Ruff

Finish the fabrication package and tooling gate.

Requirements:

- package SVG, DXF, PDF, panel manifest, seam allowance field, and final
  manufacturing receipt
- add a `can_send_to_cutter()` style check for scale, labels, notches, grain,
  seam partners, non-flat allowance, and artifact hashes
- add Ruff to development tooling and CI checks

Exit criteria: the package passes cutter-readiness checks and Ruff is clean.

## Formal Objects Not Yet Represented In Code

These are defined by the broader formalism but remain future implementation
objects:

| Object | Meaning | Status |
|---|---|---|
| `SeamFeedbackOperator` | seams modify field transport | not started |
| `SoftTissueModel` | breast/glute/hip oscillation dynamics | not started |
| `StandoffField` | body-surface standoff over motion | not started |
| `SilhouetteField` | social and ergonomic admissibility | not started |
| `FootwearCarrier` | podiatry, plantar pressure, gait, contact | not started |
| `TerrainCorpus` | grip optimization over terrain classes | not started |
| `TSFV` | support topology stable under reverse traversal | not started |

These become relevant after the pattern exporter produces honest fabrication
artifacts.
