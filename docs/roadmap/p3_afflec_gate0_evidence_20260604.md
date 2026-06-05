# P3 Afflec Gate 0 Evidence - 2026-06-04

Source references: `assets/reference_images/afflec/`

Ignored run root: `outputs/p3_afflec_gate0_20260604/`

Runner:

```bash
../.venv/bin/python scripts/run_p3_afflec_gate0_diagnostics.py --python ../.venv/bin/python
```

## Reference Preflight

MediaPipe pose detection failed on two staged references:

- `1_PAY-EXCLUSIVE-Ben-Affleck-Defends-His-Massive-Back-Tattoo-After-Admitting-Sentiment-Ran-Against-It.avif`
- `Screenshot_20260604_135454.png`

The curated lane also excludes `images.jpg` because it is only `201x251`.

## Lane Results

| Lane | Images | Return | Status | Flags | Promotion | Skull residual | Final crown | Raw crown | Refined crown |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `all_refs_raw` | 10 | 1 | n/a | none | n/a | n/a | n/a | n/a | n/a |
| `all_refs_refined` | 10 | 1 | n/a | none | n/a | n/a | n/a | n/a | n/a |
| `curated_refs_raw` | 7 | 0 | PASS | none | 1 | 0.1430 | 0.1430 | 0.1430 | 0.1430 |
| `curated_refs_refined` | 7 | 0 | PASS | none | 1 | 0.2958 | 0.2958 | 0.1430 | 0.2952 |

## Interpretation

The expanded curated reference set resolves the previous Gate 0 skull-threshold
blocker. Both curated lanes are high-trust `PASS` runs and both promote under
the current `0.35` skull residual threshold. Do not loosen the threshold based
on this evidence.

The all-reference lanes fail before diagnostics because at least one staged
image has no MediaPipe pose. That is a reference-quality/provenance issue, not
a body-threshold issue. Keep those files in the manifest as sensitivity inputs,
but do not use them in promoted Gate 0 calibration runs without a preprocessing
or exclusion policy.

Accepted current assumption: the refined final-export topology change is
acceptable for now. Measurement refinement increases crown residual from
`0.1430` to `0.2958`, and the final export changes topology from the SMPL-X
checkpoint shape (`10475` vertices, `20908` faces) to `9384` vertices and
`18764` faces. The crown metric remains below threshold, so this run may proceed
as promoted evidence while the topology mutation remains documented.

## Next P3 Target

Proceed with the curated promoted Gate 0 lane under the accepted topology
assumption. Add an explicit Gate 0 topology/export diagnostic later, before
using topology mutation as a quality blocker:

- record raw, refined-pre-repair, and final-export vertex/face counts in the
  tracked evidence summary,
- warn or block when final export topology differs materially from the SMPL-X
  checkpoint topology,
- keep `bbox` and failed-pose references diagnostic-only.
