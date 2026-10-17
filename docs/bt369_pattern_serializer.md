# BT369-native pattern partitioner

## Why this exists

The measured sequential depth-2 replay on curated Afflec confirms that the current local operator basis is exhausted on P0/P2 under fresh backend measurements. This module tests the next narrower hypothesis before introducing a new UV solver:

> can BT369 residual/orientation/refinement state generate a better face partition while the existing numerical serializers remain unchanged?

The answer is measured under the existing Gate-6 rules. BT369 receives no privileged promotion path.

## What the first rung does

`smii.unwrap.bt369_pattern_serializer` assigns each original parent face:

- measured serializer residual;
- residual trit (`-1` foldover, `0` within threshold, `+1` high residual);
- six-sector orientation bin;
- bounded base-3 refinement depth;
- triadic centroid prefix;
- normalized face-area weight;
- measured foldover/high-distortion flags.

Face adjacency is cut by a bounded policy over foldover boundaries, residual-trit changes, orientation mismatch, refinement-depth mismatch, and triadic-prefix mismatch. Connected components form an exact original-face partition.

The partition receipt is always diagnostic (`promotion = 0`, blocker `backend_metrics_not_recomputed`).

`scripts/evaluate_bt369_pattern_partition.py` then:

1. reads the exact existing `PanelUnwrapReceipt` and serializer failure fields;
2. sweeps 18 deterministic policies (`orientation limit 0/1/2 × depth limit 0/1/2 × prefix depth 0/1`);
3. deduplicates identical partitions by topology hash while retaining all policy aliases;
4. preserves any already-authorized synthetic gusset patches from correction-tree materialization;
5. rebuilds face-backed charts;
6. reruns the existing bootstrap/LSCM/xatlas competition;
7. recomputes score, worst distortion, foldovers, boundary deviation, chart count and exportability;
8. applies the unchanged parent score/distortion rule plus foldover/exportability guards;
9. emits Pareto/profile diagnostics without changing promotion.

## Curated Afflec command

From the SeaMeInIt repository root on `maxcut-rom-suit-closure`:

```bash
../.venv/bin/python -m scripts.evaluate_bt369_pattern_partition \
  --panel-unwrap-receipt outputs/demo/afflec_receipted_curated_20260605_143059/panels_fabric_operator_basis_search/panel_unwrap_receipt.json \
  --mesh outputs/demo/afflec_receipted_curated_20260605_143059/body/afflec_body.npz \
  --seam-edges outputs/demo/afflec_receipted_curated_20260605_143059/solver/seam_edges.npz \
  --panel-id P0 \
  --panel-id P2 \
  --out outputs/demo/afflec_receipted_curated_20260605_143059/panels_fabric_operator_basis_search/bt369_pattern_backend_evaluation_P0_P2.json
```

`--panel-id` is optional; with none supplied the evaluator defaults to `P0` and `P2`.

## Focused verification

```bash
../.venv/bin/python -m pytest \
  tests/smii/unwrap/test_bt369_pattern_serializer.py \
  tests/scripts/test_evaluate_bt369_pattern_partition.py -q

../.venv/bin/python -m ruff check \
  src/smii/unwrap/bt369_pattern_serializer.py \
  scripts/evaluate_bt369_pattern_partition.py \
  tests/smii/unwrap/test_bt369_pattern_serializer.py \
  tests/scripts/test_evaluate_bt369_pattern_partition.py

../.venv/bin/python -m ruff format --check \
  src/smii/unwrap/bt369_pattern_serializer.py \
  scripts/evaluate_bt369_pattern_partition.py \
  tests/smii/unwrap/test_bt369_pattern_serializer.py \
  tests/scripts/test_evaluate_bt369_pattern_partition.py
```

Then run the repository's normal full pytest suite before merging.

## Reading the result

The top-level receipt is `smii.bt369_pattern_backend_evaluation.v1`.

For each panel inspect:

- `policy_count` versus `unique_topology_count`: how much of the parameter grid collapses to the same actual cut topology;
- `best_bt369_candidate_id`;
- `promoted_candidate_ids`;
- each candidate's `metrics`, `blockers`, `pareto_frontier_member`, and `policy_aliases`;
- `partition_receipt.cut_reason_histogram` to see whether foldover, orientation, depth, trit, or prefix boundaries actually drove the cut.

A useful non-promoted candidate still matters if it moves the P0/P2 Pareto frontier. In particular, P0's folded parent has an artificially small `worst_distortion = 0.209` while carrying 1069 foldovers, so the Pareto receipt should be read alongside the hard gate rather than treating that scalar distortion as an independent indicator of good geometry.

## Next cut

- **Any P0/P2 promotion:** integrate the winning BT369 partition policy into the materialization-aware search, then test across broader assets.
- **Pareto improvement but backend-limited:** prototype a native BT369 coordinate solver on the winning partition(s).
- **No frontier movement:** do not blindly increase search depth. Add missing garment state (fabric anisotropy/grain, ROM/load/support field, or a more geometric cut generator) and rerun the same measured competition.

## Tranche C: manufacturing-aware regularized screening

The earlier unregularized P0/P2 promotions are **geometric-only diagnostics**:
single-face confetti can trivially flatten. Tranche C preserves their
`partition_receipt` and `source_manufacturing` evidence, then merges undersized
connected regions into adjacent neighbors using residual/orientation compatibility
and shared-edge length. Each region remains face-connected.

The default screening limits are exploratory proxies, **not** physical standards:

- maximum 64 charts per original panel;
- minimum 10 faces per chart;
- minimum 0.5% of original panel 3D surface area per chart;
- maximum total internal cut length / sqrt(original panel area) of 12.

The receipt reports both source and regularized chart complexity, seam length,
area fractions, and explicit `manufacturing_blockers`. Any candidate that fails
the cheap manufacturing screen is marked `screened_before_backend: true`,
`metrics: null`, `geometric_promotion: null`, and `promotion: 0`; the
expensive LSCM/xatlas competition is **not** invoked. Backend-evaluated candidates
retain measured geometric promotion separately from manufacturing admissibility.
No proxy gate authorizes actual cutting, assembly, wear, or deployment.

Run additional focused tests:

```bash
../.venv/bin/python -m pytest tests/smii/unwrap/test_bt369_manufacturing.py tests/smii/unwrap/test_bt369_pattern_serializer.py tests/scripts/test_evaluate_bt369_pattern_partition.py -q
../.venv/bin/python -m ruff check src/smii/unwrap/bt369_manufacturing.py scripts/evaluate_bt369_pattern_partition.py tests/smii/unwrap/test_bt369_manufacturing.py
../.venv/bin/python -m ruff format --check src/smii/unwrap/bt369_manufacturing.py scripts/evaluate_bt369_pattern_partition.py tests/smii/unwrap/test_bt369_manufacturing.py
```

The historical Afflec command above is unchanged. Inspect
`screened_candidate_count`, `source_manufacturing`, `manufacturing`,
`regularized_partition`, and `geometric_promotion` before drawing conclusions.
