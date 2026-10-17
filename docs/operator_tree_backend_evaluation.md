# Measured depth-N operator-tree replay

## Problem

`smii.operator_basis_search.v1` was deliberately conservative: single operators had real materialized backend measurements, while depth-2 candidates were ranked by composing the measured metric deltas of those single operators. That was useful search evidence, but not a promotable geometric result.

Two new rungs remove that ambiguity without weakening Gate 6.

## Rung 1: exact topology materialization

`smii.seams.operator_tree_materialization` and `scripts/materialize_operator_tree.py` materialize operator supports in original parent-face index space. Each operator refines the current chart partition, preserving face identity, full coverage, and pairwise disjointness.

The resulting `smii.operator_tree_materialization.v1` receipt is **always diagnostic**. Its explicit remaining blocker is:

```text
backend_metrics_not_recomputed
```

Topology alone is not distortion evidence.

## Rung 2: backend replay

`python -m scripts.evaluate_operator_tree` consumes:

```text
--panel-unwrap-receipt <existing panel_unwrap_receipt.json>
--mesh <exact source mesh>
--seam-edges <exact seam topology>
--panel-id P<n>
[--tree-id P<n>_T<m>]
--out <receipt.json>
```

If `--tree-id` is omitted, the evaluator selects the retained depth-N tree with the best recorded default-profile score.

The evaluator does **not** trust the depth-N composed metrics. Instead it:

1. reconstructs the exact source panel from the mesh and seam topology;
2. reads the existing `materialized_chart_domains` evidence from the panel receipt;
3. reuses the same single-operator materialization functions as `unwrap_panels.py`;
4. converts each requested operator into a complete original-face partition;
5. computes the common/sequential refinement of those partitions;
6. rebuilds connected chart domains from the refined face sets;
7. reruns all registered serialization backends (`bootstrap_projection`, `lscm`, and `xatlas` when available) on every chart;
8. selects backend candidates using the existing competition logic;
9. recomputes aggregate score, worst distortion, foldovers, boundary deviation, chart count, and exportability;
10. evaluates the existing parent-surgery score/distortion acceptance rule and the stricter operator-search foldover/exportability guard.

The emitted receipt is:

```text
smii.operator_tree_backend_evaluation.v1
```

and hash-binds the source panel receipt, mesh, and seam topology.

## Promotion boundary

A depth-N tree cannot promote because its delta-composed search estimate looked good. It can only obtain `promotion = 1` in the backend-evaluation receipt when the newly materialized charts produce newly measured backend metrics that satisfy the declared relative parent gate and the foldover/exportability guard.

This is still a **parent-surgery promotion**, not manufacturing or physical-suit authority. Absolute panel/fabric/metric gates downstream remain unchanged.

## Current Afflec/P0/P2 boundary

The historical generated directory referenced by the roadmap is not tracked in GitHub. Therefore this implementation does not invent a P0/P2 result. To resolve those panels, reproduce or provide the exact historical/generated artifacts:

```text
panel_unwrap_receipt.json
source mesh
seam topology
```

and run the evaluator for the retained P0/P2 trees. The output will tell us whether true sequential composition removes the earlier depth-2 basis exhaustion under real backend measurements, or whether the BT369-native serializer / larger operator basis remains the next max-cut.
