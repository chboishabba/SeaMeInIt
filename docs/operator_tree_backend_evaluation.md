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

## Empirical Afflec/P0/P2 Replay (`afflec_receipted_curated_20260605_143059`)

Replaying `scripts/evaluate_operator_tree.py` against the local curated Afflec operator-basis-search run (`outputs/demo/afflec_receipted_curated_20260605_143059/`):

- `panel_unwrap_receipt.json` SHA-256: `0f4fb1a8c32e6fa7dc6d86cd953d1ff209717c01f0706798f8c08312fa153030`
- `body/afflec_body.npz` SHA-256: `906e9d884df6c2eccaa987f78a2ba778a6e433d138143acdb38d2e933b71f058`
- `solver/seam_edges.npz` SHA-256: `6ec1fa2d32963f933c30c2859ed62d45c51b819702758ef72a32d7e01ddf7fa8`

required handling two materialization invariants from `unwrap_panels._materialized_chart_panels`:
1. auxiliary `inserted_gusset_patch:<node_id>` diamond charts (synthesized vertices `9384..9387` in `materialized_vertices` on `P0` and `P1`) are separated from parent-face partitions and appended once to the refined chart list;
2. `cutout_r1` / `cutout_r2` variants complete the parent partition with their carved-out ring complement so original-face coverage and pairwise disjointness remain exact.

### Measured Depth-2 Results

- **`P0` (initial: `score = 10690.3`, `worst_distortion = 0.209`, `foldovers = 1069`) — all 7 retained depth-2 trees fail (`promotion = 0`):**
  - `P0_T005 ['branch_spoke_r1', 'relief_split']`: `charts = 6`, `valid = True`, `score = 1902.9` (est `891.1`), `foldovers = 190` (est `86`), `worst_distortion = 1.084` (est `19.899`) → blocked by `distortion_regresses` (`1.084 > 0.209`).
  - `P0_T011 ['branch_spoke_r2', 'relief_split']`: `charts = 6`, `valid = True`, `score = 1862.9` (est `1319.1`), `foldovers = 186` (est `129`), `worst_distortion = 1.082` (est `18.543`) → blocked by `distortion_regresses`.
  - `P0_T021 ['failure_lens_patch', 'failure_relief_path']`: `charts = 12`, `valid = False`, `score = 6130.3`, `foldovers = 610`, `worst_distortion = 20.015` → blocked by `materialized_tree_backend_domain_invalid`, `distortion_regresses`.
  - `P0_T023 ['failure_lens_patch', 'relief_split']` (default-profile winner): `charts = 8`, `valid = True`, `score = 1682.7` (est `0.0`), `foldovers = 168` (est `0`), `worst_distortion = 0.720` (est `17.051`) → blocked by `distortion_regresses`.
  - `P0_T025 ['failure_relief_path', 'pareto_guided_dart_wedge']`: `charts = 14`, `valid = False`, `score = 5773.7`, `foldovers = 575`, `worst_distortion = 15.612` → blocked by `materialized_tree_backend_domain_invalid`, `distortion_regresses`.
  - `P0_T026 ['failure_relief_path', 'relief_split']`: `charts = 14`, `valid = False`, `score = 202.2`, `foldovers = 20`, `worst_distortion = 0.835` → blocked by `materialized_tree_backend_domain_invalid`, `distortion_regresses`.
  - `P0_T028 ['pareto_guided_dart_wedge', 'relief_split']`: `charts = 6`, `valid = False`, `score = 6911.9`, `foldovers = 691`, `worst_distortion = 1.087` → blocked by `materialized_tree_backend_domain_invalid`, `distortion_regresses`.

- **`P2` (initial: `score = 10688.1`, `worst_distortion = 5.411`, `foldovers = 1068`) — all 6 retained depth-2 trees fail (`promotion = 0`):**
  - `P2_T002 ['branch_spoke_r1', 'cutout_r2']`: `charts = 9`, `valid = True`, `score = 10418.5` (`< 10688.1`), `foldovers = 1041` (`< 1068`), `worst_distortion = 5.676` (`> 5.411`) → blocked by `distortion_regresses`.
  - `P2_T007 ['branch_spoke_r2', 'cutout_r2']` (default-profile winner): `charts = 5`, `valid = True`, `score = 10418.6` (`< 10688.1`), `foldovers = 1041` (`< 1068`), `worst_distortion = 5.676` (`> 5.411`) → blocked by `distortion_regresses`.
  - `P2_T008 ['branch_spoke_r2', 'pareto_guided_dart_wedge']`: `charts = 6`, `valid = True`, `worst_distortion = 5.262` (`< 5.411`), `score = 13258.0`, `foldovers = 1325` → blocked by `score_not_improved`, `foldovers_regress`.
  - `P2_T014 ['cutout_r2', 'failure_relief_path']`: `charts = 6`, `valid = True`, `score = 11938.3`, `foldovers = 1193`, `worst_distortion = 5.490` → blocked by `score_not_improved`, `distortion_regresses`, `foldovers_regress`.
  - `P2_T015 ['cutout_r2', 'pareto_guided_dart_wedge']`: `charts = 6`, `valid = True`, `worst_distortion = 5.262` (`< 5.411`), `score = 13258.0`, `foldovers = 1325` → blocked by `score_not_improved`, `foldovers_regress`.
  - `P2_T017 ['failure_relief_path', 'pareto_guided_dart_wedge']`: `charts = 4`, `valid = False`, `score = 10538.2` (`< 10688.1`), `foldovers = 1053` (`< 1068`), `worst_distortion = 5.413` (`> 5.411`) → blocked by `materialized_tree_backend_domain_invalid`, `distortion_regresses`.

- **`P3` (`initial: score = 9160.1, worst_distortion = 20.039, foldovers = 913`) — clears parent surgery (`promotion = 1`):**
  - `P3_T007 ['branch_spoke_r2', 'relief_split']`: `charts = 5`, `valid = True`, `score = 3183.6`, `worst_distortion = 0.903`, `foldovers = 318`, `promotion = 1`.

### Conclusion

True sequential partition materialization + fresh backend competition removes the delta-composition fiction and confirms:
1. Composed delta estimates overstated `P0` distortion regression (`17.05–19.90` estimated vs `0.720–1.084` measured) and understated `P0` foldovers (`0` estimated vs `168` measured on `P0_T023`), while `P0` still fails relative parent acceptance because splitting the `1069`-foldover parent chart raises `worst_distortion` from `0.209` to `0.720`.
2. On `P2`, every retained depth-2 pair trades foldover reduction (`1068 → 1041` with `worst_distortion = 5.676 > 5.411`) against distortion reduction (`worst_distortion = 5.262 < 5.411` with `foldovers = 1325 > 1068`); no pair satisfies both simultaneously.
3. The current depth-2 geometric operator basis is therefore **empirically exhausted on `P0` and `P2` under actual sequential materialization**, making BT369-native serialization / a larger geometric operator basis the next operative cut for Gate 6.
