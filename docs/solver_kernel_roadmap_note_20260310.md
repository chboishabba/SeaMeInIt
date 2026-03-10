# Solver / Kernel Roadmap Note

Date: 2026-03-10

This note distills four refreshed archived threads into concrete roadmap
guidance for the ROM kernel and seam solver work:

- `Branch · Three-kernel coupling for ROM`
  - online UUID: `696f0c80-f2e0-8322-b8a3-7b59b1ce3835`
  - canonical: `2732a8b3196238d99153d6dfe71992a95d59bd7e`
- `Pose Sweep Strategy`
  - online UUID: `69707049-9248-8323-b22d-efb493470795`
  - canonical: `5fe149c7b3c1e841ab0f8e6419b9fd225a3f5db9`
- `Seam Walker Troubleshooting`
  - online UUID: `698d5e21-6d54-839a-a127-088c1dc21227`
  - canonical: `0eff7f41332ca191629d9246ad3677518461fa55`
- `Seam Graph Generation Debug`
  - online UUID: `699050a6-e13c-839a-9a66-be7653b4db13`
  - canonical: `6d14ca5f93671d7fb8e923db48654ecb5ef63b42`

## Decisions

1. ROM remains an operator over admissible pose space, not a mesh/orbit artifact.
- The kernel roadmap should continue to treat ROM as:
  - pose schedule,
  - sampled field/coefficient artifacts,
  - completeness / certificate logic.
- `human` / `ogre` outputs are not the ROM object and should not be used as the semantic reference for kernel correctness.

2. The current kernel should be judged against explicit field semantics, not visual intuition alone.
- Current finite-difference kernel computes a motion-direction-gated sensitivity field.
- Roadmap implication:
  - keep `seam_sensitivity` as one operator output,
  - but compare it explicitly against alternate candidate fields such as:
    - displacement magnitude,
    - derivative magnitude,
    - chain/legality-weighted variants.
- Kernel work should answer:
  - which field best represents the design signal we actually want,
  - rather than assuming the current field is final because it is implemented.

3. No-op detection must be a first-class invariant in ROM/body reruns.
- If mesh / hotspot outputs look unchanged, treat the run as a no-op until mtimes
  and content hashes prove otherwise.
- Roadmap implication:
  - every body/ROM/seam pipeline stage should emit stable hashes/lineage,
  - run reports should surface whether outputs were regenerated or reused.

4. Seam quality should be improved through structural constraints, not anatomy folklore.
- Do not encode “crotch bad”, “mouth bad”, etc. as special-case costs.
- Instead, solver work should focus on:
  - loop / panel constraints,
  - valid cut-graph structure,
  - flattenability / sewability objectives,
  - explicit fragmentation control.

5. Starburst/porcupine failure is a real optimization pathology and should be treated as such.
- Optimizing distortion without cut-complexity control encourages over-fragmented,
  semantically useless seams.
- Roadmap implication:
  - add chart/cut complexity regularization,
  - add flattenability-aware scoring,
  - do not accept “low distortion” alone as success.

6. `human` / `ogre` naming must stay provenance-only.
- These labels are not morphology descriptors.
- Roadmap implication:
  - when possible, prefer stage/topology labels such as `fit_v3240`,
    `base_layer_v9438`, `rom_*`,
  - any workflow docs or reports should state clearly that role labels are provenance.

## Near-Term Actions

1. Kernel track:
- Keep the new kernel diagnostic page as the reference inspection surface.
- Compare `seam_sensitivity` against at least one alternate candidate field on the
  same topology before changing solver objectives.

2. Solver track:
- Investigate shortest-path insensitivity under fixed topology.
- Prioritize:
  - anchor/component fallback debugging,
  - comparison against `mincut` and `pda`,
  - loop/panel constraints as problem-definition controls.

3. Sewability track:
- Add explicit flattenability / fragmentation terms to the seam roadmap.
- Treat “porcupine seams” as a failure of the objective, not a weird but acceptable output.

## Current Practical Conclusion

The kernel question and the solver question are separable:

- Kernel side:
  the current field is mathematically coherent, but may not be the final design signal.
- Solver side:
  current shortest-path behavior appears too insensitive to cost-field changes on a fixed topology, so solver diagnostics and structural constraints are the next priority.
