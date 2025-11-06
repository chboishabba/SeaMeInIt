# Tent Module Deployment Workflow

The tent module augments the base suit with an inflatable canopy that can be
packed behind the shoulders and deployed for emergency shelter. The workflow is
split across three layers:

1. **Geometry templates** – stored in `assets/modules/tent/` and loaded through
   `modules.tent.deployment.load_canopy_template()` and
   `modules.tent.deployment.load_canopy_seams()`.
2. **Deployment planning** – created with
   `modules.tent.deployment.build_deployment_kinematics()` which generates
   attachment anchors, fold paths, and an ordered deployment sequence from the
   suit landmarks.
3. **Export orchestration** – handled by
   `exporters.tent_bundle.export_suit_tent_bundle()` which combines the suit GLB
   export with printable PDF instructions for the canopy fold sequence.

## Preparing inputs

* Provide anatomical landmarks for the wearer, at minimum `c7_vertebra`,
  `sternum`, `left_acromion`, and `right_acromion`.
* Use `load_canopy_template()` to obtain the baseline panel meshes, and
  `load_canopy_seams()` for fold path definitions and seam allowances.
* Feed the resulting data into `build_deployment_kinematics()` to generate the
  runtime deployment plan; this automatically verifies anchor coverage and fold
  connectivity.

## Exporting a bundle

```python
from exporters import PatternExporter, UnityUnrealExporter, export_suit_tent_bundle
from modules.tent import build_deployment_kinematics, load_canopy_seams, load_canopy_template

canopy_mesh = load_canopy_template()
seams = load_canopy_seams()
kinematics = build_deployment_kinematics(landmarks, seams)

unity = UnityUnrealExporter(config)
patterns = PatternExporter()
result = export_suit_tent_bundle(
    suit_template=template,
    canopy_mesh=canopy_mesh,
    seam_plan=seams,
    deployment=kinematics,
    unity_exporter=unity,
    pattern_exporter=patterns,
    output_dir="outputs/bundles/demo",
)
```

The exporter produces a GLB at `suit_with_tent.glb`, PDF instructions in the
`instructions/` directory, and summarises the anchors and fold sequence in
`bundle.json`.

## Testing and validation

The test suite at `tests/modules/test_tent.py` covers anchor placement,
fold-path validation, and bundle export integration. Run the targeted tests via:

```bash
pytest tests/modules/test_tent.py
```
