# Test Dummy Export

This folder contains a minimal SMPL-X derived asset that exercises the Unity/Unreal exporter:

- `dummy.fbx` – ASCII FBX summary with engine coordinate metadata and rig layout.
- `rig_diagram.txt` – ASCII visualization of the generated hierarchy.

## Regenerating Binary Samples

Binary exports (GLB/FBX) are intentionally not tracked to keep the repository free of binary payloads. To regenerate them locally ru
n the following snippet from the repository root:

```bash
python - <<'PY'
from pathlib import Path
from src.exporters.unity_unreal_export import UNITY_CONFIG, UnityUnrealExporter, ExportFormat, load_smplx_template

template = load_smplx_template(
    vertices=[
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.1, 0.5, 0.0],
        [-0.1, 0.5, 0.0],
    ],
    faces=[[0, 1, 2], [0, 1, 3]],
    joint_names=["pelvis", "spine1", "neck", "head"],
    joint_parents=[-1, 0, 1, 2],
    joint_positions=[
        [0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.2, 0.0],
        [0.0, 0.3, 0.0],
    ],
    skin_weights=[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    skin_joints=[
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [3, 0, 0, 0],
    ],
)

exporter = UnityUnrealExporter(UNITY_CONFIG)
exporter.export(template, Path("exports/test_dummy/dummy.glb"), ExportFormat.GLB)
exporter.export(template, Path("exports/test_dummy/dummy.fbx"), ExportFormat.FBX)
PY
```

This produces a GLB mesh alongside the ASCII FBX summary checked into the repository.
