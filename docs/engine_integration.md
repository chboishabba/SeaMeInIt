# Engine Integration Guide

This guide documents the recommended workflow for bringing SeaMeInIt SMPL-X characters into Unity and Unreal Engine using the `UnityUnrealExporter` utility.

## Preparing Exports

1. Load your SMPL-X mesh, joint hierarchy, and skin weights into a `SMPLXTemplate` via `load_smplx_template`.
2. Instantiate `UnityUnrealExporter` with either `UNITY_CONFIG` or `UNREAL_CONFIG` depending on the target engine.
3. Call `export(template, output_path, format)` with `format` set to `ExportFormat.GLB` (binary glTF) or `ExportFormat.FBX` (ASCII summary).
4. For reproducibility, commit the ASCII summary and documentation alongside regeneration instructions similar to `exports/test_dummy/` (binary payloads can be kept locally).

## Unity

- **Units & Axes:** Exports use a scale of 1.0 (meters), Y-up, Z-forward. Unity’s default import matches these settings.
- **Import Steps:**
  1. Drag `*.glb` or `*.fbx` into the `Assets/` folder.
  2. In the Model tab, set **Rig → Animation Type** to *Humanoid* and **Avatar Definition** to *Create From This Model*.
  3. Enable **Import Constraints** to preserve joint transforms.
  4. For GLB files, Unity automatically applies the correct coordinate system—no additional conversions required.
- **Scripting Tip:** Use `AssetDatabase.ImportAsset` in editor tooling to batch refresh exports after running the Python exporter.

## Unreal Engine

- **Units & Axes:** Exports target centimetres (scale 0.01), Z-up, X-forward when using `UNREAL_CONFIG`. Always re-export with this configuration before import.
- **Import Steps:**
  1. Choose **Import** in the Content Browser and select the generated `*.fbx` or `*.glb` file.
  2. In the FBX Import Options dialog, set **Skeletal Mesh** to enabled and assign a new skeleton.
  3. Ensure **Convert Scene → Force Front XAxis** remains checked (default) to match exporter orientation.
  4. Apply **Import Uniform Scale = 1.0** because the exporter already handles unit conversion.
- **Optional Script:** In Editor Utility Blueprints, call `ReimportAsset` on the mesh after regenerating files to avoid manual refreshes.

## Validation

- Use the included pytest suite (`tests/test_unity_unreal_export.py`) to confirm exported rigs preserve bone naming and hierarchy metadata.
- External validation can be performed with Autodesk FBX Review or `assimp info <file>` to inspect the generated mesh and joints.
