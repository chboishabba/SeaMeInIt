# Hard Layer Shell Generation

The hard-layer shell builds a rigid protective envelope around a fitted SMPL-X avatar.
This module inflates the body along the vertex normals to produce a watertight hard
shell with configurable thickness behaviour.

## Overview

1. Load a fitted avatar mesh (JSON or NPZ) containing `vertices` and `faces` arrays.
2. Resolve per-vertex shell thickness using either a global value, per-region overrides, or
   a direct per-vertex profile.
3. Apply optional exclusion masks to keep sensitive regions untouched.
4. Offset vertices along their normals and export the generated shell and metadata to
   `outputs/hard_layer/<body name>/`.

The generation logic lives in `suit_hard.shell_generator.ShellGenerator`, and the
end-to-end workflow is exposed via the CLI pipeline
`smii.pipelines.generate_hard_shell`.

## Configuration Options

| Option | Description |
| --- | --- |
| `default_thickness` | Baseline thickness used for all vertices when no overrides are provided. |
| `region_masks` | Mapping of region names to boolean vertex masks used by regional thickness and exclusions. |
| `thickness_profile` | Either a float (uniform), a `{region: value}` mapping, or a per-vertex NumPy array. |
| `exclusions` | Region names or boolean masks that force the shell to remain flush with the body. |
| `enforce_watertight` | Ensures the input mesh is watertight before generating the shell. |

### Region Mask File

The CLI accepts a JSON file describing region masks. The file should map each region
name to an array of vertex indices:

```json
{
  "helmet": [120, 121, 122],
  "torso": [10, 42, 88, 91]
}
```

Indices outside the mesh range raise a validation error.

## CLI Usage

Run the pipeline via `python -m smii.pipelines.generate_hard_shell` or install the
package in editable mode and use the entry point directly.

### Uniform Thickness Shell

```bash
python -m smii.pipelines.generate_hard_shell data/fits/alex.npz \
  --default-thickness 0.005
```

Outputs:

- `outputs/hard_layer/alex/shell_layer.npz` containing `vertices`, `faces`, and `thickness`.
- `outputs/hard_layer/alex/metadata.json` summarising configuration, shell area, and exclusions.

### Regional Overrides with Exclusions

```bash
python -m smii.pipelines.generate_hard_shell data/fits/alex.npz \
  --region-masks configs/masks/alex_regions.json \
  --region-thickness torso=0.006 limbs=0.004 \
  --exclude helmet --exclude hands
```

This command inflates the torso and limb regions using dedicated thickness values while
keeping the helmet and hands aligned with the body mesh.

### Per-Vertex Thickness Profile

```bash
python -m smii.pipelines.generate_hard_shell data/fits/alex.npz \
  --vertex-thickness exports/thickness/alex_shell.npy
```

The pipeline reads the NumPy array, validates its length, and uses it directly for shell
inflation.

## Metadata

`metadata.json` augments the generator output with:

- shell surface area and excluded vertex counts,
- the resolved thickness statistics (min/max/mean), and
- references back to the source body record and output directory.

Use these details to drive downstream quality checks or to audit shell parameter choices.
