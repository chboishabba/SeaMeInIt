# Hard Layer Shell Segmentation

The hard-shell layer encapsulates the undersuit with rigid armour plates that must articulate smoothly with the avatar's joints.  This document describes how the segmentation system interprets joint motion, which configuration parameters influence panel geometry, and how to work with the CLI pipeline.

## Overview

Segmentation is handled by `suit_hard.segmentation.HardShellSegmenter`.  The segmenter consumes SMPL-X joint positions (either as a mapping from joint name to XYZ coordinates or as an ordered array accompanied by joint names).  For every major limb articulation the system constructs:

- A **motion axis** defined by the vector between the proximal and distal joints (e.g., shoulder→elbow).
- A **cut point** positioned along the limb at a configurable ratio.
- An **elliptical boundary** representing the panel seam.  The ellipse is sampled into a closed polyline and mirrored into the output artefacts.
- A **hinge line** that extends beyond the ellipse width by an allowance factor so hinges have clearance.

The default articulation set includes both shoulders, elbows, hips, and knees.  Panels are named `<side>_<joint>_panel` and provide hinge allowances that downstream CAD tooling can apply during shell fabrication.

## Configuring Articulation Zones

`HardShellSegmentationOptions` exposes the primary tuning parameters:

| Option | Description |
| --- | --- |
| `hinge_allowance` | Clearance applied along hinge seams, in metres. |
| `panel_width_scale` | Fraction of limb length used for the panel's lateral span (tangent direction). |
| `panel_height_scale` | Fraction of limb length used for the circumferential span (binormal direction). |
| `hinge_extension_scale` | Additional hinge length beyond the ellipse radius to accommodate hardware. |
| `boundary_points` | Number of samples used to approximate the elliptical boundary; a final closing point is appended automatically. |

The segmenter also supports custom articulation definitions if more granular panels are required.  Instantiate `HardShellSegmenter` with your own `ArticulationDefinition` sequence to override the defaults.

## CLI Pipeline: `generate_hard_shell.py`

The CLI is available at `python -m smii.pipelines.generate_hard_shell` and accepts fitted body records in either JSON or NPZ format.  Records must include joint positions (`joint_positions` or `joints`) and, ideally, a `joint_names` array.  When names are omitted, a reduced default ordering is assumed for the first twenty joints.

Example usage:

```bash
python -m smii.pipelines.generate_hard_shell data/avatar_record.npz \
  --output outputs/hard_shell/avatar_record \
  --hinge-allowance 0.006 \
  --panel-width-scale 0.28 \
  --panel-height-scale 0.2
```

The pipeline writes one `.npz` file per panel containing the boundary, hinge line, cut point, and motion axis, plus a `manifest.json` summarising configuration metadata and panel counts.

## Interpreting Output Panels

Each panel NPZ exposes arrays ready for CAD import:

- `boundary`: `(N, 3)` array of XYZ vertices forming a closed seam loop.  Adjacent points are guaranteed to be non-zero length, supporting loft/sweep operations.
- `hinge_line`: `2x3` array representing the hinge axis segment in world space.
- `cut_point` and `cut_normal`: describe the plane of the articulation for debugging or visual overlays.
- `motion_axis`: the normalised direction of limb motion, useful when orienting mechanical components.
- `allowance`: scalar representing the hinge clearance, repeated in the manifest for convenience.

Use the `manifest.json` file to confirm the parameters used during generation and to cross-reference each panel.  Because the boundaries are generated with a predictable ordering, downstream tooling can pair opposite sides (e.g., anterior/posterior) using a simple index walk.
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
