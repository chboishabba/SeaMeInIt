# Hard Shell Clearance Analysis

The hard-shell clearance tooling simulates motion of a rigid armour shell across a set of target
poses.  It highlights where the shell interpenetrates the avatar, measures the closest approach for
each pose, and suggests offsets to expand the shell where contact is persistent.

## Required inputs

The clearance pipeline needs three artefacts:

1. **Shell mesh** – a watertight triangle mesh representing the inside surface of the hard shell.
2. **Target mesh** – the dressed avatar or undersuit surface that the shell needs to accommodate.
3. **Pose transforms** – a JSON list of 4×4 (or 3×3) matrices describing the avatar motion sequence.

Meshes can be provided either as `.npz` archives (`vertices` and `faces` arrays) or JSON payloads with
the same structure.  Pose transforms default to an identity transform if omitted.

## Running the analyser

Use the new CLI entry point to generate a full report:

```bash
python -m smii.pipelines.analyze_clearance shell.npz target.npz --poses poses.json \
  --samples-per-segment 2 --output outputs/clearance/demo_run
```

The command interpolates additional poses between each key transform (controlled by
`--samples-per-segment`) to capture motion arcs.  Results are written to the specified output
directory or to `outputs/clearance/<shell>_vs_<target>` by default.

The pipeline emits three artefacts:

- `report.json` – machine-readable summary containing per-pose clearance metrics, contact points, and
  a global recommended offset vector.
- `report.txt` – human-readable outline of worst penetration, best clearance, and pose-by-pose
  statistics.
- `poses.csv` – tabular snapshot of per-pose clearance values suitable for spreadsheets or plotting.

## Interpreting metrics

Each pose entry captures the minimum signed clearance (`min_clearance`) and the maximum penetration
(`max_penetration`).  Negative minimum clearance values indicate that part of the target mesh pushed
through the shell.  The aggregate `worst_penetration` field is simply the largest penetration across
all poses.

When collisions are detected the analyser averages the normals of all interpenetrating vertices to
produce `recommended_offset`.  This vector can be added to the shell's design clearance (for example
by expanding the shell along that direction) to alleviate the deepest conflicts.

## Workflow tips

1. Generate an initial shell using the existing undersuit or armour tooling.
2. Export a representative motion clip as a sequence of transform matrices.
3. Run `analyze_clearance` and inspect the textual report for any negative clearances.
4. Apply the recommended offset to the shell mesh (or locally sculpt the regions identified by the
   contact list), then rerun the pipeline.
5. Repeat until the best clearance remains positive for the entire sequence.

The JSON contact list includes per-vertex penetration depths and normals, making it straightforward
to drive downstream tooling such as Blender scripts or CAD adjustments.
