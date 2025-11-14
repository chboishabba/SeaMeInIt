# Undersuit Generation Pipeline

The undersuit generator derives watertight bodysuit meshes from fitted SMPL-X body
records and can optionally add insulating and comfort-liner layers. The
implementation lives in [`src/suit/undersuit_generator.py`](../src/suit/undersuit_generator.py)
and is exposed through the pipeline entry point
[`src/smii/pipelines/generate_undersuit.py`](../src/smii/pipelines/generate_undersuit.py).

## Workflow Overview

1. Load a fitted body mesh (JSON or NPZ) produced by one of the fitting
   pipelines.
2. Optionally supply a measurement JSON file to control ease and proportional
   scaling.
3. Run the undersuit pipeline to emit meshes and metadata under
   `outputs/suits/<body-record-name>/`.

### Example CLI Invocation

```bash
python -m smii.pipelines.generate_undersuit \
  outputs/meshes/manual_measurement_fit.json \
  --measurements path/to/measurement_overrides.json \
  --base-thickness 0.0015 \
  --insulation-thickness 0.003 \
  --comfort-thickness 0.001 \
  --ease-percent 0.04
```

This command writes the following artefacts:

- `base_layer.npz`: watertight outer skin of the bodysuit.
- `insulation_layer.npz`: optional insulating shell expanded from the base
  layer.
- `comfort_layer.npz`: optional liner mesh derived from the insulation layer.
- `metadata.json`: serialised sizing metadata including layer thicknesses,
  surface areas, seam continuity metrics, scaling factors and paths to the
  generated artefacts.

## Layering Options

Layer toggles and thicknesses are configured via CLI switches or the
`UnderSuitOptions` dataclass when integrating programmatically.

- `--no-insulation` / `include_insulation=False`: skip the insulation layer.
- `--no-comfort` / `include_comfort_liner=False`: skip the comfort liner.
- `--base-thickness`, `--insulation-thickness`, `--comfort-thickness` adjust
  layer offsets (in metres).
- `--ease-percent` applies a global proportional ease to accommodate movement.
- `--weight chest_circumference=2.0`: emphasise individual measurements when
  computing scaling.

By default all layers are generated using the fitted mesh normals to maintain
seam continuity. Metadata includes a `seam_max_deviation` metric that should
remain near zero for watertight inputs.

## Programmatic Usage

```python
from pathlib import Path

from suit import UnderSuitGenerator, UnderSuitOptions
from smii.pipelines.generate_undersuit import load_body_record

body = load_body_record(Path("outputs/meshes/manual_measurement_fit.json"))
measurements = {"chest_circumference": 100.0, "waist_circumference": 82.0}
options = UnderSuitOptions(base_thickness=0.002, include_comfort_liner=False)

generator = UnderSuitGenerator()
result = generator.generate(body, options=options, measurements=measurements)

# Persist meshes or inspect metadata
print(result.metadata["layers"])
```

When integrating into broader pipelines ensure the source mesh is watertight;
the generator validates this before producing layers.

## Seam-Minimal Panel Extraction

The undersuit pipeline now follows a **metric-guided minimal-seam** strategy to
prevent starburst artefacts, reduce manual clean-up, and support non-human
topologies such as dogs. The process enforces four rules that combine geometric
developability with biomechanical constraints.

### 1. Detect flattenable regions

- Evaluate Gaussian and mean curvature per vertex (or use triangle angular
  defect) to measure local developability.
- Treat near-zero curvature zones as candidates for large, seam-free panels and
  flag high-curvature ridges as mandatory seam locations.

### 2. Optimise for the fewest seams

- Thresholded curvature yields a graph where nodes are surface regions and edges
  mark potential cuts.
- Solve for the minimum contiguous cut set that converts the surface into
  topological disks while avoiding over-segmentation.
- Humans typically resolve to 3–4 panels (centre-back, inner-arm, inner-leg,
  neck ring). Dogs generally require 2–4 panels (belly, optional dorsal, shoulder
  loops).

### 3. Align seams with tension maps

- Use measurement inputs (circumferences, limb lengths) to estimate tension
  direction and magnitude.
- Prohibit seams across high-tension axes (e.g., human chest, canine shoulder
  saddle) and prefer placements where loads are minimal or visibility is low.

### 4. Supervise UV unwrapping

- Cut the mesh along the approved seam graph and unwrap each panel with LSCM or
  ABF.
- Reject panels that exceed distortion thresholds and iteratively subdivide
  until developability limits are satisfied.

These steps keep seam counts low, eliminate radial distortion, and extend the
pipeline to quadruped body plans without special cases—only the curvature field
and measurement priors change between species.




# Design goals

For **undersuits, compression layers, and animal garments** , our pattern generator is designed to:


- Minimise seams: Use as few pattern pieces as possible for comfort and strength.
- Avoid “starburst” panels: Prevent extreme, pointy pattern shapes that are impractical to cut or
sew.
- Respect body movement: Place seams where the body bends least, and ensure the fabric can
stretch where the body needs to flex.
- Generalise across species: The same approach works for human suits, canine vests, etc.,
without being hand-tuned to one anatomy.
- Stay vendor-neutral: We don’t imitate any specific brand’s style or cut. Instead, we base our
designs on fundamental geometric and biomechanical principles rather than a corporate “house
style.”

_(In other words, we look to state-of-the-art athletic and tactical wear techniques, but without name-dropping
or endorsing any particular company—especially those with questionable labor or inclusion records.)_

# Metric-guided minimal-seam segmentation

We treat **seam placement** as a geometric optimization problem, guided by both the body’s shape **and**
measured dimensions. The goal is to balance several factors:

- Flattenability: How easily each region of the 3D body can become a flat pattern without
distortion.
- Tension & movement: Where the suit will be under stretch or compression when worn.
- Topology: How many cuts (seams) are needed to turn the 3D surface into flat pieces.
- Measurement accuracy: Ensuring the flat pattern, once sewn, matches target measurements
(chest, waist, girth, etc.).

The pipeline follows four main steps:

**1. Compute flattenable regions on the body**

First we analyze the 3D body mesh (whether human or animal) to find which areas are **easy to flatten**
versus where there’s a lot of curvature:

We calculate curvature metrics on the surface (e.g. Gaussian curvature). Near-zero curvature
areas can likely be one continuous panel, whereas highly curved areas (shoulders, hips, snout of
a dog, etc.) will require cuts.
This gives us a heatmap of the body: flatter regions that could be large seamless panels, and
high-curvature regions where a seam or dart might be necessary.

In essence, we identify natural “break lines” where cutting the pattern would relieve 3D curvature
tension.

## • • • • • • • • • • •


**2. Find the minimal set of seams for flattening**

Next, we determine the **fewest seams needed** to make the whole suit lay flat:


We model the body surface as a network and search for cut lines that split it into developable
(flattenable) patches. The criteria is to minimize total seam length and count while still allowing the
pieces to lie flat.
The seams are chosen to be as long and smooth as possible (fewer, longer seams are better
than many small jagged ones for comfort and strength).
We avoid placing seams arbitrarily; they’re aligned with the curvature analysis from step 1. For
example, a seam might go along the side of the torso or inner arm where it naturally divides a
curved surface.

Typically, this results in about **2–6 main panels** for a full-body suit, depending on the complexity of the
shape: - _Human example:_ A common solution is a front or back torso panel, two leg panels, and two arm
panels (with maybe a small gusset or neck piece) – so around 4–6 panels total. - _Dog example:_ Often a
back panel and a belly panel can cover the torso, plus separate panels for each limb if needed – as few
as 3–4 panels.

**3. Place seams in low-stress, less visible areas (accounting for anisotropy)**

Not all possible seam placements are equal. We refine seam positions by considering **body mechanics
and fabric behavior** :

Using the person or animal’s measurements (chest, waist, limb lengths, etc.), we infer where the
suit will experience tension or need to stretch. For instance, around the chest and shoulders
there is a lot of expansion during movement, whereas along the spine or side of the body there
is relatively less.
Guiding rule: Avoid running seams through high-motion or high-stress zones. It’s better to put a
seam along the side of a muscle or along a low-motion area like the inner arm, rather than
across the center of the knee or the front of the chest.

We also factor in **material stretch anisotropy** here. Most performance fabrics stretch more in one
direction than the other (think of a knit that stretches horizontally around your body, but not as much
vertically). We align each pattern panel so that the fabric’s **high-stretch direction** corresponds to the
body’s primary stretch need:


For areas that need to stretch around the body (circumferentially), like the chest or abdomen,
we orient that panel so the fabric’s most stretchy axis goes around the torso.
For areas that need lengthwise stretch, like down the arms or legs, we rotate those pattern
pieces so the fabric’s stretch runs along the limb length.
If a region of the body transitions from one stretch orientation to another and the fabric can’t
accommodate that twist, that’s a signal to insert a seam there. In other words, whenever the
required stretch direction changes too much for one panel, we break the panel into two aligned
pieces.

By planning seams this way, we ensure the suit will be comfortable (the fabric gives where it needs to)
and durable (seams aren’t placed where they’ll be pulled apart constantly). It also keeps the suit looking
clean, since we prefer seams along out-of-sight areas (like your inner arm or the side of a leg) over
front-and-center lines.

## • • • • • • • •


**4. Unwrap panels with guided flattening (no “starburst” shapes)**

With the seam lines decided, we “cut” the digital 3D mesh along those seams to produce flat panels, and
then unwrap each panel to get its 2D pattern shape:


We use a conformal unwrapping algorithm (LSCM/ABF methods) on each panel. Essentially, this
mathematically flattens the mesh patch with minimal distortion, like unpeeling an orange peel in
pieces.
Because we already placed proper seams, each panel is a topological disk (no holes or weird
branching), so the unwrapping solver won’t create those crazy spiky starburst patterns. Every
panel will flatten to a sensible 2D shape.
We take care to align the orientation of the flattened panel to the fabric’s grain. In practice, after
unwrapping, we can rotate the pattern piece so that one axis of the pattern lines up with the
fabric’s warp/weft (as determined in step 3 for stretch). This ensures, for example, that the
horizontal direction on the pattern corresponds to the most stretchy direction of the fabric if that
panel needs to stretch around the body.
Finally, we evaluate each panel’s flat pattern for any remaining distortion. If a panel is still
stretching too much (e.g. a particularly curvy piece might still be a bit distorted), the pipeline can
decide to introduce an additional relief cut (another seam) through that panel and flatten again.
In practice, our earlier steps aim to avoid this, but the option is there as a safeguard.

The end result is a set of flat pattern pieces that **meet the body’s measurements** , **minimize sewing
complexity** , and **take advantage of fabric properties**. We achieve a tailored fit with the least number
of seams, and each seam is placed thoughtfully: out of the way, along low-stress lines, and allowing the
fabric to perform as needed. Crucially, by integrating material anisotropy into the digital pattern
generation, we make sure the suit will be comfortable and functional when cut and assembled — no
surprise tight spots or baggy sections due to misaligned stretch.

This approach is **adaptive** and works across different body types or even species because it’s driven by
geometry and metrics, not hard-coded patterns. As long as we have a 3D body mesh and some key
measurements, the system can generate an optimal undersuit pattern that upholds these principles.

## •

## •

## •

## •


