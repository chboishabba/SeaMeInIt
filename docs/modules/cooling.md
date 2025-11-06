# Cooling Module Integration

The cooling module couples the undersuit's thermal zoning with either phase-change material (PCM) packs or an actively pumped liquid loop. This document outlines how to install the tooling, run simulations, and configure media-specific behaviour.

## Installation

1. Create or activate a Python 3.11 virtual environment.
2. Install the project along with development extras:

   ```bash
   pip install -e .[dev,test]
   ```

3. (Optional) Install GPU-capable solvers or CFD back-ends if you intend to validate loop dynamics with external tools. The cooling manifests emitted by the pipeline are JSON and can be ingested by most thermal solvers.

## Generating Cooling Manifests

The undersuit generator can emit an accompanying routing manifest when invoked with the `--embed-cooling` flag. The manifest is written to `outputs/modules/cooling/<undersuit-id>/` and references the suit output folder.

```bash
python -m smii.pipelines.generate_undersuit fitted_body.npz \
  --output outputs/suits/demo_body \
  --embed-cooling \
  --cooling-medium liquid
```

The command above produces standard undersuit meshes under `outputs/suits/demo_body/` and a liquid-loop manifest at `outputs/modules/cooling/demo_body/demo_body_liquid.json`.

## Medium Configuration

### PCM Packs

* Select with `--cooling-medium pcm`.
* Allocates latent-capacity PCM packs to each thermal zone proportional to its baseline heat load.
* Flow rates are reported as zero because the loop relies on passive conduction into the PCM reservoirs.
* Use when missions prioritise simplicity and low power draw.

### Liquid Loop

* Default mode (`--cooling-medium liquid`).
* Sizing heuristics target at least 1.5 L/min total flow and distribute the rate proportionally across zones.
* The manifest includes pump sizing, per-zone loop segments, and a routing graph linking manifolds to zones.
* Suitable when missions require sustained high heat rejection or integration with vehicle chillers.

## Simulation Workflow

1. Generate undersuit meshes and cooling manifests as shown above.
2. Feed the manifest JSON into your thermal solver of choice. Each circuit entry includes loop length, inner diameter, and nominal flow rate.
3. Validate load balancing by confirming zone capacities in `zone_pcm_capacity` and `zone_flow_rate` match mission profiles.
4. Iterate on medium selection or override global capacity/flow inputs via the programmatic API (`modules.cooling.plan_cooling_layout`).
