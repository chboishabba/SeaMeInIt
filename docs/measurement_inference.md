# Measurement Inference

SeaMeInIt's fitting pipeline can now complete partial manual measurement sets by
conditioning a multivariate Gaussian model on the values provided by the user or
the Ben Afflec fixture extraction. The default dataset lives at
`data/templates/manual_measurement_samples.csv` and captures 15 representative
samples covering the canonical manual measurements used by the SMPL-X
calibration workflow.

## Usage Overview

1. The CLI (`python -m smii.pipelines.fit_from_measurements`) accepts raw
   measurements from JSON or Afflec fixture images.
2. Before validation, the pipeline loads the Gaussian model defined in
   `pipelines.measurement_inference` and infers any missing measurements.
3. The completed vector, along with provenance metadata, is stored in the
   `measurement_report` section of the resulting payload. Each entry records:
   - `source`: `measured` or `inferred`
   - `confidence`: a normalised score derived from the conditional variance
   - `variance`: the estimated variance associated with the inferred value
4. Coverage metrics surface in the CLI output bundle so downstream tools can
   visualise which values were observed versus interpolated.

## Refinement Authority Boundary

Measurement completion and measurement-based shape refinement are not authority
to replace an image-derived body fit. The current least-squares refinement is a
candidate-generation step. The planned Gate 0 contract will bind the effective
measurement policy (matrix, normalisation, weights, scale rule, beta domain,
prior/anchor weights, and solver settings) and use the image-derived beta
vector as an explicit anchor.

The bounded result must have measurement and skull diagnostics recomputed from
the final candidate. A hash-linked refinement receipt will decide `promote`,
`abstain`, or `reject`; only `promote` can select a refined pre-repair mesh as
the canonical source. A refinement abstention leaves the raw image fit to its
own body-trust guard. Post-solve clipping without recomputing those diagnostics
is not an admissible refinement policy.

## Visualisation Hooks

`MeasurementReport.visualization_payload()` returns a serialisable list of
measurements that downstream dashboards can use to highlight inferred values
(e.g., by colour-coding low-confidence predictions). The report also exposes a
`coverage` ratio to drive summary widgets showing how many manual inputs were
available for a given subject.
