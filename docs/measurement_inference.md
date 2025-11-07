# Measurement Inference

SeaMeInIt's fitting pipeline can now complete partial manual measurement sets by
conditioning a multivariate Gaussian model on the values provided by the user or
Afflec image extraction. The default dataset lives at
`data/templates/manual_measurement_samples.csv` and captures 15 representative
samples covering the canonical manual measurements used by the SMPL-X
calibration workflow.

## Usage Overview

1. The CLI (`python -m smii.pipelines.fit_from_measurements`) accepts raw
   measurements from JSON or Afflec images.
2. Before validation, the pipeline loads the Gaussian model defined in
   `pipelines.measurement_inference` and infers any missing measurements.
3. The completed vector, along with provenance metadata, is stored in the
   `measurement_report` section of the resulting payload. Each entry records:
   - `source`: `measured` or `inferred`
   - `confidence`: a normalised score derived from the conditional variance
   - `variance`: the estimated variance associated with the inferred value
4. Coverage metrics surface in the CLI output bundle so downstream tools can
   visualise which values were observed versus interpolated.

## Visualisation Hooks

`MeasurementReport.visualization_payload()` returns a serialisable list of
measurements that downstream dashboards can use to highlight inferred values
(e.g., by colour-coding low-confidence predictions). The report also exposes a
`coverage` ratio to drive summary widgets showing how many manual inputs were
available for a given subject.
