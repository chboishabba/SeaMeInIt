# Hard Layer Export Workflow

The hard layer exporter converts segmented shell panels into rigid fabrication assets. It
creates STL and STEP representations, embeds identification details, and produces a
self-contained bundle suitable for manufacturing partners.

## Requirements

* Python 3.11+
* Installed package dependencies (`pip install -e .[dev,test]`)
* CAD backend capable of writing STL and STEP (the default exporter emits JSON-backed
  placeholders that are useful for testing but not production)
* Optional: pytest and ruff for local validation

## Preparing Input Data

1. Collect the segmented shell descriptions as a JSON array. Each entry should include:
   * `name`: human readable panel name (e.g. "Chest Panel")
   * `label`: fabrication label applied to both STL and STEP outputs
   * `mesh`: optional path to the originating mesh asset
   * `size_marker`: optional alignment or sizing marker (e.g. "M")
   * `fit_test_geometry`: optional mapping describing QA primitives to embed
2. Gather any attachments (fit notes, reference images, etc.) that need to accompany the
   CAD bundle.
3. Prepare an optional JSON metadata file with programme identifiers, revision codes, or
   operator notes. This information is merged into the exporter-generated metadata.

## Running the CLI

```bash
python -m smii.pipelines.export_hard_layer \
  --panels data/hard_layer/panels.json \
  --metadata data/hard_layer/metadata.json \
  --attachments docs/hard_layer/inspection_checklist.pdf \
  --output exports/hard_layer
```

The CLI performs the following steps:

1. Loads the panel definitions and instantiates `HardLayerExporter`.
2. Generates STL and STEP files for each panel and records per-panel metadata.
3. Copies the provided attachments into `exports/hard_layer/attachments/`.
4. Writes `metadata.json`, combining generated details (labels, fit-test descriptors,
   file paths) with optional metadata and attachment manifests.

## Validating the Bundle

Before sending the bundle for fabrication:

1. Review `metadata.json` to confirm labels, sizing markers, and attachment listings.
2. Open the STL/STEP files inside the target CAD package to ensure marker geometry is
   embedded correctly.
3. Verify attachments contain the latest revision of inspection checklists and assembly
   notes.
4. Run `pytest tests/exporters/test_hard_layer_export.py` to validate exporter behaviour
   after making code changes.

## Troubleshooting

* **Missing output files** – ensure the CAD backend supports both STL and STEP exports.
  The default textual backend is for documentation/testing only.
* **Duplicate panel names** – panel names are slugified before export; duplicates raise a
  `ValueError` to avoid overwriting files.
* **Attachment copy failures** – confirm each attachment path exists and the process has
  permission to read it.
