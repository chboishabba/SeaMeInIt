# Afflec Image Fixtures

The fixtures in this directory emulate Afflec captures by embedding measurement
metadata inside ASCII portable graymap headers. They are intentionally tiny so
that they remain human-readable and avoid committing binary blobs.

Each comment line that starts with `# measurement:` supplies a measurement name
and value. The extractor used by `smii.pipelines.fit_from_images` parses these
lines and aggregates the measurements across all provided images.
