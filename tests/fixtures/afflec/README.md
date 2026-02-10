# Afflec Image Fixtures

The fixtures in this directory are cheeky Ben Afflec stills that emulate a
fully-instrumented capture pipeline. The real regression path now uses the RGB
photos (`afflec1.jpg`, `afflec2.jpg`, `afflec3.avif`) and derives measurements
from the fitted SMPL-X mesh.

For reproducibility (and to avoid binary-only metadata), the canonical
measurement record lives in `measurements.yaml`. Legacy `.pgm` files are kept
for back-compatibility of the header parser but are no longer used for
regression by default.
