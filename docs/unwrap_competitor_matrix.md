# Unwrap Competitor Matrix

SeaMeInIt compares unwrap methods as declared benchmark candidates, not as
global flattening proofs. A promoted benchmark winner means "best measured
candidate in this declared slice under these metrics." It does not mean a
curved surface flattened isometrically, an optional external solver became
authoritative, or a rectangle became the source geometry.

## Sphere Carrier Slice

The executable harness lives at `smii.unwrap.external_competitors`.

Measured dependency-light candidates:

- `bt369`: DASHI-native equal-area inverse pullback plus BT369 cell receipt,
  seam/braid counts, trit histogram, and MDL-bounded depth.
- `equal_area`: Lambert cylindrical equal-area rectangle control.
- `equirect`: compatibility lat-long export with explicit area distortion.
- `cubed_sphere`: graphics-style six-face carrier sampled through a cube map.
- `octahedral`: compact folded-square direction-map carrier.

Optional or external candidates:

- `healpix`: scientific equal-area hierarchical spherical carrier. When
  `healpy` is installed, the harness measures a HEALPix reconstruction receipt;
  otherwise it reports `available=false` with a reason.
- mesh/UV competitors such as `xatlas`, `slim`, `bff`, `optcuts`, and
  `blender_unwrap`: registered as unavailable diagnostic receipts until a real
  adapter is bound.

## Adversarial Field Suite

`benchmark_adversarial_sphere_fields` runs the same declared competitor set
against deterministic synthetic fields:

- `constant`
- `linear_xyz`
- `low_frequency_harmonic`
- `high_frequency_harmonic`
- `polar_cap`
- `longitude_seam_stripe`
- `checkerboard_geodesic`
- `localized_gaussian_bump`
- `binary_hemisphere`
- `band_limited_mix`

The suite records a winner per field plus a winner histogram. This prevents a
single friendly smoke field from becoming the benchmark claim.

## Metrics

Every run receipt records the common score fields:

- `edge_length_residual`
- `area_residual`
- `angle_residual`
- `foldover_ratio`
- `residual_l2_area_weighted`
- `aggregate_score`
- `agreement_depth`
- `agreement_distance`
- `seam_length`
- `chart_count`
- `packing_efficiency`
- `inverse_roundtrip_error`
- `field_reconstruction_error`
- `dart_pressure_score`
- `grain_alignment_score`
- `panel_internal_variance`
- `seam_on_high_strain_penalty`
- `manufacturability_score`

Sphere-only carriers set garment metrics to neutral diagnostic values. Garment
promotion must still pass the panel, dart/ease, grain, seam-risk, and
manufacturing receipts before any flattened artifact is production-promoted.

## Claim Boundary

BT369 is a certified adaptive atlas and serialization strategy. It is not a
proof that curved surfaces flatten isometrically. HEALPix may remain the better
scientific spherical-analysis carrier; BT369 is expected to matter when the
benchmark includes DASHI-native trit receipts, seam/braid ledgers,
MDL-bounded refinement, and garment/pattern transfer semantics.
