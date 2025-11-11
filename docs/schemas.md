# Schema Conventions

The `schemas/body_unified.yaml` document defines the canonical data contract for
anthropometric assets that travel through SeaMeInIt. The schema follows
JSON Schema draft 2020-12 and is stored in YAML for readability.

## Naming

- **Identifiers** (landmarks, joints, measurements) are lowercase with
  hyphens, underscores, or periods (`^[a-z][a-z0-9_\-.]*$`).
- **Measurement categories** include `length`, `circumference`, `breadth`,
  `depth`, and `angle`.
- **Joint types** adhere to anatomical terminology (`ball_and_socket`,
  `hinge`, etc.).

## Coordinate Frames

Two right-handed frames are recognised:

- `smii_body_local` — origin at the pelvic centroid, +X anterior, +Y left,
  +Z superior.
- `smii_world` — global simulation frame, +X forward, +Y right, +Z up.

Landmarks and joints inherit the frame declared in `metadata.coordinate_frame`
unless they override it locally via `coordinate_frame`.

## Units

All numeric quantities are metric. Length-like units default to millimetres,
while angular quantities are given in degrees. Consumers should convert values
as required for downstream tooling.

## Consumption Pattern

The validator utilities in `src/schemas/validators.py` provide thin wrappers
around `jsonschema` that load and cache schema definitions. Pipelines should
call `validate_body_payload` before emitting data so that downstream modules
receive structurally consistent payloads. The `validate_file` helper is
available for batch ingestion of stored YAML/JSON assets.

## Mobility Constraints

The unified schema now exposes an optional `mobility_constraints` array for
capturing per-joint flexibility metadata gathered during functional
assessments. Each constraint references a joint identifier, specifies whether it
represents a `range_override`, `mobility_class`, or `stiffness_adjustment`, and
may include axis-specific bounds (`limit`) as well as qualitative
classifications. Consumers should treat these records as refinements layered on
top of canonical `motion_ranges`; for example, a subject with limited shoulder
external rotation can declare a tighter `max` bound, while spine flexibility
can be encoded as a `mobility_class` with a supporting reference to the
assessment protocol.

## Downstream Integrations

- **Fitting pipelines** assemble a record containing `metadata`,
  `anatomical_landmarks`, `measurements`, `joints`, `motion_ranges`, and any
  `mobility_constraints`, validating the result prior to persistence or
  message-bus publication.
- **Simulation loaders** ingest the same record to seed rig generation and
  enforce joint limits during runtime.
- **Analytics tooling** rely on the measurement definitions to map raw
  anthropometry to derived indicators and now have access to mobility
  confidence signals for additional filtering.
