# CLI reference and walkthrough

The SeaMeInIt command-line interface is available via `python -m smii ...` (or the
`smii` console script when installed). Use `--help` on any subcommand to see the
latest flags and defaults.

## Prerequisites

- Install the required extras: `pip install -e .[tools,vision]` for the download
  helper and MediaPipe detector, and add `jsonschema` if you plan to run the
  Afflec demo end to end.
- Provision SMPL-compatible assets before running commands that emit meshes:
  - `python tools/download_smplx.py --model smplerx --dest assets/smplerx` for
    the quickest non-commercial bundle.
  - `python tools/download_smplx.py --model smplx --dest assets/smplx` if you
    have the licensed SMPL-X archive.
- When using custom asset directories, ensure the folder contains the
  `manifest.json` produced by `tools/download_smplx.py` so the pipelines can
  locate the right model files.

## Hard-shell clearance (`interactive`)

Use this when you want to validate a rigid shell against a target soft-body mesh.
The command prompts for any missing arguments so you can run it with only a
subset of flags.

- **Inputs:** shell mesh (JSON/NPZ), target mesh (JSON/NPZ), optional pose JSON,
  and an optional samples-per-segment integer to insert additional interpolated
  poses between keyframes.
- **Outputs:** an `outputs/` subdirectory (auto-generated when not provided)
  containing the clearance reports produced by `smii.pipelines.run_clearance`.
- **Example:**
  ```bash
  python -m smii interactive \
    --shell exports/shell.json \
    --target outputs/avatar/subject_body.npz \
    --poses data/poses/example.json \
    --samples-per-segment 4
  ```

## Ben Afflec smoke test (`afflec-demo`)

A guided, reproducible way to exercise the full fitting and mesh generation
pipeline using the annotated Afflec fixtures under `tests/fixtures/afflec`.

- **Inputs:** optional custom images (file paths or directories); specify
  `--model-backend smplerx` for the quickest path or `--model-backend smplx` if
  you have licensed assets. Provide `--assets-root` to point to a custom asset
  folder containing a `manifest.json`.
- **Outputs:** the fitted parameter JSON (`afflec_measurement_fit.json`), a
  watertight SMPL-X mesh (`afflec_body.npz`), SMPL-X parameter payloads, and a
  measurement plot written to the chosen output directory (defaults to
  `outputs/afflec_demo`).
- **Example:**
  ```bash
  python -m smii afflec-demo \
    --model-backend smplerx \
    --assets-root assets/smplerx
  ```

## General image-to-avatar regression (`fit-from-images`)

Run this command to fit SMPL-X parameters from one or more RGB photos of a
subject. Directories are scanned recursively for supported image extensions so
you can point the CLI at a dataset folder.

- **Inputs:** `--images` pointing to files or directories, optional
  `--subject-id` for naming outputs, and SMPL asset settings via
  `--model-backend` and `--assets-root`. Leave measurement refinement enabled to
  incorporate anthropometric priors; use `--skip-measurement-refinement` when
  you want a pure detector-driven fit.
- **Outputs:** parameter JSON (`<subject>_smplx_params.json`) and a watertight
  mesh archive (`<subject>_smplx_body.npz`) written to `outputs/<subject>/` by
  default.
- **Example:**
  ```bash
  python -m smii fit-from-images \
    --images data/photo_front.jpg data/photo_side.jpg \
    --output outputs/alex \
    --subject-id alex \
    --model-backend smplerx
  ```
