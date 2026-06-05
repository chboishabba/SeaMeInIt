# Codebase Audit Findings - 2026-06-04

This audit used six parallel read-only lanes: source/runtime, tests/quality,
docs/roadmap, data/schema contracts, UI/export/reporting, and
dependency/security/privacy governance.

## FORMAL MODEL: O, R, C, S, L, P, G, F

- `O`: repo owners, agents, contributors, PR reviewers, users generating body,
  ROM, seam, panel, and manufacturing artifacts.
- `R`: make the roadmap clear, prioritized, and grounded in observed codebase
  risks and known bad cases.
- `C`: `README.md`, `TODO.md`, `ROADMAP.md`, `pyproject.toml`, receipt classes,
  scripts, tests, docs, generated-output policy, license/dependency surfaces.
- `S`: receipt gates are implemented, but quality gates are currently
  unreproducible, governance surfaces conflict, and several production claims
  still rely on bootstrap or diagnostic behavior.
- `L`: unclear backlog -> normalized roadmap -> runnable gates -> trusted
  receipts -> promoted production artifacts.
- `P`: promote the roadmap index and this audit page as the current planning
  entrypoint, then execute P0 remediations before deeper geometry work.
- `G`: no binary commits, passing documented local checks, explicit skipped
  checks, receipt promotion only from verified upstream artifacts.
- `F`: missing executable gates, missing strict receipt schemas, unresolved
  binary/license cleanup, and incomplete end-to-end lineage checks.

## P0 Findings

### Quality Gates Are Not Reproducible

`AGENTS.md` documents `pip install -e .[dev,test]`, `pytest`, Ruff, and mypy.
The first P0 patch added `dev` and `test` extras to `pyproject.toml` and aligned
`requirements-dev.txt` with them. The ambient Python 3.14 shell still should not
be treated as the authoritative runtime for this checkout.

The sibling ITIR venv is the intended runtime for this checkout:
`../.venv/bin/python -m pytest --maxfail=1 -q` passed with 333 tests and 1 skip
on 2026-06-04. That venv has mypy, but not Ruff.

Required remediation:

- keep `../.venv` documented as the expected local runtime,
- declare `jsonschema`, `pyyaml`, pytest plugins, Ruff, mypy, Hypothesis, and
  other required test/dev tools in the chosen environment surface,
- install or sync the sibling venv so `../.venv/bin/python -m ruff` works,
- align `requirements-dev.txt` with `pyproject.toml` or choose one source,
- add a CI workflow or document the external CI surface.

### Tracked Binaries Conflict With Repo Policy

The policy says not to commit binaries. The audit found tracked binary-like
files under `outputs/`, root Afflec images, `exports/test_dummy/dummy.fbx`,
`docs/rom_basis_validation.ipynb`, and fixture media. A local check found 95
tracked files matching common binary/generated extensions in this checkout.

Required remediation:

- decide which fixtures are intentional source assets,
- remove generated outputs from version control in a dedicated cleanup,
- add regeneration commands to `TODO.md` while outputs are absent,
- ignore `exports/` or move default export destinations under an ignored path.

### License Surface Is Contradictory

`LICENSE` and README do not describe the same grant and restrictions. This
blocks responsible reuse and PR review.

Required remediation:

- choose the actual project license,
- make README and `LICENSE` match,
- separate future licensing intent from the current legal grant.

## P1 Findings

### Receipt Consumption Is Weaker Than Receipt Emission

Emitter scripts enforce some promotion rules, hashes, and shape checks, but
loaded receipt consumers commonly check only `promotion == 1` and consumer
blocks. A malformed promoted receipt can look consumable if it was not produced
through the intended emitter.

Required remediation:

- add machine-readable receipt schemas or strict validators,
- reject non-finite metrics and loose hash fields,
- make the DAG reader verify hash/provenance chains across gates,
- add tests for malformed promoted receipts.

### Runtime Orchestration Is Still Script-Driven

`read_receipt_dag` is a passive reader, not a strong DAG authority. Current
CLIs mostly enforce their own local gates. This leaves gaps in whole-run
promotion reasoning.

Required remediation:

- clarify whether `run_afflec_receipted_demo.py` is only a demo runner or the
  first production runner,
- wire CLIs to a shared strict DAG validation surface before promotion,
- add a package script entry point instead of relying on `PYTHONPATH=src`
  commands for core workflows.

### Known Behavioral Bad Cases

- `scripts/solve_seams.py` accepts solver mode labels where the actual solve
  path does not yet differ for every advertised mode.
- `scripts/unwrap_panels.py` now has a real NumPy `lscm` path; ABF and ARAP
  remain pending and should not be advertised as implemented.
- `generate_undersuit()` can emit artifacts without a body receipt unless the
  caller opts into `require_body_receipt=True`.
- Manufacturing promotion is not yet equivalent to cutter-valid output.
- HTML/SVG report/export fields need broader escaping and ID sanitization.

### Security And Privacy Controls Are Mostly Aspirational

The repo documents sensitive body/image data handling, but enforcement is thin.
Asset download tooling can print or persist authenticated URLs, zip extraction
uses direct extraction, and local licensed body-model assets are present in the
working tree.

Required remediation:

- sanitize or omit tokens/query strings in asset logs and manifests,
- harden zip extraction like tar extraction,
- add privacy/retention controls before claiming cloud/body-data handling,
- keep licensed model assets untracked and guarded against accidental add.

## P2 Findings

- `docs/pipeline_runner.md` still documents `bbox` as the default for the
  mesh-first runner; production receipted runs default to MediaPipe.
- `ROADMAP.md` is useful historical material but currently competes with the
  production roadmap for “blocking gap” authority.
- `TODO.md` mixes completed status, historical notes, runbooks, and active next
  actions. Split or prefix it so the active queue is obvious.
- `docs/schemas.md` and schema files have small parity drift, such as
  measurement category coverage.

## Promotion Recommendation

Hold production promotion. Promote only the normalized roadmap/audit docs for
planning. The next implementation round should start with reproducible quality
gates and binary/license cleanup, then move to Gate 0 body trust and strict
receipt validation.
