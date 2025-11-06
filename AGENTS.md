# Repository Agent Guidelines

These instructions apply to the entire repository. Any nested `AGENTS.md` files may override or extend these rules for their respective subdirectories.

# CRITICAL
- DO NOT COMMIT BINARY FILES - this is against agent policy in most environments and will prevent your PR.
- If you wish to commit binary files, instead, ensure CLI commands exist to replicate, and the user will run these commands on their local machine.
- Add said CLI commands to be run to the TODO.md, and remove them when the appropriate files have been generated/commands run.

## Project Structure
- Python orchestration code lives in `src/`, with packages such as `smii`
- Shared utilities reside in `scripts/`; CLI entry points are under `smii/`.
- UI assets are in `ui/`.
- Legal corpora, fixtures, and sample payloads live in `data/` and `examples/`.
- Tests mirror the package layout within `tests/`, with reusable pieces in `tests/fixtures/` and seeded payloads in `tests/templates/`.
- Documentation (design notes, automation walkthroughs, deep dives) belongs in `docs/`.

## Tooling & Commands
- Create a virtual environment and install dependencies with `pip install -e .[dev,test]`.
- Run the full test suite using `pytest`; scope to individual modules when needed 
- Format code via `ruff format` and lint with `ruff check --fix`. Use `ruff check --select I` for import sorting.
- Validate typing with `mypy .`.
- Launch dashboards using [command here] and access the CLI with [command here]
## Coding Conventions
- Target Python 3.11 features while keeping modules import-safe for 3.10.
- Use 4-space indentation, snake_case for variables/functions, PascalCase for Pydantic models and FastAPI routers, and UPPER_SNAKE_CASE for constants.
- Prefer dependency injection over module-level globals and expose public APIs via explicit `__all__` definitions.
- Rely exclusively on Ruff for formatting to avoid style churn.

## Testing Expectations
- Place new tests under `tests/`, mirroring the source module path (`tests/path/to/test_module.py`).
- Name test files `test_<feature>.py` and use descriptive test function names.
- Employ Hypothesis for property-based scenarios and mock external I/O via `pytest-mock`.
- Run `pytest --maxfail=1 -q` before submitting changes, and capture regressions with fixtures where applicable.

## Commit & PR Standards
- Use imperative tense commit messages scoped to logical change sets.
- Ensure CI checks pass locally (`pytest`, `ruff check`, `ruff format --check`) before opening a PR.
- PR descriptions should summarize the change, link relevant issues, describe behaviour differences (with before/after context), and include updated screenshots for UI modifications.
- Document any intentionally skipped checks in the PR body.
