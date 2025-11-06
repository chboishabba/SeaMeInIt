# Contributing to SeaMeInIt

We welcome contributions that advance SensibLaw's mission. Before you begin, please review the repository guidelines in [`AGENTS.md`](AGENTS.md) for expectations around project structure, tooling, and review standards.

## Getting Started
1. Create a virtual environment compatible with Python 3.11.
2. Install development and testing dependencies:
   ```bash
   pip install -e .[dev,test]
   ```
3. Verify your setup by running the full test suite:
   ```bash
   pytest
   ```

If you only need the testing extras, you can install them with:
```bash
pip install -e .[test]
```
This includes [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing.

## Development Workflow
- Format code with `ruff format` and lint via `ruff check --fix`. Run `ruff check --select I` to enforce import ordering when necessary.
- Validate typing coverage with `mypy .`.
- When working on Streamlit dashboards, launch them locally using `streamlit run streamlit_app.py`.
- Explore CLI features with `python -m sensiblaw.cli --help`.

## Writing Tests
New features and bug fixes must include automated tests.

- Mirror source paths in the `tests/` directory and name modules `test_<feature>.py`.
- Use descriptive test function names such as `test_handles_multi_column_toc`.
- Leverage fixtures from `tests/fixtures/` and templates from `tests/templates/` where possible.
- Hypothesis is available for property-based coverage, and external I/O should be patched using `pytest-mock`.

A helper script exists to scaffold a test module:
```bash
python scripts/gen_test.py src/path/to/module.py
```
This generates a file like `tests/path/to/test_module.py` with placeholder fixtures and a skipped test. Replace the TODOs and remove `@pytest.mark.skip` once real assertions are ready.

Before opening a pull request, run:
```bash
pytest --maxfail=1 -q
ruff format --check
ruff check
```
Address any failures locally. Document intentionally skipped checks in your PR description.

## Commit and Pull Request Process
- Use imperative commit messages (e.g., `Improve multi-column TOC parsing`) and keep each commit focused on a logical change set.
- Summarize changes clearly in the PR description, link related issues, and highlight behavior differences with before/after notes.
- Include updated screenshots for UI changes.
- Confirm all required checks pass locally prior to submission.

Thank you for contributing to SensibLaw and helping to improve the platform!
