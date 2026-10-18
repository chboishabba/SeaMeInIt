# Repository Agent Guidelines

These instructions apply to the entire repository. Any nested `AGENTS.md` files may override or extend these rules for their respective subdirectories.

# CRITICAL
- NEVER ATTACH TO OR STEAL THE USER'S BROWSER:
  - Do NOT connect to the user's desktop browser (no Chrome DevTools Protocol / CDP on port 9222 or any running Chrome instance).
  - Do NOT use DOM scraping, Selenium, Playwright, or browser automation to interact with ChatGPT.
  - ALWAYS use the headless `curl_cffi` methods provided by `reverse-engineered-chatgpt` (`re_gpt`). Never fall back to the DOM.
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
- Use the sibling ITIR virtual environment when present: `../.venv/bin/python`.
- Install dependencies with `../.venv/bin/python -m pip install -e .[dev,test]`.
- Run the full test suite using `../.venv/bin/python -m pytest`; scope to individual modules when needed.
- Format code via `../.venv/bin/python -m ruff format` and lint with `../.venv/bin/python -m ruff check --fix`. Use `../.venv/bin/python -m ruff check --select I` for import sorting.
- Validate typing with `../.venv/bin/python -m mypy .`.
- Launch dashboards using [command here] and access the CLI with [command here]

## External LLM / ChatGPT Collaboration (Strict curl_cffi Rule)
When consulting or delegating implementation tasks to ChatGPT / Custom GPT threads (see `__context/convo_ids.md` and `docs/chatgpt_collaboration_guidelines.md`):
- Refer to the full documentation in [`../reverse-engineered-chatgpt/README.md`](../reverse-engineered-chatgpt/README.md).
- **NEVER use browser automation, CDP (port 9222), or DOM fallback**. Agents must **ALWAYS** use the headless async `curl_cffi` methods.
- The `curl_cffi` transport is the **sole preferred method** because it is **significantly faster, far lower overhead, and vastly more reliable** than browser/DOM automation.
- **CLI Ingestion & Consultation** (run from `../reverse-engineered-chatgpt` using `../.venv/bin/python`):
  ```bash
  ../.venv/bin/python -m re_gpt.entrypoint \
    --send <CONVERSATION_ID> \
    --prompt-file <PATH_TO_PROMPT_FILE> \
    --write-transport async \
    --no-dom-fallback \
    --write-timeout 300 \
    --follow
  ```
  *(Or use `--prompt "<text>"` for short queries. Keep `--write-timeout 300` because reasoning models take >60s before streaming response chunks).*
- **MCP Server**:
  Use `re_gpt.mcp_server` (`python -m re_gpt.mcp_server`). When invoking `chatgpt_continue`, always specify `dom_fallback=False` and `write_transport="async"`.
- **Implementation Delegation Protocol**:
  1. Formulate clear requirements in the prompt.
  2. Tell ChatGPT explicitly: *"Please implement this directly in the repository using your GitHub connector (create a branch/PR or commit to the repo)."*
  3. Send via headless CLI/MCP with `--no-dom-fallback`.
  4. Fetch the resulting branch/PR locally via `git fetch origin`.
  5. Locally referee: run `../.venv/bin/python -m pytest`, `../.venv/bin/python -m ruff check`, and check receipts. Never merge unverified code.

## Coding Conventions
- Target Python 3.11 features while keeping modules import-safe for 3.10.
- Use 4-space indentation, snake_case for variables/functions, PascalCase for Pydantic models and FastAPI routers, and UPPER_SNAKE_CASE for constants.
- Prefer dependency injection over module-level globals and expose public APIs via explicit `__all__` definitions.
- Rely exclusively on Ruff for formatting to avoid style churn.

## Testing Expectations
- Place new tests under `tests/`, mirroring the source module path (`tests/path/to/test_module.py`).
- Name test files `test_<feature>.py` and use descriptive test function names.
- Employ Hypothesis for property-based scenarios and mock external I/O via `pytest-mock`.
- Run `../.venv/bin/python -m pytest --maxfail=1 -q` before submitting changes, and capture regressions with fixtures where applicable.

## Commit & PR Standards
- Use imperative tense commit messages scoped to logical change sets.
- Ensure CI checks pass locally (`../.venv/bin/python -m pytest`, `../.venv/bin/python -m ruff check`, `../.venv/bin/python -m ruff format --check`) before opening a PR.
- PR descriptions should summarize the change, link relevant issues, describe behaviour differences (with before/after context), and include updated screenshots for UI modifications.
- Document any intentionally skipped checks in the PR body.
