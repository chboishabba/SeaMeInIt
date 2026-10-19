# External LLM / ChatGPT Collaboration Guidelines

This document defines the strict protocol for collaborating with ChatGPT / Custom GPT threads from within the SeaMeInIt repository.

---

## 1. Absolute Invariants

> [!CAUTION]
> **NEVER ATTACH TO OR STEAL THE USER'S BROWSER.**
> - Do **NOT** connect via Chrome DevTools Protocol (CDP) on port `9222` or any active Chrome window.
> - Do **NOT** launch browser automation tools (Selenium, Playwright, Puppeteer) targeting ChatGPT web endpoints.
> - Do **NOT** attempt DOM scraping or browser relay fallback.

### The Headless Rule (Sole Preferred)
Agents must **always** use the direct headless `curl_cffi` methods provided by the sibling repository `reverse-engineered-chatgpt` (`re_gpt`).
See the comprehensive CLI & MCP documentation in [`../../reverse-engineered-chatgpt/README.md`](../../reverse-engineered-chatgpt/README.md).

The headless `curl_cffi` transport is the **sole preferred method** because it is **significantly faster, requires no browser resources, and is vastly more reliable** than browser/DOM automation. Browser stealing or DOM fallback is strictly prohibited.

---

## 2. Tooling Options: CLI & MCP

### Option A: Headless CLI (Recommended for Scripts & Subagents)
Run from `../reverse-engineered-chatgpt` using the project virtual environment `../.venv/bin/python`:

```bash
../.venv/bin/python -m re_gpt.entrypoint \
  --send <CONVERSATION_ID> \
  --prompt "<YOUR_PROMPT>" \
  --write-transport async \
  --no-dom-fallback \
  --write-timeout 300 \
  --follow
```

#### Multi-line Prompts via File
When formulating complex mathematical specifications or implementation requests, write the prompt to a file and pass `--prompt-file`:

```bash
../.venv/bin/python -m re_gpt.entrypoint \
  --send <CONVERSATION_ID> \
  --prompt-file /path/to/prompt.txt \
  --write-transport async \
  --no-dom-fallback \
  --write-timeout 300 \
  --follow
```

#### Key Flags
- `--no-dom-fallback`: **Mandatory**. Prevents any attempt to fall back to browser/DOM automation.
- `--write-transport async`: Uses the high-performance async `curl_cffi` transport.
- `--write-timeout 300`: Allows reasoning models (e.g., o1/o3) sufficient time (up to 300 seconds) to deliberate before emitting initial Server-Sent Event (SSE) chunks.
- `--follow`: Streams the response chunks as they arrive.

---

### Option B: Native MCP Server
`reverse-engineered-chatgpt` includes a native FastMCP server:

```bash
../.venv/bin/python -m re_gpt.mcp_server
```

#### Available Mutation Tools
- `chatgpt_continue(conversation_id, prompt, attachments=None, follow=True, write_transport="async", dom_fallback=False)`
- `chatgpt_new(prompt, follow=True)`
- `chatgpt_branch(conversation_id, parent_node_id, prompt, follow=True)`
- `chatgpt_regenerate(conversation_id, user_node_id, follow=True)`

**Always pass `dom_fallback=False` and `write_transport="async"`.**

---

## 3. Delegation Protocol: Implementing via GitHub Connector

When delegating a major algorithmic or architecture implementation to ChatGPT:

1. **Check Thread ID**: Look up the relevant conversation ID in [`__context/convo_ids.md`](file:///home/c/Documents/code/ITIR-suite/SeaMeInIt/__context/convo_ids.md).
2. **Formulate Invariants & Criteria**:
   - Explicitly detail the physics/geometry requirements (e.g. ROM strain cost minimization vs. geometric developability).
   - Specify required test gates, promotion conditions, and formatting standards (Ruff, Python 3.10/3.11).
3. **Explicit Call to Action**:
   - Instruct ChatGPT: *"Please implement this directly in the repository using your GitHub connector (create a branch/PR or commit directly to the repository)."*
4. **Dispatch via CLI/MCP**:
   - Send the prompt with `--no-dom-fallback --write-transport async`.
5. **Local Referee Gate**:
   - Pull the created branch/PR (`git fetch origin`, `git checkout <branch>`).
   - Run verification locally:
     ```bash
     ../.venv/bin/python -m pytest
     ../.venv/bin/python -m ruff check
     ../.venv/bin/python -m ruff format --check
     ```
   - Verify numerical receipts (e.g. `gate7_promoted: 1`, `self_intersection_count: 0`).
   - Never merge unverified code.
