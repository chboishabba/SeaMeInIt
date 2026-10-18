# SeaMeInIt Context

This folder is SeaMeInIt's local conversation context lane.

- `convo_ids.md` tracks SeaMeInIt-relevant conversation IDs from the local archive.
- `last_sync/` stores timestamped summaries and sync reports for those threads.
- `scripts/sync_smii_context_from_archive.sh` generates a fresh local report from `~/.chat_archive.sqlite`.

Default archive path:

- `~/.chat_archive.sqlite`

Override via:

- `SMII_CHAT_ARCHIVE_DB=/absolute/path/to/archive.sqlite`

## Interacting with Context Threads (CLI & MCP)

Agents should consult or continue these threads to delegate implementations or clarify architectural context. Detailed documentation lives in [`../../reverse-engineered-chatgpt/README.md`](../../reverse-engineered-chatgpt/README.md).

### CRITICAL: Strict Headless curl_cffi Only (Sole Preferred)
- **NEVER use browser automation, CDP (port 9222), or DOM fallback**.
- The headless async `curl_cffi` methods are the **sole preferred approach**: they are **significantly faster, lighter, and vastly more reliable** than browser/DOM interaction.
- Always execute with `--no-dom-fallback --write-transport async`.

### Method 1: Headless CLI
From `../reverse-engineered-chatgpt`:
```bash
../.venv/bin/python -m re_gpt.entrypoint \
  --send <CONVERSATION_ID> \
  --prompt "Your instruction here" \
  --write-transport async \
  --no-dom-fallback \
  --write-timeout 300 \
  --follow
```
*(Or use `--prompt-file /path/to/prompt.txt` for multi-line instructions).*

### Method 2: Native MCP Server
Run `python -m re_gpt.mcp_server`. Available tools:
- `chatgpt_continue(conversation_id, prompt, write_transport="async", dom_fallback=False, follow=True)`
- `chatgpt_new(prompt, follow=True)`
- `chatgpt_session_status()`

### Delegating Implementations to ChatGPT via GitHub Connector
1. Send prompt with exact mathematical invariants and test criteria.
2. Instruct ChatGPT to implement directly using its GitHub connector (creating a branch/PR).
3. Pull the branch locally (`git fetch origin`, `git checkout <branch>`).
4. Locally referee with `../.venv/bin/python -m pytest` and `ruff`. Never merge unverified code.

