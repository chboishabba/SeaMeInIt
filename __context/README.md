# SeaMeInIt Context

This folder is SeaMeInIt's local conversation context lane.

- `convo_ids.md` tracks SeaMeInIt-relevant conversation IDs from the local archive.
- `last_sync/` stores timestamped summaries and sync reports for those threads.
- `scripts/sync_smii_context_from_archive.sh` generates a fresh local report from `~/.chat_archive.sqlite`.

Default archive path:

- `~/.chat_archive.sqlite`

Override via:

- `SMII_CHAT_ARCHIVE_DB=/absolute/path/to/archive.sqlite`
