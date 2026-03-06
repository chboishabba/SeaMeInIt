#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONVO_FILE="$ROOT_DIR/__context/convo_ids.md"
OUT_DIR="$ROOT_DIR/__context/last_sync"
TS_UTC="$(date -u +"%Y%m%dT%H%M%SZ")"
OUT_FILE="$OUT_DIR/${TS_UTC}_smii_context_sync.txt"
DB_PATH="${SMII_CHAT_ARCHIVE_DB:-$HOME/.chat_archive.sqlite}"
DB_URI="file:${DB_PATH}?mode=ro&immutable=1"

if [[ ! -f "$CONVO_FILE" ]]; then
  echo "Missing $CONVO_FILE" >&2
  exit 1
fi

if [[ ! -f "$DB_PATH" ]]; then
  echo "Missing archive database at $DB_PATH" >&2
  exit 1
fi

if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "sqlite3 is required but not installed" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

{
  echo "# SeaMeInIt Context Sync"
  echo "Generated (UTC): $TS_UTC"
  echo "Source IDs: $CONVO_FILE"
  echo "Archive DB: $DB_PATH"
  echo
} > "$OUT_FILE"

while IFS='|' read -r _ raw_id raw_title raw_tail raw_notes _; do
  id="$(echo "${raw_id:-}" | xargs)"
  title="$(echo "${raw_title:-}" | xargs)"
  tail_lines="$(echo "${raw_tail:-}" | xargs)"
  notes="$(echo "${raw_notes:-}" | xargs)"

  if [[ -z "$id" || "$id" == "id" || "$id" == "---" ]]; then
    continue
  fi

  if [[ -z "$tail_lines" || ! "$tail_lines" =~ ^[0-9]+$ ]]; then
    tail_lines=30
  fi

  {
    echo "---"
    echo "Conversation ID: $id"
    echo "Title: ${title:-unknown}"
    echo "Notes: ${notes:-}"
  } >> "$OUT_FILE"

  sqlite3 -header -column "$DB_URI" "
    PRAGMA temp_store=MEMORY;
    SELECT
      MIN(ts) AS first_ts,
      MAX(ts) AS last_ts,
      COUNT(*) AS total_msgs,
      SUM(CASE WHEN role='user' THEN 1 ELSE 0 END) AS user_msgs,
      SUM(CASE WHEN role='assistant' THEN 1 ELSE 0 END) AS assistant_msgs
    FROM messages
    WHERE canonical_thread_id='${id}';
  " >> "$OUT_FILE"

  echo >> "$OUT_FILE"
  echo "First user message:" >> "$OUT_FILE"
  sqlite3 -header -column "$DB_URI" "
    PRAGMA temp_store=MEMORY;
    SELECT
      ts,
      substr(replace(replace(text,char(10),' '),char(13),' '),1,220) AS preview
    FROM messages
    WHERE canonical_thread_id='${id}'
      AND role='user'
      AND trim(ifnull(text,''))<>''
    ORDER BY ts ASC
    LIMIT 1;
  " >> "$OUT_FILE"

  echo >> "$OUT_FILE"
  echo "Latest assistant message:" >> "$OUT_FILE"
  sqlite3 -header -column "$DB_URI" "
    PRAGMA temp_store=MEMORY;
    SELECT
      ts,
      substr(replace(replace(text,char(10),' '),char(13),' '),1,320) AS preview
    FROM messages
    WHERE canonical_thread_id='${id}'
      AND role='assistant'
      AND trim(ifnull(text,''))<>''
    ORDER BY ts DESC
    LIMIT 1;
  " >> "$OUT_FILE"

  echo >> "$OUT_FILE"
  echo "Tail (${tail_lines} messages):" >> "$OUT_FILE"
  sqlite3 -header -column "$DB_URI" "
    PRAGMA temp_store=MEMORY;
    SELECT
      ts,
      role,
      substr(replace(replace(text,char(10),' '),char(13),' '),1,180) AS preview
    FROM messages
    WHERE canonical_thread_id='${id}'
    ORDER BY ts DESC
    LIMIT ${tail_lines};
  " >> "$OUT_FILE"

  echo >> "$OUT_FILE"
done < <(rg "^\|" "$CONVO_FILE")

printf 'Wrote %s\n' "$OUT_FILE"
