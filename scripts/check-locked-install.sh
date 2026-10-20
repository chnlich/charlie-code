#!/usr/bin/env bash
# Prove that this checkout, installed under the committed uv.lock, produces a
# charlie-code whose request path works — without touching the live uv tool
# install.
#
# The lock exports to a constraints file; the checkout installs from it into a
# temporary UV_TOOL_DIR/UV_TOOL_BIN_DIR; the staged entry point then makes one
# offline request against 127.0.0.1:9, where nothing listens. A healthy
# install walks the whole request path, imports included, and fails only at
# the connect, so the output carries a JSON error event whose message contains
# "Connection error". A broken dependency set fails earlier with a different
# message — litellm 1.106.0 without mcp, the 2026-10-19 incident, reports
# "No module named 'mcp'". Anything other than the connection error prints the
# run output and exits non-zero.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

constraints="$tmp/constraints.txt"
uv export --locked --no-emit-project --no-hashes --no-dev -o "$constraints"

export UV_TOOL_DIR="$tmp/tools"
export UV_TOOL_BIN_DIR="$tmp/bin"
uv tool install --reinstall --constraints "$constraints" "$repo_root"

printf 'say hi\n' > "$tmp/task.md"

status=0
output="$("$UV_TOOL_BIN_DIR/charlie-code" \
    --json \
    --model openai/test \
    --api-base http://127.0.0.1:9 \
    --session-dir "$tmp" \
    --cwd "$tmp" \
    --task-file "$tmp/task.md" 2>&1)" || status=$?

if printf '%s\n' "$output" | python3 -c '
import json
import sys

for line in sys.stdin:
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if event.get("type") == "error" and "Connection error" in event.get("message", ""):
        sys.exit(0)
sys.exit(1)
'; then
    echo "staged install check passed: offline request failed with Connection error as expected"
else
    echo "staged install check FAILED: no JSON error event with 'Connection error' (exit $status)" >&2
    echo "----- charlie-code output -----" >&2
    printf '%s\n' "$output" >&2
    exit 1
fi
