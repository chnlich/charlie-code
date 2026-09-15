#!/usr/bin/env bash
# Make a merged charlie-code change take effect on this host.
#
# The host runs charlie-code from a non-editable `uv tool` install of this
# checkout, so a merge to origin/main changes nothing until the checkout is
# fast-forwarded and the tool is reinstalled from it. This script is that
# recipe end to end, with the checks that prove the installed copy is the new
# code: the reinstalled entry point runs, and its own interpreter imports an
# Environment that has kill_running and no sweep. Any failing step exits
# non-zero; a checkout that cannot fast-forward stops here for a human.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

git fetch origin
git merge --ff-only origin/main

uv tool install --reinstall "$repo_root"

charlie-code --help > /dev/null

"$(uv tool dir)/charlie-code/bin/python" - <<'PY'
from environment import Environment

assert hasattr(Environment, "kill_running"), "installed Environment lacks kill_running"
assert not hasattr(Environment, "sweep"), "installed Environment still has sweep"
print("installed charlie-code: Environment has kill_running and no sweep")
PY

echo "charlie-code reinstalled from $repo_root at $(git rev-parse --short HEAD)"
