#!/usr/bin/env bash
# Make a merged charlie-code change take effect on this host.
#
# The host runs charlie-code from a non-editable `uv tool` install of this
# checkout, so a merge to origin/main changes nothing until the checkout is
# fast-forwarded and the tool is reinstalled from it. This script is that
# recipe end to end, with the checks that prove the installed copy is the new
# code: the reinstalled entry point runs, and its own interpreter imports an
# Environment that has kill_running and no sweep, plus a Model that carries
# the top_p/temperature sampling knobs. Any failing step exits non-zero; a
# checkout that cannot fast-forward stops here for a human.
#
# The reinstall goes through `uv tool upgrade`, which reads the existing
# install's receipt: the same source directory and the same dependency
# constraints the operator installed with, so a reinstall changes only this
# repository's own code and never re-resolves the dependency set.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

git fetch origin
git merge --ff-only origin/main

uv tool upgrade --reinstall charlie-code

charlie-code --help > /dev/null

"$(uv tool dir)/charlie-code/bin/python" - <<'PY'
from environment import Environment
from model import Model

assert hasattr(Environment, "kill_running"), "installed Environment lacks kill_running"
assert not hasattr(Environment, "sweep"), "installed Environment still has sweep"
model = Model("openai/test", "http://localhost", "key", 10, True,
              top_p=0.95, temperature=0.7)
assert (model.top_p, model.temperature) == (0.95, 0.7), (
    "installed Model lacks the top_p/temperature sampling knobs"
)
print("installed charlie-code: Environment has kill_running and no sweep;"
      " Model forwards top_p/temperature")
PY

echo "charlie-code reinstalled from $repo_root at $(git rev-parse --short HEAD)"
