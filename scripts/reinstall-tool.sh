#!/usr/bin/env bash
# Make a merged charlie-code change take effect on this host.
#
# The host runs charlie-code from a non-editable `uv tool` install of this
# checkout, so a merge to origin/main changes nothing until the checkout is
# fast-forwarded and the tool is reinstalled from it. This script is that
# recipe end to end. The committed uv.lock pins every dependency version: it
# exports to a constraints file, and the install runs under those constraints,
# so a reinstall never re-resolves dependencies to the day's latest releases.
# Before anything touches the live install, scripts/check-locked-install.sh
# stages the same locked install in a temporary directory and proves its
# request path with an offline request; the live install is replaced only
# after that check passes. The reinstalled entry point then runs --help, and
# its own interpreter imports an Environment that has kill_running and no
# sweep, plus a Model that carries the top_p/temperature sampling knobs. Any
# failing step exits non-zero; a checkout that cannot fast-forward stops here
# for a human.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

git fetch origin
git merge --ff-only origin/main

scripts/check-locked-install.sh

constraints="$(mktemp)"
trap 'rm -f "$constraints"' EXIT
uv export --locked --no-emit-project --no-hashes --no-dev -o "$constraints"
uv tool install --reinstall --constraints "$constraints" "$repo_root"

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
