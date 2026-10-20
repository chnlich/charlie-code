#!/usr/bin/env bash
# Upgrade one dependency in uv.lock and prove the new set before it lands.
#
# The argument is a package name, optionally pinned as <package>==<version>.
# `uv lock --upgrade-package` re-resolves only that package; the test suite
# then runs under the locked versions, and scripts/check-locked-install.sh
# stages the locked install and proves its request path with an offline
# request. When all three pass, the uv.lock diff is ready to land as a commit
# of its own.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

package="${1:?usage: scripts/upgrade-dependency.sh <package>[==<version>]}"

uv lock --upgrade-package "$package"
uv run --locked --extra dev pytest -q
scripts/check-locked-install.sh

echo "uv.lock is ready for a commit of its own"
