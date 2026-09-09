"""Agent Skills catalog loading.

Two levels feed one catalog: host-level roots (user directories such as
~/.agents/skills and ~/.claude/skills) and repo-level directories (.claude/skills and
.agents/skills under the git worktree root that contains the working directory).
main.py orders the roots repo level first; the first root that supplies a skill name
wins, so a repo skill overrides a host skill of the same name, and a skill linked under
the same name into two host roots lists once.
"""

import os
import re
import sys
from pathlib import Path

import yaml

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


def find_repo_root(cwd):
    """The git worktree root containing *cwd*, or None when cwd is outside any repo.

    Walks up from cwd to the filesystem root and returns the first directory holding
    a `.git` entry. A linked worktree's `.git` is a file (a `gitdir:` pointer) and a
    primary checkout's is a directory; both count, so no git subprocess is needed.
    """
    path = Path(os.path.expanduser(str(cwd))).resolve()
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _scan_root(root):
    """(name, description, resolved SKILL.md path) per valid skill under *root*.

    A missing root is simply no skills (silent). A SKILL.md without frontmatter or
    without a description is skipped; invalid YAML gets one stderr warning and is
    skipped. Sorted by path so the merge in load_skill_catalog is deterministic.
    """
    root = Path(os.path.expanduser(str(root)))
    if not root.is_dir():
        return []
    entries = []
    for path in root.glob("*/SKILL.md"):
        text = path.read_text()
        match = _FRONTMATTER_RE.match(text)
        if not match:
            continue
        try:
            frontmatter = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            print(f"Skipping skill {path}: invalid frontmatter: {exc}", file=sys.stderr)
            continue
        if not isinstance(frontmatter, dict):
            continue
        description = frontmatter.get("description")
        if not description:
            continue
        name = frontmatter.get("name") or path.parent.name
        entries.append((name, " ".join(str(description).split()), str(path.resolve())))
    return sorted(entries, key=lambda entry: entry[2])


def load_skill_catalog(roots) -> str:
    """Catalog text for the system prompt, or "" when no root holds a valid skill.

    *roots* is an ordered list of directories. The first root that supplies a skill
    name wins; later roots' entries of the same name are dropped.
    """
    by_name = {}
    for root in roots:
        for name, description, path in _scan_root(root):
            by_name.setdefault(name, (name, description, path))

    if not by_name:
        return ""

    lines = [
        "# Available skills - read a skill's full instructions before using it:",
        "#   cat <path>     (the skill's directory may also hold scripts/other files)",
        "",
    ]
    for name, description, path in sorted(by_name.values()):
        lines.append(f"- {name}: {description}")
        lines.append(f"    {path}")
    return "\n".join(lines)
