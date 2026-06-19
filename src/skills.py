"""Agent Skills catalog loading."""

import os
import re
import sys
from pathlib import Path

import yaml

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


def load_skill_catalog(root) -> str:
    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        return ""

    entries = []
    for path in Path(root).glob("*/SKILL.md"):
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

    if not entries:
        return ""

    lines = [
        "# Available skills - read a skill's full instructions before using it:",
        "#   cat <path>     (the skill's directory may also hold scripts/other files)",
        "",
    ]
    for name, description, path in sorted(entries):
        lines.append(f"- {name}: {description}")
        lines.append(f"    {path}")
    return "\n".join(lines)
