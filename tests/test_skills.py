"""Tests for Agent Skills catalog loading and prompt injection."""

from agent import COMPLETION_SENTINEL, load_config, render
from skills import load_skill_catalog


def _skill(root, dirname, body):
    path = root / dirname / "SKILL.md"
    path.parent.mkdir()
    path.write_text(body)
    return path


def test_catalog_contains_valid_skills_sorted_by_name(tmp_path):
    beta = _skill(
        tmp_path,
        "beta-dir",
        "---\nname: beta\ndescription: Beta skill\n---\n# Body\n",
    )
    alpha = _skill(
        tmp_path,
        "alpha-dir",
        "---\nname: alpha\ndescription: Alpha skill\n---\n# Body\n",
    )

    catalog = load_skill_catalog(str(tmp_path))

    assert "alpha: Alpha skill" in catalog
    assert "beta: Beta skill" in catalog
    assert str(alpha.resolve()) in catalog
    assert str(beta.resolve()) in catalog
    assert catalog.index("- alpha:") < catalog.index("- beta:")


def test_missing_root_and_empty_dir_return_empty(tmp_path):
    assert load_skill_catalog(str(tmp_path / "missing")) == ""
    assert load_skill_catalog(str(tmp_path)) == ""


def test_malformed_or_incomplete_skills_are_skipped_without_breaking_valid(tmp_path):
    valid = _skill(
        tmp_path,
        "valid",
        "---\nname: valid\ndescription: |\n  Multi-line\n  description\n---\n# Body\n\n---\n",
    )
    _skill(tmp_path, "plain", "# No frontmatter\n\ndescription: not metadata\n")
    _skill(tmp_path, "missing-desc", "---\nname: missing-desc\n---\n# Body\n")

    catalog = load_skill_catalog(str(tmp_path))

    assert "valid: Multi-line description" in catalog
    assert str(valid.resolve()) in catalog
    assert "plain" not in catalog
    assert "missing-desc" not in catalog


def test_system_template_injects_non_empty_catalog():
    config = load_config()
    catalog = "# Available skills - read a skill's full instructions before using it:\n- demo: Demo skill\n    /tmp/demo/SKILL.md"

    prompt = render(
        config["templates"]["system"],
        cwd="/repo",
        completion_sentinel=COMPLETION_SENTINEL,
        skills=catalog,
    )

    assert catalog in prompt


def test_system_template_has_no_skills_header_when_catalog_empty():
    config = load_config()

    prompt = render(
        config["templates"]["system"],
        cwd="/repo",
        completion_sentinel=COMPLETION_SENTINEL,
        skills="",
    )

    assert "Available skills" not in prompt
