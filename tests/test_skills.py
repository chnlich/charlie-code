"""Tests for Agent Skills catalog loading, repo-root discovery, and prompt injection.

Two levels feed the catalog: host roots (--skills-root / CHARLIE_CODE_SKILLS_ROOT /
config `skills.roots`) and the repo directories (`skills.repo_dirs`) under the git
worktree root that contains --cwd. All fixtures are temp directories; nothing here
depends on the skills installed on the host running the tests.
"""

import pytest
import typer
from typer.testing import CliRunner

import main as cli_main
from agent import load_config, render
from conftest import assistant
from model import Model
from skills import find_repo_root, load_skill_catalog


def _skill(root, dirname, body):
    path = root / dirname / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(body)
    return path


def _valid(name, description="Skill body"):
    return f"---\nname: {name}\ndescription: {description}\n---\n# Body\n"


def _names(catalog):
    return [line[2:].split(":", 1)[0] for line in catalog.splitlines() if line.startswith("- ")]


def _path_for(catalog, name):
    lines = catalog.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(f"- {name}:"):
            return lines[index + 1].strip()
    raise AssertionError(f"{name} not in catalog:\n{catalog}")


# --- catalog over one root (existing contract) ------------------------------------


def test_catalog_contains_valid_skills_sorted_by_name(tmp_path):
    beta = _skill(tmp_path, "beta-dir", _valid("beta", "Beta skill"))
    alpha = _skill(tmp_path, "alpha-dir", _valid("alpha", "Alpha skill"))

    catalog = load_skill_catalog([str(tmp_path)])

    assert "alpha: Alpha skill" in catalog
    assert "beta: Beta skill" in catalog
    assert str(alpha.resolve()) in catalog
    assert str(beta.resolve()) in catalog
    assert catalog.index("- alpha:") < catalog.index("- beta:")


def test_missing_root_and_empty_dir_and_no_roots_return_empty(tmp_path):
    assert load_skill_catalog([str(tmp_path / "missing")]) == ""
    assert load_skill_catalog([str(tmp_path)]) == ""
    assert load_skill_catalog([]) == ""


def test_malformed_or_incomplete_skills_are_skipped_without_breaking_valid(tmp_path):
    valid = _skill(
        tmp_path,
        "valid",
        "---\nname: valid\ndescription: |\n  Multi-line\n  description\n---\n# Body\n\n---\n",
    )
    _skill(tmp_path, "plain", "# No frontmatter\n\ndescription: not metadata\n")
    _skill(tmp_path, "missing-desc", "---\nname: missing-desc\n---\n# Body\n")

    catalog = load_skill_catalog([str(tmp_path)])

    assert "valid: Multi-line description" in catalog
    assert str(valid.resolve()) in catalog
    assert "plain" not in catalog
    assert "missing-desc" not in catalog


# --- catalog over several roots --------------------------------------------------


def test_roots_merge_in_order_and_sort_by_name(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    zeta = _skill(first, "zeta", _valid("zeta"))
    alpha = _skill(second, "alpha", _valid("alpha"))

    catalog = load_skill_catalog([str(first), str(second)])

    assert _names(catalog) == ["alpha", "zeta"]
    assert _path_for(catalog, "zeta") == str(zeta.resolve())
    assert _path_for(catalog, "alpha") == str(alpha.resolve())


def test_first_root_wins_a_name_collision(tmp_path):
    repo_dir = tmp_path / "repo" / ".claude" / "skills"
    host = tmp_path / "host"
    repo_tool = _skill(repo_dir, "tool", _valid("tool", "Repo flavour"))
    _skill(host, "tool", _valid("tool", "Host flavour"))

    catalog = load_skill_catalog([str(repo_dir), str(host)])

    assert _names(catalog) == ["tool"]
    assert "tool: Repo flavour" in catalog
    assert "Host flavour" not in catalog
    assert _path_for(catalog, "tool") == str(repo_tool.resolve())


def test_missing_roots_are_skipped_among_valid_ones(tmp_path):
    host = tmp_path / "host"
    only = _skill(host, "only", _valid("only"))

    catalog = load_skill_catalog([str(tmp_path / "absent-repo-dir"), str(host), str(tmp_path / "gone")])

    assert _names(catalog) == ["only"]
    assert _path_for(catalog, "only") == str(only.resolve())


# --- repo root discovery ---------------------------------------------------------


def _no_git_above(path):
    return not any((parent / ".git").exists() for parent in (path, *path.parents))


def test_find_repo_root_walks_up_from_a_subdirectory_to_the_git_directory(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    nested = repo / "pkg" / "sub"
    nested.mkdir(parents=True)

    assert find_repo_root(nested) == repo.resolve()
    assert find_repo_root(repo) == repo.resolve()


def test_find_repo_root_accepts_a_git_file_as_a_linked_worktree_does(tmp_path):
    worktree = tmp_path / "wt"
    (worktree / "src").mkdir(parents=True)
    (worktree / ".git").write_text("gitdir: /somewhere/.git/worktrees/wt\n")

    assert find_repo_root(worktree / "src") == worktree.resolve()


def test_find_repo_root_is_none_outside_any_repository(tmp_path):
    if not _no_git_above(tmp_path):
        pytest.skip("temp directory lives inside a git repository")
    plain = tmp_path / "plain" / "dir"
    plain.mkdir(parents=True)

    assert find_repo_root(plain) is None


# --- prompt template -------------------------------------------------------------


def test_system_template_injects_non_empty_catalog():
    config = load_config()
    catalog = "# Available skills - read a skill's full instructions before using it:\n- demo: Demo skill\n    /tmp/demo/SKILL.md"

    prompt = render(
        config["templates"]["system"],
        cwd="/repo",
        skills=catalog,
    )

    assert catalog in prompt


def test_system_template_has_no_skills_header_when_catalog_empty():
    config = load_config()

    prompt = render(
        config["templates"]["system"],
        cwd="/repo",
        skills="",
    )

    assert "Available skills" not in prompt


# --- through the CLI entry point -------------------------------------------------


def _cli_app():
    app = typer.Typer()
    app.command()(cli_main.run)
    return app


def _capture_model(monkeypatch):
    """One-reply fake Model that records every prompt it is shown."""
    seen = []

    def query(self, messages, tools=None):
        seen.append([dict(message) for message in messages])
        return assistant("task acknowledged")

    monkeypatch.setattr(Model, "query", query)
    monkeypatch.setattr(
        Model,
        "usage",
        lambda self: {"n_calls": 1, "input_tokens": 2, "output_tokens": 3},
    )
    return seen


def _run(tmp_path, cwd, extra_args, seen):
    task = tmp_path / "task.md"
    task.write_text("say hello\n", encoding="utf-8")
    result = CliRunner().invoke(
        _cli_app(),
        ["--task-file", str(task), "--json", "--cwd", str(cwd),
         "--session-dir", str(tmp_path / "sessions"), "--steps", "1", *extra_args],
    )
    assert result.exit_code == 0, result.output
    system = seen[0][0]
    assert system["role"] == "system"
    return _catalog_block(system["content"])


def _catalog_block(system_content):
    """The rendered {{skills}} block: from its header to the template text after it."""
    lines = system_content.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("# Available skills")]
    assert len(starts) == 1, system_content
    end = next(i for i in range(starts[0], len(lines)) if lines[i].startswith("Ending your turn"))
    return "\n".join(lines[starts[0]:end])


def _fixture_repo_and_host(tmp_path):
    """A git repo with both repo-level dirs, a subdirectory cwd, and a host root.

    `shared` exists at both levels (repo copy under .agents/skills); `repo-only` lives
    in the repo's .claude/skills; `host-only` lives in the host root.
    """
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    repo_only = _skill(repo / ".claude" / "skills", "repo-only", _valid("repo-only"))
    shared_repo = _skill(repo / ".agents" / "skills", "shared", _valid("shared", "Repo flavour"))
    host = tmp_path / "host"
    _skill(host, "shared", _valid("shared", "Host flavour"))
    host_only = _skill(host, "host-only", _valid("host-only"))
    cwd = repo / "pkg" / "sub"
    cwd.mkdir(parents=True)
    return repo, cwd, host, {"repo-only": repo_only, "shared": shared_repo, "host-only": host_only}


def test_cli_system_message_lists_repo_and_host_skills_with_repo_winning(tmp_path, monkeypatch):
    seen = _capture_model(monkeypatch)
    repo, cwd, host, paths = _fixture_repo_and_host(tmp_path)

    system = _run(tmp_path, cwd, ["--skills-root", str(host)], seen)

    assert _names(system) == ["host-only", "repo-only", "shared"]
    assert _path_for(system, "repo-only") == str(paths["repo-only"].resolve())
    assert _path_for(system, "host-only") == str(paths["host-only"].resolve())
    assert _path_for(system, "shared") == str(paths["shared"].resolve())
    assert "Host flavour" not in system


def test_cli_outside_a_repository_lists_host_skills_only(tmp_path, monkeypatch):
    if not _no_git_above(tmp_path):
        pytest.skip("temp directory lives inside a git repository")
    seen = _capture_model(monkeypatch)
    _repo, _cwd, host, _paths = _fixture_repo_and_host(tmp_path)
    plain = tmp_path / "plain"
    plain.mkdir()

    system = _run(tmp_path, plain, ["--skills-root", str(host)], seen)

    assert _names(system) == ["host-only", "shared"]
    assert "Host flavour" in system


def test_cli_environment_variable_supplies_the_host_root_and_the_flag_beats_it(tmp_path, monkeypatch):
    seen = _capture_model(monkeypatch)
    _repo, cwd, host, _paths = _fixture_repo_and_host(tmp_path)
    env_root = tmp_path / "env-root"
    _skill(env_root, "env-only", _valid("env-only"))
    monkeypatch.setenv("CHARLIE_CODE_SKILLS_ROOT", str(env_root))

    from_env = _run(tmp_path, cwd, [], seen)
    assert _names(from_env) == ["env-only", "repo-only", "shared"]

    seen.clear()
    from_flag = _run(tmp_path, cwd, ["--skills-root", str(host)], seen)
    assert _names(from_flag) == ["host-only", "repo-only", "shared"]
    assert "env-only" not in from_flag
