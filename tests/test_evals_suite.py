"""Suite YAML loading + schema validation for evals/run.py."""

import textwrap

import pytest

from evals_helper import load_script

run = load_script("evals_run", "run.py")


def _make_suite(tmp_path, tasks):
    suite_dir = tmp_path / "dev"
    (suite_dir / "fixtures").mkdir(parents=True)
    (suite_dir / "grade").mkdir(parents=True)
    for spec in tasks:
        (suite_dir / "fixtures" / spec["fixture_name"]).mkdir()
        (suite_dir / "fixtures" / spec["fixture_name"] / "seed.txt").write_text("x")
        (suite_dir / "grade" / spec["grade_name"]).write_text("import sys; sys.exit(1)")
        (suite_dir / f"{spec['id']}.yaml").write_text(spec["yaml"])
    return suite_dir


def _full_yaml(tid):
    return textwrap.dedent(f"""
        id: {tid}
        prompt: "solve {tid}"
        fixture: fixtures/{tid}
        grade: grade/{tid}.py
        timeout_s: 30
        step_limit: 5
        tags: [pilot]
    """).strip()


def test_load_suite_returns_tasks_with_resolved_paths(tmp_path):
    suite_dir = _make_suite(tmp_path, [
        {"id": "t1", "fixture_name": "t1", "grade_name": "t1.py", "yaml": _full_yaml("t1")},
    ])
    tasks = run.load_suite(suite_dir)
    assert [t["id"] for t in tasks] == ["t1"]
    t = tasks[0]
    for field in run.REQUIRED_TASK_FIELDS:
        assert field in t
    assert t["tags"] == ["pilot"]
    assert t["_fixture"].is_dir()
    assert t["_grade"].is_file()


def test_load_suite_missing_field_is_rejected(tmp_path):
    bad = "id: t2\nprompt: x\nfixture: fixtures/t2\ngrade: grade/t2.py\n"
    suite_dir = _make_suite(tmp_path, [
        {"id": "t2", "fixture_name": "t2", "grade_name": "t2.py", "yaml": bad},
    ])
    with pytest.raises(SystemExit, match="missing required field"):
        run.load_suite(suite_dir)


def test_load_suite_missing_fixture_dir_is_rejected(tmp_path):
    suite_dir = _make_suite(tmp_path, [
        {"id": "t3", "fixture_name": "t3", "grade_name": "t3.py", "yaml": _full_yaml("t3")},
    ])
    import shutil
    shutil.rmtree(suite_dir / "fixtures" / "t3")
    with pytest.raises(SystemExit, match="fixture directory not found"):
        run.load_suite(suite_dir)


def test_load_suite_missing_grade_script_is_rejected(tmp_path):
    suite_dir = _make_suite(tmp_path, [
        {"id": "t4", "fixture_name": "t4", "grade_name": "t4.py", "yaml": _full_yaml("t4")},
    ])
    (suite_dir / "grade" / "t4.py").unlink()
    with pytest.raises(SystemExit, match="grade script not found"):
        run.load_suite(suite_dir)


def test_load_suite_empty_dir_is_rejected(tmp_path):
    suite_dir = tmp_path / "empty"
    suite_dir.mkdir()
    with pytest.raises(SystemExit, match="no task YAML files"):
        run.load_suite(suite_dir)
