"""Null-check control flow with always-fail and passing stub graders for evals/run.py."""

import textwrap

import pytest

from evals_helper import load_script

run = load_script("evals_run", "run.py")


def _make_suite(tmp_path, graders):
    """graders: {task_id: exit_code} stub graders."""
    suite_dir = tmp_path / "dev"
    (suite_dir / "fixtures").mkdir(parents=True)
    (suite_dir / "grade").mkdir(parents=True)
    for tid, code in graders.items():
        (suite_dir / "fixtures" / tid).mkdir()
        (suite_dir / "fixtures" / tid / "seed.txt").write_text("pristine")
        (suite_dir / "grade" / f"{tid}.py").write_text(f"import sys; sys.exit({code})")
        (suite_dir / f"{tid}.yaml").write_text(textwrap.dedent(f"""
            id: {tid}
            prompt: "solve {tid}"
            fixture: fixtures/{tid}
            grade: grade/{tid}.py
            timeout_s: 30
            step_limit: 5
            tags: [pilot]
        """).strip())
    return suite_dir


def test_null_check_all_fail_exits_zero(tmp_path, capsys):
    suite_dir = _make_suite(tmp_path, {"t1": 1, "t2": 2, "t3": 7})
    code = run.run_null_check(suite_dir, parallel=2)
    out = capsys.readouterr().out
    assert code == 0
    assert "OK" in out
    assert "UNRESOLVED" in out
    for tid in ("t1", "t2", "t3"):
        assert tid in out


def test_null_check_passing_grader_exits_nonzero(tmp_path, capsys):
    suite_dir = _make_suite(tmp_path, {"t1": 1, "t2": 0, "t3": 1})
    code = run.run_null_check(suite_dir, parallel=1)
    out = capsys.readouterr().out
    assert code == 1
    assert "FAILED" in out
    assert "t2" in out


def test_null_check_grader_runs_in_pristine_fixture(tmp_path):
    suite_dir = _make_suite(tmp_path, {"t1": 1})
    # grader asserts no agent-created file exists; pristine fixture only has seed.txt
    (suite_dir / "grade" / "t1.py").write_text(textwrap.dedent("""
        import os, sys
        if os.path.exists("answer.txt"):
            sys.exit(0)
        sys.exit(1)
    """).strip())
    code = run.run_null_check(suite_dir, parallel=1)
    assert code == 0
