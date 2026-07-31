"""Summary aggregation + fail_class mapping with stub episodes for evals/run.py."""

import json

import pytest

from evals_helper import load_script

run = load_script("evals_run", "run.py")


def _ndjson_result(n_steps=2, tokens_in=11, tokens_out=22):
    line = json.dumps({
        "type": "result", "completed": True, "n_steps": n_steps,
        "final_output": "done",
        "usage": {"n_calls": 1, "input_tokens": tokens_in, "output_tokens": tokens_out},
    })
    return '{"type": "session", "session_id": "s"}\n' + line


def _ndjson_error(message):
    return '{"type": "session", "session_id": "s"}\n' + json.dumps(
        {"type": "error", "message": message}
    )


def test_grade_outcome_resolved_when_grader_passes():
    rec = run.grade_outcome({"id": "t"}, _ndjson_result(), None, 1.5, 0, None)
    assert rec == {
        "resolved": True, "steps": 2, "tokens_in": 11, "tokens_out": 22,
        "wall_s": 1.5, "fail_class": None,
    }


def test_grade_outcome_wrong_answer_on_clean_grader_nonzero():
    rec = run.grade_outcome({"id": "t"}, _ndjson_result(), None, 2.0, 3, "wrong_answer")
    assert rec["resolved"] is False
    assert rec["fail_class"] == "wrong_answer"


def test_grade_outcome_step_limit_wins_over_grader_when_unresolved():
    rec = run.grade_outcome({"id": "t"}, _ndjson_error("Step limit (5) exceeded"),
                            "step_limit", 3.0, 1, "wrong_answer")
    assert rec["resolved"] is False
    assert rec["fail_class"] == "step_limit"
    assert rec["steps"] == 0  # no result event -> zero


def test_grade_outcome_env_error_when_model_error():
    rec = run.grade_outcome({"id": "t"}, _ndjson_error("model exploded"),
                            "env_error", 0.5, 1, "wrong_answer")
    assert rec["resolved"] is False
    assert rec["fail_class"] == "env_error"


def test_grade_outcome_infra_when_no_events():
    rec = run.grade_outcome({"id": "t"}, "", "infra", 0.0, None, "infra")
    assert rec["resolved"] is False
    assert rec["fail_class"] == "infra"


def test_grade_outcome_resolved_even_if_episode_hit_step_limit():
    # agent produced the right files before the loop gave up; grader still passes.
    rec = run.grade_outcome({"id": "t"}, _ndjson_error("Step limit (5) exceeded"),
                            "step_limit", 4.0, 0, None)
    assert rec["resolved"] is True
    assert rec["fail_class"] is None


def test_wilson_ci_extremes():
    assert run.wilson_ci(0, 10)[0] == 0.0
    assert run.wilson_ci(10, 10)[1] == 1.0
    lo, hi = run.wilson_ci(5, 10)
    assert 0.0 < lo < 0.5 < hi < 1.0
    mid = (lo + hi) / 2
    assert abs(mid - 0.5) < 0.02  # symmetric around 0.5 for 5/10


def test_aggregate_schema_and_rates():
    runs = [
        run.grade_outcome({"id": "t1"}, _ndjson_result(), None, 1.0, 0, None),
        run.grade_outcome({"id": "t1"}, _ndjson_result(), None, 2.0, 1, "wrong_answer"),
        run.grade_outcome({"id": "t2"}, _ndjson_error("model down"), "env_error", 0.1, 1, "wrong_answer"),
    ]
    summary = run.aggregate("glm52", "dev", 1, [("t1", runs[:2]), ("t2", runs[2:])])
    assert summary["model"] == "glm52"
    assert summary["suite"] == "dev"
    assert summary["k"] == 1
    assert summary["resolved"] == 1
    assert summary["total"] == 3
    assert summary["resolve_rate"] == round(1 / 3, 4)
    assert summary["wilson_ci95"] == run.wilson_ci(1, 3)
    assert [t["id"] for t in summary["per_task"]] == ["t1", "t2"]
    assert summary["per_task"][0]["resolve_frac"] == 0.5
    assert len(summary["per_task"][0]["runs"]) == 2
    assert summary["per_task"][1]["runs"][0]["fail_class"] == "env_error"


def test_run_grader_pass_and_fail(tmp_path):
    ok = tmp_path / "ok.py"; ok.write_text("import sys; sys.exit(0)")
    bad = tmp_path / "bad.py"; bad.write_text("import sys; sys.exit(2)")
    assert run.run_grader(ok, tmp_path, 10) == (0, None)
    code, fc = run.run_grader(bad, tmp_path, 10)
    assert code == 2 and fc == "wrong_answer"


def test_run_grader_timeout_is_infra(tmp_path):
    slow = tmp_path / "slow.py"
    slow.write_text("import time; time.sleep(5)")
    code, fc = run.run_grader(slow, tmp_path, 1)
    assert code is None and fc == "infra"


def test_run_grader_exception_exit_is_wrong_answer(tmp_path):
    crash = tmp_path / "crash.py"
    crash.write_text("raise RuntimeError('boom')")
    code, fc = run.run_grader(crash, tmp_path, 10)
    # a crashing grader subprocess exits non-zero; that is a clean "not resolved"
    assert code != 0 and fc == "wrong_answer"
