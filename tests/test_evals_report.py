"""Report CI math and HTML hygiene for evals/report.py."""

import json
import os

import pytest

from evals_helper import load_script

report = load_script("evals_report", "report.py")


def _summary(model, resolved, total, per_task=None):
    if per_task is None:
        per_task = [{"id": "t1", "resolve_frac": resolved / total,
                     "runs": [{"resolved": resolved == total, "steps": 1,
                               "tokens_in": 0, "tokens_out": 0,
                               "wall_s": 1.0,
                               "fail_class": None if resolved == total else "wrong_answer"}]}]
    return {
        "model": model, "suite": "dev", "resolved": resolved, "total": total,
        "resolve_rate": round(resolved / total, 4),
        "wilson_ci95": report.wilson_ci(resolved, total),
        "per_task": per_task,
    }


def test_wilson_ci_known_values():
    lo, hi = report.wilson_ci(0, 20)
    assert lo == 0.0
    assert 0 < hi < 0.2
    lo, hi = report.wilson_ci(20, 20)
    assert hi == 1.0
    assert 0.8 < lo < 1.0
    # 5/10 should be symmetric and span roughly 0.19..0.81
    lo, hi = report.wilson_ci(5, 10)
    assert abs((lo + hi) / 2 - 0.5) < 0.01
    assert lo < 0.25 < hi


def test_wilson_ci_zero_n_is_full_range():
    assert report.wilson_ci(0, 0) == [0.0, 1.0]


def test_render_html_contains_rates_and_model_ids(tmp_path):
    s = _summary("glm52", 7, 10)
    html_text = report.render_html([s])
    assert "glm52" in html_text
    assert "70.0%" in html_text
    assert "Wilson" in html_text or "95% CI" in html_text
    assert "<table" in html_text


def test_render_html_has_no_absolute_paths_or_usernames(tmp_path):
    s1 = _summary("glm52", 7, 10)
    s2 = _summary("kimi-k3", 3, 10)
    html_text = report.render_html([s1, s2])
    assert "/home/" not in html_text
    # the current user must not leak into the report
    for name in ("chaoli", "root", "Users"):
        assert name not in html_text
    assert "/v1/" not in html_text
    assert "http" not in html_text


def test_render_html_two_model_delta(tmp_path):
    s1 = _summary("glm52", 7, 10)
    s2 = _summary("kimi-k3", 4, 10)
    html_text = report.render_html([s1, s2])
    assert "Two-model delta" in html_text
    assert "delta" in html_text


def test_render_html_failure_ledger(tmp_path):
    runs = [
        {"resolved": True, "steps": 1, "tokens_in": 0, "tokens_out": 0, "wall_s": 1.0, "fail_class": None},
        {"resolved": False, "steps": 5, "tokens_in": 0, "tokens_out": 0, "wall_s": 1.0, "fail_class": "step_limit"},
        {"resolved": False, "steps": 3, "tokens_in": 0, "tokens_out": 0, "wall_s": 1.0, "fail_class": "wrong_answer"},
    ]
    s = _summary("glm52", 1, 3, [{"id": "t1", "resolve_frac": round(1 / 3, 4), "runs": runs}])
    html_text = report.render_html([s])
    assert "step_limit" in html_text
    assert "wrong_answer" in html_text


def test_report_cli_writes_file(tmp_path):
    s = _summary("glm52", 8, 10)
    src = tmp_path / "summary.json"
    src.write_text(json.dumps(s))
    out = tmp_path / "report.html"
    code = report.main([str(src), "-o", str(out)])
    assert code == 0
    text = out.read_text()
    assert "glm52" in text
    assert "/home/" not in text
