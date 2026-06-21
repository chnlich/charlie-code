import json

import pytest
import typer
from typer.testing import CliRunner

import main as cli_main
from agent import COMPLETION_SENTINEL, Agent, load_config
from environment import Environment
from model import Model


def _cli_app():
    app = typer.Typer()
    app.command()(cli_main.run)
    return app


def _json_lines(output):
    return [json.loads(line) for line in output.splitlines()]


def _patch_model(monkeypatch, *responses, usage=None):
    replies = iter(responses)

    def query(self, messages):
        return next(replies)

    monkeypatch.setattr(Model, "query", query)
    monkeypatch.setattr(
        Model,
        "usage",
        lambda self: usage or {"n_calls": 1, "input_tokens": 2, "output_tokens": 3},
    )


class _ScriptedModel:
    def __init__(self, *responses):
        self._responses = iter(responses)

    def query(self, messages):
        return next(self._responses)

    def usage(self):
        return {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}


def test_json_happy_path_streams_events_and_result(tmp_path, monkeypatch):
    _patch_model(
        monkeypatch,
        f"Writing file.\n```bash\nprintf hi > out.txt && echo {COMPLETION_SENTINEL}\n```",
    )

    result = CliRunner().invoke(
        _cli_app(),
        ["write file", "--json", "--cwd", str(tmp_path), "--steps", "3"],
    )

    assert result.exit_code == 0
    events = _json_lines(result.stdout)
    assert [event["type"] for event in events] == [
        "thought",
        "command",
        "observation",
        "result",
    ]
    assert events[1]["id"] == events[2]["id"] == "s-1"
    assert events[1]["command"].startswith("printf hi > out.txt")
    assert events[2]["returncode"] == 0
    assert COMPLETION_SENTINEL in events[2]["output"]
    assert events[-1]["completed"] is True
    assert events[-1]["usage"] == {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}
    assert (tmp_path / "out.txt").read_text() == "hi"


def test_json_step_limit_emits_error_and_nonzero_exit(tmp_path, monkeypatch):
    _patch_model(monkeypatch, "Still working.\n```bash\necho not_done\n```")

    result = CliRunner().invoke(
        _cli_app(),
        ["never complete", "--json", "--cwd", str(tmp_path), "--steps", "1"],
    )

    assert result.exit_code != 0
    events = _json_lines(result.stdout)
    assert events[-1]["type"] == "error"
    assert "Step limit (1) exceeded" in events[-1]["message"]


def test_json_model_exception_emits_error_and_nonzero_exit(tmp_path, monkeypatch):
    def query(self, messages):
        raise ValueError("model exploded")

    monkeypatch.setattr(Model, "query", query)
    monkeypatch.setattr(Model, "usage", lambda self: {})

    result = CliRunner().invoke(
        _cli_app(),
        ["fail", "--json", "--cwd", str(tmp_path), "--steps", "3"],
    )

    assert result.exit_code != 0
    events = _json_lines(result.stdout)
    assert events == [{"type": "error", "message": "model exploded"}]
    assert "Traceback" not in result.stdout


def test_agent_emit_collects_per_step_events(tmp_path):
    events = []
    agent = Agent(
        model=_ScriptedModel(
            "Thinking without a command.",
            f"Writing file.\n```bash\nprintf hi > out.txt && echo {COMPLETION_SENTINEL}\n```",
        ),
        environment=Environment(cwd=str(tmp_path), timeout=10),
        templates=load_config()["templates"],
        step_limit=3,
        emit=events.append,
    )

    result = agent.run("write out.txt")

    assert result["completed"] is True
    assert [event["type"] for event in events] == [
        "thought",
        "thought",
        "command",
        "observation",
    ]
    assert events[0] == {
        "type": "thought",
        "step": 1,
        "text": "Thinking without a command.",
    }
    assert events[2]["step"] == events[3]["step"] == 2
    assert events[2]["id"] == events[3]["id"] == "s-2"
    assert events[3]["returncode"] == 0
    assert COMPLETION_SENTINEL in events[3]["output"]
    assert (tmp_path / "out.txt").read_text() == "hi"


def test_agent_without_emit_keeps_return_and_step_limit_behavior(tmp_path):
    success = Agent(
        model=_ScriptedModel(f"```bash\necho {COMPLETION_SENTINEL}\n```"),
        environment=Environment(cwd=str(tmp_path), timeout=10),
        templates=load_config()["templates"],
        step_limit=1,
        emit=None,
    ).run("finish")

    assert success["completed"] is True
    assert success["n_steps"] == 1
    assert success["usage"] == {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}

    agent = Agent(
        model=_ScriptedModel("No completion.\n```bash\necho not_done\n```"),
        environment=Environment(cwd=str(tmp_path), timeout=10),
        templates=load_config()["templates"],
        step_limit=1,
        emit=None,
    )
    with pytest.raises(RuntimeError, match="Step limit \\(1\\) exceeded"):
        agent.run("do not finish")
