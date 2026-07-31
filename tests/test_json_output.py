import json

import pytest
import typer
from typer.testing import CliRunner

import main as cli_main
from agent import Agent, load_config
from conftest import ScriptedModel, assistant, tool_call
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

    def query(self, messages, tools=None):
        return next(replies)

    monkeypatch.setattr(Model, "query", query)
    monkeypatch.setattr(
        Model,
        "usage",
        lambda self: usage or {"n_calls": 1, "input_tokens": 2, "output_tokens": 3},
    )


def test_json_happy_path_streams_events_and_result(tmp_path, monkeypatch):
    _patch_model(
        monkeypatch,
        assistant("Writing file.", tool_calls=[tool_call(1, command="printf hi > out.txt")]),
        assistant("Wrote out.txt."),
    )

    result = CliRunner().invoke(
        _cli_app(),
        [
            "write file",
            "--json",
            "--cwd",
            str(tmp_path),
            "--session-dir",
            str(tmp_path / "sessions"),
            "--steps",
            "3",
        ],
    )

    assert result.exit_code == 0
    events = _json_lines(result.stdout)
    assert [event["type"] for event in events] == [
        "session",
        "thought",
        "command",
        "observation",
        "thought",
        "result",
    ]
    assert set(events[0]) == {"type", "session_id"}
    assert events[2]["id"] == events[3]["id"]
    assert events[2]["command"] == "printf hi > out.txt"
    assert events[3]["returncode"] == 0
    assert events[-1]["completed"] is True
    assert events[-1]["final_output"] == "Wrote out.txt."
    assert events[-1]["usage"] == {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}
    assert (tmp_path / "out.txt").read_text() == "hi"


def test_json_step_limit_emits_error_and_nonzero_exit(tmp_path, monkeypatch):
    _patch_model(
        monkeypatch,
        assistant("Still working.", tool_calls=[tool_call(1, command="echo not_done")]),
    )

    result = CliRunner().invoke(
        _cli_app(),
        [
            "never complete",
            "--json",
            "--cwd",
            str(tmp_path),
            "--session-dir",
            str(tmp_path / "sessions"),
            "--steps",
            "1",
        ],
    )

    assert result.exit_code != 0
    events = _json_lines(result.stdout)
    assert events[0]["type"] == "session"
    assert events[-1]["type"] == "error"
    assert "Step limit (1) exceeded" in events[-1]["message"]


def test_json_model_exception_emits_error_and_nonzero_exit(tmp_path, monkeypatch):
    def query(self, messages, tools=None):
        raise ValueError("model exploded")

    monkeypatch.setattr(Model, "query", query)
    monkeypatch.setattr(Model, "usage", lambda self: {})

    result = CliRunner().invoke(
        _cli_app(),
        [
            "fail",
            "--json",
            "--cwd",
            str(tmp_path),
            "--session-dir",
            str(tmp_path / "sessions"),
            "--steps",
            "3",
        ],
    )

    assert result.exit_code != 0
    events = _json_lines(result.stdout)
    assert events[0]["type"] == "session"
    assert events[1] == {"type": "error", "message": "model exploded"}
    assert "Traceback" not in result.stdout


def test_agent_emit_collects_per_step_events(tmp_path):
    events = []
    agent = Agent(
        model=ScriptedModel(
            assistant(""),
            assistant("Writing file.",
                      tool_calls=[tool_call(1, command="printf hi > out.txt")]),
            assistant("Wrote out.txt."),
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
        "command",
        "observation",
        "thought",
    ]
    assert events[0] == {"type": "thought", "step": 2, "text": "Writing file."}
    assert events[1]["step"] == events[2]["step"] == 2
    assert events[1]["id"] == events[2]["id"]
    assert events[2]["returncode"] == 0
    assert (tmp_path / "out.txt").read_text() == "hi"


def test_agent_without_emit_keeps_return_and_step_limit_behavior(tmp_path):
    success = Agent(
        model=ScriptedModel(assistant("Nothing to do.")),
        environment=Environment(cwd=str(tmp_path), timeout=10),
        templates=load_config()["templates"],
        step_limit=1,
        emit=None,
    ).run("finish")

    assert success["completed"] is True
    assert success["n_steps"] == 1
    assert success["usage"] == {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}

    agent = Agent(
        model=ScriptedModel(
            assistant("No completion.",
                      tool_calls=[tool_call(1, command="echo not_done")]),
        ),
        environment=Environment(cwd=str(tmp_path), timeout=10),
        templates=load_config()["templates"],
        step_limit=1,
        emit=None,
    )
    with pytest.raises(RuntimeError, match="Step limit \\(1\\) exceeded"):
        agent.run("do not finish")
