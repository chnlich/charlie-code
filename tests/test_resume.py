import json

import typer
from typer.testing import CliRunner

import main as cli_main
import pytest

from agent import STATE_PROTOCOL
from conftest import assistant, tool_call
from model import Model


def _cli_app():
    app = typer.Typer()
    app.command()(cli_main.run)
    return app


def _json_lines(output):
    return [json.loads(line) for line in output.splitlines()]


def test_session_resume_persists_and_reloads_messages(tmp_path, monkeypatch):
    responses = iter([
        assistant(tool_calls=[tool_call(1, command="printf 'turn-one-output\\n'")]),
        assistant("Turn one complete."),
        assistant(tool_calls=[tool_call(1, command="printf 'turn-two-output\\n'")]),
        assistant("Turn two complete."),
    ])
    captured_messages = []

    def query(self, messages, tools=None):
        captured_messages.append([message.copy() for message in messages])
        return next(responses)

    monkeypatch.setattr(Model, "query", query)
    monkeypatch.setattr(
        Model,
        "usage",
        lambda self: {"n_calls": 1, "input_tokens": 2, "output_tokens": 3},
    )

    session_dir = tmp_path / "sessions"
    runner = CliRunner()
    first = runner.invoke(
        _cli_app(),
        [
            "turn one",
            "--json",
            "--cwd",
            str(tmp_path),
            "--session-dir",
            str(session_dir),
            "--steps",
            "4",
        ],
    )

    assert first.exit_code == 0, first.output
    events = _json_lines(first.stdout)
    assert set(events[0]) == {"type", "session_id"}
    assert events[0]["type"] == "session"
    session_id = events[0]["session_id"]

    state_file = session_dir / f"{session_id}.json"
    state = json.loads(state_file.read_text())
    assert state["protocol"] == STATE_PROTOCOL
    messages = state["messages"]
    assert any(
        message["role"] == "assistant" and "Turn one complete" in message["content"]
        for message in messages
    )
    assert any(
        message["role"] == "tool" and "turn-one-output" in message["content"]
        for message in messages
    )

    second = runner.invoke(
        _cli_app(),
        [
            "turn two",
            "--resume",
            session_id,
            "--cwd",
            str(tmp_path),
            "--session-dir",
            str(session_dir),
            "--steps",
            "4",
        ],
    )

    assert second.exit_code == 0, second.output
    assert len(captured_messages) == 4
    resumed_messages = captured_messages[2]
    assert resumed_messages[-1]["role"] == "user"
    assert "turn two" in resumed_messages[-1]["content"]
    assert any(
        message["role"] == "assistant" and "Turn one complete" in message["content"]
        for message in resumed_messages[:-1]
    )
    assert any(
        message["role"] == "tool" and "turn-one-output" in message["content"]
        for message in resumed_messages[:-1]
    )


def test_resuming_a_pre_protocol_session_is_refused(tmp_path, monkeypatch):
    """An old bash-block history would tell the model to answer with fenced commands."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    state_file = session_dir / "legacy.json"
    state_file.write_text(json.dumps([{"role": "user", "content": "old turn"}]))

    monkeypatch.setattr(Model, "query", lambda self, messages, tools=None: None)

    result = CliRunner().invoke(
        _cli_app(),
        ["turn two", "--json", "--resume", "legacy", "--cwd", str(tmp_path),
         "--session-dir", str(session_dir), "--steps", "2"],
    )

    assert result.exit_code != 0
    assert "older protocol" in _json_lines(result.stdout)[-1]["message"]
