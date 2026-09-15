import json

import typer
from typer.testing import CliRunner

import main as cli_main
import pytest

from agent import INTERRUPTED_TOOL_RESULT, STATE_PROTOCOL
from conftest import assistant, final_answer, tool_call
from environment import Environment
from model import Model


def _cli_app():
    app = typer.Typer()
    app.command()(cli_main.run)
    return app


def _json_lines(output):
    return [json.loads(line) for line in output.splitlines()]


def test_session_resume_persists_and_reloads_messages(tmp_path, monkeypatch, task_file):
    responses = iter([
        assistant(tool_calls=[tool_call(1, command="printf 'turn-one-output\\n'")]),
        final_answer("Turn one complete."),
        assistant(tool_calls=[tool_call(1, command="printf 'turn-two-output\\n'")]),
        final_answer("Turn two complete."),
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
            "--task-file",
            task_file("turn one"),
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
            "--task-file",
            task_file("turn two"),
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


def test_resuming_a_pre_protocol_session_is_refused(tmp_path, monkeypatch, task_file):
    """An old bash-block history would tell the model to answer with fenced commands."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    state_file = session_dir / "legacy.json"
    state_file.write_text(json.dumps([{"role": "user", "content": "old turn"}]))

    monkeypatch.setattr(Model, "query", lambda self, messages, tools=None: None)

    result = CliRunner().invoke(
        _cli_app(),
        ["--task-file", task_file("turn two"), "--json", "--resume", "legacy",
         "--cwd", str(tmp_path), "--session-dir", str(session_dir),
         "--steps", "2"],
    )

    assert result.exit_code != 0
    assert "older protocol" in _json_lines(result.stdout)[-1]["message"]


def test_state_file_holds_the_turn_so_far_before_each_risky_step(
    tmp_path, monkeypatch, task_file
):
    """Per-message persistence, observed from inside the run.

    A manual stop (SIGTERM) skips run()'s finally, so the file must already hold
    the turn so far when a model call or a command begins. The state file is
    read at the moment each stub is entered: after the run it would be complete
    under the old code too (the finally persisted it) and prove nothing.
    """
    session_dir = tmp_path / "sessions"
    snapshots = {}

    def state_messages():
        files = list(session_dir.glob("*.json")) if session_dir.exists() else []
        if len(files) != 1:
            return None
        return json.loads(files[0].read_text())["messages"]

    responses = iter([
        assistant(tool_calls=[tool_call(1, command="printf 'one\\n'")]),
        final_answer("Done."),
    ])

    def query(self, messages, tools=None):
        snapshots.setdefault("at_first_query", state_messages())
        return next(responses)

    def execute(self, command, step, call):
        snapshots["at_execute"] = state_messages()
        return {"output": "one\n", "returncode": 0, "log_path": "unused"}

    monkeypatch.setattr(Model, "query", query)
    monkeypatch.setattr(
        Model, "usage", lambda self: {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}
    )
    monkeypatch.setattr(Environment, "execute", execute)

    result = CliRunner().invoke(
        _cli_app(),
        ["--task-file", task_file("persist me"), "--json", "--cwd", str(tmp_path),
         "--session-dir", str(session_dir), "--steps", "4"],
    )

    assert result.exit_code == 0, result.output
    at_first_query = snapshots["at_first_query"]
    assert at_first_query is not None, "state file absent when the first model call began"
    assert at_first_query[-1]["role"] == "user"
    assert "persist me" in at_first_query[-1]["content"]
    at_execute = snapshots["at_execute"]
    assert at_execute is not None, "state file absent when the first command began"
    assert at_execute[-1]["role"] == "assistant"
    assert [call["id"] for call in at_execute[-1]["tool_calls"]] == ["call-1"]


def test_resume_answers_dangling_tool_calls_before_the_new_task(
    tmp_path, monkeypatch, task_file
):
    """A session killed between an assistant's tool calls and their results
    resumes with a protocol-valid history: every unanswered call gets an
    interruption placeholder, placed before the new task message."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    cut_short = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "",
         "tool_calls": [tool_call(1, command="printf a"), tool_call(2, command="printf b")]},
        {"role": "tool", "tool_call_id": "call-1", "content": "a"},
    ]
    (session_dir / "cut.json").write_text(
        json.dumps({"protocol": STATE_PROTOCOL, "messages": cut_short})
    )
    prompts = []

    def query(self, messages, tools=None):
        prompts.append([message.copy() for message in messages])
        return final_answer("Turn two complete.")

    monkeypatch.setattr(Model, "query", query)
    monkeypatch.setattr(
        Model, "usage", lambda self: {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}
    )

    result = CliRunner().invoke(
        _cli_app(),
        ["--task-file", task_file("turn two"), "--json", "--resume", "cut",
         "--cwd", str(tmp_path), "--session-dir", str(session_dir), "--steps", "2"],
    )

    assert result.exit_code == 0, result.output
    history = prompts[0]
    assert history[:4] == cut_short
    assert history[4] == {"role": "tool", "tool_call_id": "call-2",
                          "content": INTERRUPTED_TOOL_RESULT}
    assert history[5]["role"] == "user"
    assert "turn two" in history[5]["content"]
    persisted = json.loads((session_dir / "cut.json").read_text())["messages"]
    assert persisted[4] == history[4]


def test_resuming_a_missing_session_fails_loudly(tmp_path, monkeypatch, task_file):
    """--resume names an explicit target: a missing state file is an error that
    names the path, never a silent fresh session."""
    session_dir = tmp_path / "sessions"
    queries = []
    monkeypatch.setattr(
        Model, "query", lambda self, messages, tools=None: queries.append(messages)
    )

    result = CliRunner().invoke(
        _cli_app(),
        ["--task-file", task_file("turn two"), "--json", "--resume", "ghost",
         "--cwd", str(tmp_path), "--session-dir", str(session_dir), "--steps", "2"],
    )

    assert result.exit_code != 0
    error = _json_lines(result.stdout)[-1]
    assert error["type"] == "error"
    assert str(session_dir / "ghost.json") in error["message"]
    assert queries == []
    assert not (session_dir / "ghost.json").exists()
