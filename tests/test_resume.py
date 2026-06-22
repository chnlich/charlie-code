import json

import typer
from typer.testing import CliRunner

import main as cli_main
from agent import COMPLETION_SENTINEL
from model import Model


def _cli_app():
    app = typer.Typer()
    app.command()(cli_main.run)
    return app


def _json_lines(output):
    return [json.loads(line) for line in output.splitlines()]


def test_session_resume_persists_and_reloads_messages(tmp_path, monkeypatch):
    responses = iter([
        (
            "Turn one complete.\n"
            f"```bash\nprintf 'turn-one-output\\n' && echo {COMPLETION_SENTINEL}\n```"
        ),
        (
            "Turn two complete.\n"
            f"```bash\nprintf 'turn-two-output\\n' && echo {COMPLETION_SENTINEL}\n```"
        ),
    ])
    captured_messages = []

    def query(self, messages):
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
            "2",
        ],
    )

    assert first.exit_code == 0, first.output
    events = _json_lines(first.stdout)
    assert set(events[0]) == {"type", "session_id"}
    assert events[0]["type"] == "session"
    session_id = events[0]["session_id"]

    state_file = session_dir / f"{session_id}.json"
    messages = json.loads(state_file.read_text())
    assert any(
        message["role"] == "assistant" and "Turn one complete" in message["content"]
        for message in messages
    )
    assert any(
        message["role"] == "user" and "turn-one-output" in message["content"]
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
            "2",
        ],
    )

    assert second.exit_code == 0, second.output
    assert len(captured_messages) == 2
    resumed_messages = captured_messages[1]
    assert resumed_messages[-1]["role"] == "user"
    assert "turn two" in resumed_messages[-1]["content"]
    assert any(
        message["role"] == "assistant" and "Turn one complete" in message["content"]
        for message in resumed_messages[:-1]
    )
    assert any(
        message["role"] == "user" and "turn-one-output" in message["content"]
        for message in resumed_messages[:-1]
    )
