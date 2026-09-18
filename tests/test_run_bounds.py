"""CLI-level wiring for the unattended-run bounds: log-dir lifecycle and the
SIGTERM and KeyboardInterrupt paths killing the running command. No network is
touched.
"""

import json
import os
import signal
import threading
import time

import pytest
import typer
from typer.testing import CliRunner

import main as cli_main
from conftest import assistant, tool_call
from environment import Environment
from model import Model


@pytest.fixture(autouse=True)
def _restore_sigterm_disposition():
    """main.run installs a SIGTERM handler; give the test process its own back."""
    previous = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, previous)


def _cli_app():
    app = typer.Typer()
    app.command()(cli_main.run)
    return app


def _json_lines(output):
    return [json.loads(line) for line in output.splitlines()]


def _spy_environments(monkeypatch):
    """Record every Environment instance main.py creates, so tests can inspect it."""
    created = []
    original_init = Environment.__init__

    def spy_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(Environment, "__init__", spy_init)
    return created


def test_session_log_dir_is_kept_after_a_successful_run(tmp_path, monkeypatch, task_file):
    """Command logs are never deleted: the transcript stubs point at them,
    so a clean exit retains the run's directory like a failed one does."""
    created = _spy_environments(monkeypatch)

    def query(self, messages, tools=None):
        return assistant("Nothing to do.")

    monkeypatch.setattr(Model, "query", query)

    result = CliRunner().invoke(
        _cli_app(),
        ["--task-file", task_file("finish"), "--cwd", str(tmp_path),
         "--session-dir", str(tmp_path / "sessions")],
    )

    assert result.exit_code == 0, result.output
    assert len(created) == 1
    assert os.path.exists(created[0].log_dir)


def _raise_model_exploded(self, messages, tools=None):
    raise ValueError("model exploded")


def test_log_dir_is_retained_and_path_printed_on_failure(tmp_path, monkeypatch, task_file):
    created = _spy_environments(monkeypatch)
    monkeypatch.setattr(Model, "query", _raise_model_exploded)

    result = CliRunner().invoke(
        _cli_app(),
        ["--task-file", task_file("fail"), "--cwd", str(tmp_path),
         "--session-dir", str(tmp_path / "sessions")],
    )

    assert result.exit_code != 0
    assert len(created) == 1
    log_dir = created[0].log_dir
    assert os.path.exists(log_dir)
    assert log_dir in result.stderr


def test_log_dir_retention_message_stays_off_the_json_stream(tmp_path, monkeypatch, task_file):
    """The forensic path print must not corrupt the NDJSON error event's shape."""
    created = _spy_environments(monkeypatch)
    monkeypatch.setattr(Model, "query", _raise_model_exploded)

    result = CliRunner().invoke(
        _cli_app(),
        [
            "--task-file", task_file("fail"), "--json", "--cwd", str(tmp_path),
            "--session-dir", str(tmp_path / "sessions"),
        ],
    )

    assert result.exit_code != 0
    log_dir = created[0].log_dir
    assert os.path.exists(log_dir)
    assert log_dir in result.stderr
    events = _json_lines(result.stdout)
    assert events[-1] == {"type": "error", "message": "model exploded"}


def _reaped_or_gone(pid, timeout=2):
    """True once `pid` (our child) is dead: reaped here, or already reaped elsewhere."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            waited, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return True
        if waited == pid:
            return True
        time.sleep(0.05)
    return False


def test_sigterm_kills_the_running_command_persists_state_and_exits_143(
    tmp_path, monkeypatch, task_file
):
    pid_file = tmp_path / "command.pid"

    def query(self, messages, tools=None):
        threading.Timer(0.5, os.kill, args=(os.getpid(), signal.SIGTERM)).start()
        return assistant(tool_calls=[
            tool_call(1, command=f"echo $$ > {pid_file}; exec sleep 30")])

    monkeypatch.setattr(Model, "query", query)

    result = CliRunner().invoke(
        _cli_app(),
        ["--task-file", task_file("wait"), "--cwd", str(tmp_path),
         "--session-dir", str(tmp_path / "sessions")],
    )

    assert result.exit_code == 143
    pid = int(pid_file.read_text())
    assert _reaped_or_gone(pid), "the running command survived the SIGTERM handler"
    state_files = list((tmp_path / "sessions").glob("*.json"))
    assert len(state_files) == 1
    messages = json.loads(state_files[0].read_text())["messages"]
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["tool_calls"][0]["id"] == "call-1"


def test_keyboard_interrupt_kills_the_running_command_via_the_cli_handler(
    tmp_path, monkeypatch, task_file
):
    kills = []
    monkeypatch.setattr(Environment, "kill_running", lambda self: kills.append(True))

    def raise_interrupt(self, messages, tools=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(Model, "query", raise_interrupt)

    result = CliRunner().invoke(
        _cli_app(),
        ["--task-file", task_file("do it"), "--cwd", str(tmp_path),
         "--session-dir", str(tmp_path / "sessions")],
    )

    assert result.exit_code != 0
    assert kills == [True]
