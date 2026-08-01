"""CLI-level wiring for the unattended-run bounds: --wall-seconds, log-dir lifecycle,
and the KeyboardInterrupt path sweeping the environment. No network is touched.
"""

import json
import os

import typer
from typer.testing import CliRunner

import main as cli_main
from environment import Environment
from model import Model


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


def test_wall_seconds_flag_overrides_the_config_default(tmp_path, monkeypatch):
    def query(self, messages, tools=None):
        raise AssertionError("model must not be queried once the wall budget is already spent")

    monkeypatch.setattr(Model, "query", query)

    result = CliRunner().invoke(
        _cli_app(),
        [
            "do it", "--json", "--cwd", str(tmp_path),
            "--session-dir", str(tmp_path / "sessions"),
            "--wall-seconds", "0",
        ],
    )

    assert result.exit_code != 0
    events = _json_lines(result.stdout)
    assert events[-1]["type"] == "error"
    assert "wall" in events[-1]["message"].lower()


def test_log_dir_is_removed_after_a_successful_run(tmp_path, monkeypatch):
    created = _spy_environments(monkeypatch)

    def query(self, messages, tools=None):
        return {"role": "assistant", "content": "Nothing to do."}, "stop"

    monkeypatch.setattr(Model, "query", query)

    result = CliRunner().invoke(
        _cli_app(),
        ["finish", "--cwd", str(tmp_path), "--session-dir", str(tmp_path / "sessions")],
    )

    assert result.exit_code == 0, result.output
    assert len(created) == 1
    assert not os.path.exists(created[0].log_dir)


def _raise_model_exploded(self, messages, tools=None):
    raise ValueError("model exploded")


def test_log_dir_is_retained_and_path_printed_on_failure(tmp_path, monkeypatch):
    created = _spy_environments(monkeypatch)
    monkeypatch.setattr(Model, "query", _raise_model_exploded)

    result = CliRunner().invoke(
        _cli_app(),
        ["fail", "--cwd", str(tmp_path), "--session-dir", str(tmp_path / "sessions")],
    )

    assert result.exit_code != 0
    assert len(created) == 1
    log_dir = created[0].log_dir
    assert os.path.exists(log_dir)
    assert log_dir in result.stderr


def test_log_dir_retention_message_stays_off_the_json_stream(tmp_path, monkeypatch):
    """The forensic path print must not corrupt the NDJSON error event's shape."""
    created = _spy_environments(monkeypatch)
    monkeypatch.setattr(Model, "query", _raise_model_exploded)

    result = CliRunner().invoke(
        _cli_app(),
        [
            "fail", "--json", "--cwd", str(tmp_path),
            "--session-dir", str(tmp_path / "sessions"),
        ],
    )

    assert result.exit_code != 0
    log_dir = created[0].log_dir
    assert os.path.exists(log_dir)
    assert log_dir in result.stderr
    events = _json_lines(result.stdout)
    assert events[-1] == {"type": "error", "message": "model exploded"}


def test_keyboard_interrupt_sweeps_the_environment_via_both_wired_call_sites(
    tmp_path, monkeypatch
):
    sweep_calls = []
    original_sweep = Environment.sweep

    def spy_sweep(self):
        sweep_calls.append(True)
        original_sweep(self)

    def raise_interrupt(self, messages, tools=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(Environment, "sweep", spy_sweep)
    monkeypatch.setattr(Model, "query", raise_interrupt)

    result = CliRunner().invoke(
        _cli_app(),
        ["do it", "--cwd", str(tmp_path), "--session-dir", str(tmp_path / "sessions")],
    )

    assert result.exit_code != 0
    # Agent.run's finally sweeps once; main.py's own interrupt handler sweeps again.
    assert len(sweep_calls) == 2
