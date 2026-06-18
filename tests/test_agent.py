"""Tests for completion detection in the agent loop (model mocked, local only).

These cover the termination fix: a command and the completion sentinel emitted in
the SAME bash block must still run the command, and completion is detected from the
command's OUTPUT (last non-empty line == sentinel), not from the command text. No
network or SGLang endpoint is touched.
"""

from agent import COMPLETION_SENTINEL, Agent, is_completion, load_config
from environment import Environment
from model import Model


def _model(monkeypatch, *responses):
    """A real Model whose query() yields the given canned responses in order."""
    replies = iter(responses)
    model = Model(model_name="dummy", api_base="http://unused", api_key="EMPTY")
    monkeypatch.setattr(model, "query", lambda messages: next(replies))
    return model


class _ScriptedEnv:
    """Environment stub that returns queued results and records each command."""

    def __init__(self, *results):
        self.cwd = "/unused"
        self._results = iter(results)
        self.commands = []

    def execute(self, command):
        self.commands.append(command)
        return next(self._results)


def _agent(model, environment):
    return Agent(
        model=model,
        environment=environment,
        templates=load_config()["templates"],
        step_limit=40,
    )


def test_command_and_sentinel_in_one_block_still_runs_command(tmp_path, monkeypatch):
    model = _model(
        monkeypatch,
        f"Working.\n```bash\necho hi > f && echo {COMPLETION_SENTINEL}\n```",
    )
    agent = _agent(model, Environment(cwd=str(tmp_path), timeout=10))

    result = agent.run("write f then finish")

    assert result["completed"] is True
    assert result["n_steps"] == 1
    assert (tmp_path / "f").read_text().strip() == "hi"


def test_standalone_sentinel_completes(tmp_path, monkeypatch):
    model = _model(monkeypatch, f"Done.\n```bash\necho {COMPLETION_SENTINEL}\n```")
    agent = _agent(model, Environment(cwd=str(tmp_path), timeout=10))

    result = agent.run("just finish")

    assert result["completed"] is True
    assert result["n_steps"] == 1


def test_sentinel_not_on_last_line_does_not_stop(monkeypatch):
    # First command's output mentions the sentinel mid-output (e.g. `cat`/`grep` of
    # src/agent.py); the loop must continue and only stop on the real sentinel.
    false_stop = f'COMPLETION_SENTINEL = "{COMPLETION_SENTINEL}"\nmore code below\n'
    assert is_completion(false_stop) is False
    env = _ScriptedEnv(
        {"output": false_stop, "returncode": 0},
        {"output": f"{COMPLETION_SENTINEL}\n", "returncode": 0},
    )
    model = _model(
        monkeypatch,
        "Inspecting.\n```bash\ncat src/agent.py\n```",
        f"Done.\n```bash\necho {COMPLETION_SENTINEL}\n```",
    )
    agent = _agent(model, env)

    result = agent.run("inspect then finish")

    assert result["completed"] is True
    assert result["n_steps"] == 2  # step 1 did NOT terminate; loop proceeded
    assert env.commands == ["cat src/agent.py", f"echo {COMPLETION_SENTINEL}"]


def test_no_bash_block_triggers_reminder(tmp_path, monkeypatch):
    model = _model(
        monkeypatch,
        "I will start now.",  # no bash block -> reminder, no command
        f"```bash\necho {COMPLETION_SENTINEL}\n```",
    )
    agent = _agent(model, Environment(cwd=str(tmp_path), timeout=10))

    result = agent.run("finish eventually")

    assert result["completed"] is True
    assert result["steps"][0]["command"] is None
    assert result["steps"][0]["note"] == "no bash block"
    assert "No bash code block" in result["steps"][0]["observation"]


def test_multiple_blocks_executes_first_with_note(tmp_path, monkeypatch):
    model = _model(
        monkeypatch,
        "```bash\necho first\n```\n```bash\necho second\n```",
        f"```bash\necho {COMPLETION_SENTINEL}\n```",
    )
    agent = _agent(model, Environment(cwd=str(tmp_path), timeout=10))

    result = agent.run("two blocks then finish")

    step = result["steps"][0]
    assert step["command"] == "echo first"
    assert step["note"] == "2 bash blocks found; executed only the first."
    assert step["observation"].startswith("[2 bash blocks found")


def test_is_completion_matches_only_last_nonempty_line():
    assert is_completion(f"{COMPLETION_SENTINEL}\n") is True
    assert is_completion(f"{COMPLETION_SENTINEL}\n\n  \n") is True  # trailing blanks
    assert is_completion(f"work done\n{COMPLETION_SENTINEL}") is True
    assert is_completion(f"{COMPLETION_SENTINEL}\nmore output") is False
    assert is_completion(f"grep hit: {COMPLETION_SENTINEL} here") is False
    assert is_completion("") is False
