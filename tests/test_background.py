"""Tests for the bash tool's background parameter.

A background call returns at once with the task id, pid and log path; exited
tasks' records ride appended to later tool results, at most four per result; a
reply that ends the run with work still in flight is reminded and the run
continues; every exit path kills what still runs.
"""

import os, re, time, signal
import pytest
import agent as agent_module
from agent import Agent, load_config
from conftest import ScriptedModel, assistant, tool_call
from environment import Environment

def _agent(tmp_path, templates, *replies, step_limit=8, emit=None, compact=None):
    return Agent(model=ScriptedModel(*replies),
                 environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                                         kill_after_seconds=10, log_dir=str(tmp_path)),
                 templates=templates, step_limit=step_limit, emit=emit, compact=compact)

def _pid(tool_message):
    return int(re.search(r"pid (\d+),", tool_message["content"]).group(1))


def test_background_call_returns_at_once_and_record_rides_a_later_result(
    tmp_path, templates
):
    events = []
    agent = _agent(
        tmp_path, templates,
        assistant(tool_calls=[tool_call(1, command="sleep 0.3; echo bg-done",
                                        background=True)]),
        assistant(tool_calls=[tool_call(1, command="sleep 0.6; echo fg")]),
        assistant("done"),
        emit=events.append,
    )
    try:
        agent.run("task")
    finally:
        agent.environment.kill_running()

    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert tool_messages[0]["content"].startswith("Background task s-1-1 started: pid ")
    assert "tail --pid=" in tool_messages[0]["content"]
    observations = [e for e in events if e.get("type") == "observation"]
    assert observations[0]["id"] == "s-1-1"
    assert observations[0]["returncode"] is None
    assert observations[0]["background"] is True
    assert "fg" in tool_messages[1]["content"]
    assert "[background task s-1-1 exited: code 0 after" in tool_messages[1]["content"]
    assert "bg-done" in tool_messages[1]["content"]
    assert "background task s-1-1 exited" in observations[1]["output"]
    assert "background" not in observations[1]


def test_foreground_result_does_not_wait_for_a_running_task(tmp_path, templates):
    agent = _agent(
        tmp_path, templates,
        assistant(tool_calls=[
            tool_call(1, command="sleep 0.8; echo bg", background=True),
            tool_call(2, command="echo hi"),
        ]),
        assistant(tool_calls=[tool_call(1, command="sleep 1; echo later")]),
        assistant("done"),
    )
    try:
        agent.run("task")
    finally:
        agent.environment.kill_running()

    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert "hi" in tool_messages[1]["content"]
    assert "[background task" not in tool_messages[1]["content"]
    assert "later" in tool_messages[2]["content"]
    assert "[background task s-1-1 exited: code 0 after" in tool_messages[2]["content"]


class _ReplyFromResult:
    """A model stub whose second reply is built from the first reply's result."""

    def __init__(self):
        self.model_name = "openai/fake-model"
        self.last_prompt_tokens = 0
        self.last_cached_tokens = 0
        self._calls = 0

    def query(self, messages, tools=None):
        self._calls += 1
        if self._calls == 1:
            return assistant(tool_calls=[tool_call(
                1, command="sleep 0.5; echo done", background=True)])
        if self._calls == 2:
            pid, log = re.search(r"pid (\d+), log (\S+)\.",
                                 messages[-1]["content"]).groups()
            return assistant(tool_calls=[tool_call(
                1, command=f"timeout 5 tail --pid={pid} -f {log}")])
        return assistant("done")

    def usage(self):
        return {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}


def test_tail_waits_for_the_background_task(tmp_path, templates):
    agent = Agent(
        model=_ReplyFromResult(),
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                                kill_after_seconds=10, log_dir=str(tmp_path)),
        templates=templates, step_limit=8,
    )
    started = time.monotonic()
    try:
        agent.run("task")
    finally:
        agent.environment.kill_running()

    assert time.monotonic() - started < 4
    tail_result = [m for m in agent.messages if m.get("role") == "tool"][1]["content"]
    assert "done" in tail_result
    assert "[background task s-1-1 exited: code 0 after" in tail_result


def test_reply_ending_with_a_running_task_is_reminded(tmp_path, templates):
    agent = _agent(
        tmp_path, templates,
        assistant(tool_calls=[tool_call(1, command="sleep 1.5", background=True)]),
        assistant("early"),
        assistant(tool_calls=[tool_call(1, command="sleep 1.8; echo waited")]),
        assistant("late"),
    )
    try:
        result = agent.run("task")
    finally:
        agent.environment.kill_running()

    assert result["final_output"] == "late"
    assert result["n_steps"] == 4
    user_messages = [m for m in agent.messages if m.get("role") == "user"]
    assert any("background task s-1-1 is still running (pid " in m["content"]
               and " of 10 s; log " in m["content"] for m in user_messages)
    assert any(step.get("note") == "background pending" for step in result["steps"])
    pid = _pid([m for m in agent.messages if m.get("role") == "tool"][0])
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_non_boolean_background_is_rejected_without_starting(tmp_path, templates):
    agent = _agent(
        tmp_path, templates,
        assistant(tool_calls=[tool_call(1, command="echo x", background="yes")]),
        assistant("done"),
    )
    agent.run("task")

    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert "must be true or false" in tool_messages[0]["content"]
    assert agent.environment.running_background() == []
    assert agent.environment.finished_pending() == 0


def test_background_record_is_gated_and_bounded_on_its_own(tmp_path, templates,
                                                           monkeypatch):
    """Each record passes the marker gate and the observation cap on its own."""
    monkeypatch.setattr(agent_module, "CONTROL_MARKERS", ("FAKE-MARKER",))
    # The note names a marker only through its inert label; the fake marker has
    # no delimiters to strip, so the label is pinned for the assertion below.
    monkeypatch.setattr(agent_module, "_marker_label", lambda marker: "model-marker")
    agent = _agent(
        tmp_path, templates,
        assistant(tool_calls=[tool_call(1, command="echo FAKE-MARKER; echo tail",
                                        background=True)]),
        assistant(tool_calls=[tool_call(1, command="sleep 0.5; echo ok")]),
        assistant("done"),
    )
    try:
        agent.run("task")
    finally:
        agent.environment.kill_running()

    content = [m for m in agent.messages if m.get("role") == "tool"][1]["content"]
    assert "output withheld" in content
    output_section = content.rsplit("Output:\n", 1)[1]
    assert "FAKE-MARKER" not in output_section

    compact = dict(load_config()["compact"])
    compact["command_observation_chars"] = 200
    agent = _agent(
        tmp_path, templates,
        assistant(tool_calls=[tool_call(1, command="seq 1 400", background=True)]),
        assistant(tool_calls=[tool_call(1, command="sleep 0.5; echo ok")]),
        assistant("done"),
        compact=compact,
    )
    try:
        agent.run("task")
    finally:
        agent.environment.kill_running()

    content = [m for m in agent.messages if m.get("role") == "tool"][1]["content"]
    output_section = content.rsplit("Output:\n", 1)[1]
    assert "[... truncated:" in output_section
    assert len(output_section) < 300


def test_background_record_survives_a_context_reset(tmp_path, templates, monkeypatch):
    monkeypatch.setattr(
        Agent, "_maybe_reset",
        lambda self, step_idx: self._reset_context(step_idx, "threshold", 0)
        if step_idx == 2 else None,
    )
    events = []
    agent = _agent(
        tmp_path, templates,
        assistant(tool_calls=[tool_call(1, command="sleep 0.3; echo survived",
                                        background=True)]),
        assistant(tool_calls=[tool_call(1, command="sleep 0.6; echo ok")]),
        assistant("done"),
        emit=events.append,
    )
    try:
        agent.run("task")
    finally:
        agent.environment.kill_running()

    assert any(event.get("type") == "compact" for event in events)
    # The reset rebuilt the context, so the step-2 result is the only tool
    # message left in it; the exited task's record rides appended to it.
    content = [m for m in agent.messages if m.get("role") == "tool"][-1]["content"]
    assert "[background task s-1-1 exited: code 0 after" in content
    assert "survived" in content


def test_resumed_session_keeps_the_stored_background_handoff(tmp_path, monkeypatch,
                                                             task_file):
    """A state file cut after a background start resumes without repair work."""
    import json
    import typer
    from typer.testing import CliRunner

    import main as cli_main
    from agent import STATE_PROTOCOL
    from model import Model

    stored = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "",
         "tool_calls": [tool_call(1, command="sleep 5", background=True)]},
        {"role": "tool", "tool_call_id": "call-1",
         "content": "Background task s-1-1 started: pid 99999, log /tmp/none.log."},
    ]
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    (session_dir / "bg.json").write_text(
        json.dumps({"protocol": STATE_PROTOCOL, "messages": stored})
    )

    monkeypatch.setattr(Model, "query",
                        lambda self, messages, tools=None: assistant("resumed"))
    monkeypatch.setattr(
        Model, "usage", lambda self: {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}
    )

    app = typer.Typer()
    app.command()(cli_main.run)
    result = CliRunner().invoke(
        app,
        ["--task-file", task_file("turn two"), "--json", "--resume", "bg",
         "--cwd", str(tmp_path), "--session-dir", str(session_dir), "--steps", "2"],
    )

    assert result.exit_code == 0, result.output
    persisted = json.loads((session_dir / "bg.json").read_text())["messages"]
    assert persisted[:4] == stored


def test_step_limit_kills_the_running_background_task(tmp_path, templates):
    agent = _agent(
        tmp_path, templates,
        assistant(tool_calls=[tool_call(1, command="sleep 30", background=True)]),
        step_limit=1,
    )
    try:
        with pytest.raises(RuntimeError, match="Step limit"):
            agent.run("task")
    finally:
        agent.environment.kill_running()

    pid = _pid([m for m in agent.messages if m.get("role") == "tool"][0])
    deadline = time.monotonic() + 1
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        assert time.monotonic() < deadline, "task alive 1 s after the run ended"
        time.sleep(0.02)


def test_at_most_four_records_per_result(tmp_path, templates):
    agent = _agent(
        tmp_path, templates,
        assistant(tool_calls=[
            tool_call(n, command=f"sleep 0.4; echo {n}", background=True)
            for n in range(1, 7)
        ]),
        assistant(tool_calls=[tool_call(1, command="sleep 0.9; echo a")]),
        assistant(tool_calls=[tool_call(1, command="echo b")]),
        assistant("done"),
    )
    try:
        agent.run("task")
    finally:
        agent.environment.kill_running()

    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 8
    second = tool_messages[6]["content"]
    assert second.count("[background task") == 4
    assert re.findall(r"\[background task (s-1-\d) exited", second) == [
        "s-1-1", "s-1-2", "s-1-3", "s-1-4",
    ]
    third = tool_messages[7]["content"]
    assert "[background task s-1-5 exited" in third
    assert "[background task s-1-6 exited" in third
