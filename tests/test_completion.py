"""Completion: a tool-less reply is the final answer, delivered as written.

Covers the four mechanism cases: a tool-less reply completes and reaches the
result, the event stream and the session file verbatim; an empty tool-less
reply raises at the agent level and under --json; a reply that
strip_leaked_reasoning reduces to empty takes the same error path; and
session-file fidelity across resume. All offline: the model is a scripted
stand-in.
"""

import json

import pytest
import typer
from typer.testing import CliRunner

import main as cli_main
from agent import Agent
from conftest import ScriptedModel, assistant
from environment import Environment
from model import Model, strip_leaked_reasoning


def _cli_app():
    app = typer.Typer()
    app.command()(cli_main.run)
    return app


def _agent(tmp_path, templates, *replies, step_limit=5, emit=None, state_file=None,
           resume=False):
    return Agent(
        model=ScriptedModel(*replies),
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                              kill_after_seconds=10, log_dir=str(tmp_path)),
        templates=templates,
        step_limit=step_limit,
        emit=emit,
        state_file=state_file,
        resume=resume,
    )


def test_tool_less_reply_completes_and_is_delivered_verbatim(tmp_path, templates):
    """A reply with no tool call ends the session: the text is the final output,
    the one thought event, and the stored assistant message, all verbatim."""
    events = []
    agent = _agent(tmp_path, templates, assistant("The task is done."),
                   emit=events.append)

    result = agent.run("finish the task")

    assert result["completed"] is True
    assert result["n_steps"] == 1
    assert result["final_output"] == "The task is done."
    assert [e for e in events if e["type"] == "thought"] == [
        {"type": "thought", "step": 1, "text": "The task is done."},
    ]
    stored = [m for m in agent.messages if m["role"] == "assistant"]
    assert stored == [{"role": "assistant", "content": "The task is done."}]


def test_empty_reply_raises_at_the_agent_level(tmp_path, templates):
    """No text and no tool call is an error: there is nothing to deliver."""
    agent = _agent(tmp_path, templates, assistant(""))

    with pytest.raises(RuntimeError, match="empty reply"):
        agent.run("say something")


def test_empty_reply_error_event_under_json(tmp_path, monkeypatch, task_file):
    """The last --json event is the error event, verbatim."""
    replies = iter([assistant("")])
    monkeypatch.setattr(
        Model, "query", lambda self, messages, tools=None: next(replies))
    monkeypatch.setattr(Model, "usage", lambda self: {})

    result = CliRunner().invoke(
        _cli_app(),
        ["--task-file", task_file("say something"), "--json",
         "--cwd", str(tmp_path), "--session-dir", str(tmp_path / "sessions")],
    )

    assert result.exit_code != 0
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1] == {
        "type": "error",
        "message": "Step 1: empty reply with no tool call; nothing to deliver.",
    }


def test_leaked_reasoning_only_reply_takes_the_same_error_path(tmp_path, templates):
    """Content that strip_leaked_reasoning reduces to empty leaves nothing to
    deliver, so it is the same error as an empty reply."""
    leaked = "<think>all reasoning, no answer</think>"
    assert strip_leaked_reasoning(leaked).strip() == ""

    agent = _agent(tmp_path, templates, assistant(leaked))

    with pytest.raises(RuntimeError, match="empty reply"):
        agent.run("say something")


def test_session_file_keeps_the_reply_verbatim_and_resume_continues(
        tmp_path, templates):
    """The stored assistant message is the reply verbatim; a resumed run picks
    the session up normally."""
    state_file = tmp_path / "session.json"
    first = _agent(tmp_path, templates, assistant("Turn one answer."),
                   state_file=str(state_file))
    first.run("turn one")

    stored = [m for m in first.messages if m["role"] == "assistant"]
    assert stored == [{"role": "assistant", "content": "Turn one answer."}]
    assert json.loads(state_file.read_text())["messages"] == first.messages

    resumed = _agent(tmp_path, templates, assistant("Turn two answer."),
                     state_file=str(state_file), resume=True)
    result = resumed.run("turn two")

    assert result["completed"] is True
    assert result["final_output"] == "Turn two answer."
    assert any(m.get("role") == "assistant"
               and m["content"] == "Turn one answer."
               for m in resumed.messages)

