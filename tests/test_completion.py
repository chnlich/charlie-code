"""The completion sentinel: a run ends only when the model declares it.

Covers the eight mechanism cases: the reminder branch for tool-less replies
without the line (half-sentence, empty, bare line), lenient matching of the
line itself, counter reset on tool calls, the three-strikes failure and its
--json error event, the rendered system message, and session-file fidelity
across resume. All offline: the model is a scripted stand-in.
"""

import json

import pytest
import typer
from typer.testing import CliRunner

import main as cli_main
from agent import Agent, load_config
from conftest import ScriptedModel, assistant, final_answer, tool_call
from environment import Environment
from model import Model

SENTINEL = load_config()["agent"]["completion_sentinel"]


def _cli_app():
    app = typer.Typer()
    app.command()(cli_main.run)
    return app


def _agent(tmp_path, templates, *replies, step_limit=5, emit=None, state_file=None,
           resume=False):
    return Agent(
        model=ScriptedModel(*replies),
        environment=Environment(cwd=str(tmp_path), timeout=10),
        templates=templates,
        step_limit=step_limit,
        emit=emit,
        state_file=state_file,
        resume=resume,
    )


def test_half_sentence_is_reminded_then_a_closed_reply_completes(tmp_path, templates):
    """Case 1: no closing line keeps the session open; the reminder enters
    history; the next closed reply completes with the line stripped everywhere."""
    events = []
    agent = _agent(
        tmp_path, templates,
        assistant("The tests pass and the docstring"),
        final_answer("The tests pass and the docstring is fixed."),
        emit=events.append,
    )

    result = agent.run("finish the docstring")

    assert result["completed"] is True
    assert result["n_steps"] == 2
    assert result["steps"][0]["note"] == "unfinished reply"
    assert result["final_output"] == "The tests pass and the docstring is fixed."
    reminder = agent.messages[-2]
    assert reminder["role"] == "user"
    assert SENTINEL in reminder["content"]
    assert [e["text"] for e in events if e["type"] == "thought"] == [
        "The tests pass and the docstring",
        "The tests pass and the docstring is fixed.",
    ]
    assert SENTINEL not in result["final_output"]
    assert all(SENTINEL not in e["text"] for e in events if e["type"] == "thought")


@pytest.mark.parametrize("wrapped", ["`{}`", "**{}**", "*{}*"])
def test_marked_up_completion_line_still_completes(tmp_path, templates, wrapped):
    """Case 2: emphasis around the line is tolerated, the answer above is not."""
    reply = assistant("Done.\n" + wrapped.format(SENTINEL))

    result = _agent(tmp_path, templates, reply).run("finish")

    assert result["completed"] is True
    assert result["final_output"] == "Done."


def test_tool_calls_alongside_the_line_run_and_reset_the_counter(tmp_path, templates):
    """Case 3: a reply may both call tools and carry the line; the tools run and
    the strike counter resets, so later stray replies do not compound."""
    agent = _agent(
        tmp_path, templates,
        assistant("half a sentence, no line yet"),
        final_answer("Running the checks.",
                     tool_calls=[tool_call(1, command="echo ran")]),
        assistant("another stray half-sentence"),
        assistant("and one more without the line"),
        final_answer("All checks ran."),
    )

    result = agent.run("run the checks then finish")

    assert result["completed"] is True
    assert result["final_output"] == "All checks ran."
    tool_messages = [m for m in agent.messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert "ran" in tool_messages[0]["content"]


def test_three_consecutive_unfinished_replies_raise(tmp_path, templates):
    """Case 4, agent level: three strikes end the run loud."""
    agent = _agent(
        tmp_path, templates,
        assistant("one"),
        assistant("two"),
        assistant("three"),
    )

    with pytest.raises(RuntimeError, match="3 consecutive replies"):
        agent.run("never completes")


def test_three_strikes_error_event_under_json(tmp_path, monkeypatch, task_file):
    """Case 4, CLI level: the last --json event is the error event, verbatim."""
    replies = iter([assistant("one"), assistant("two"), assistant("three")])
    monkeypatch.setattr(
        Model, "query", lambda self, messages, tools=None: next(replies))
    monkeypatch.setattr(Model, "usage", lambda self: {})

    result = CliRunner().invoke(
        _cli_app(),
        ["--task-file", task_file("never completes"), "--json",
         "--cwd", str(tmp_path), "--session-dir", str(tmp_path / "sessions")],
    )

    assert result.exit_code != 0
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1] == {
        "type": "error",
        "message": "Step 3: 3 consecutive replies without a tool call or the "
                   "completion line; giving up.",
    }


def test_empty_reply_takes_the_same_reminder_branch(tmp_path, templates):
    """Case 5: no text and no tool call is one unfinished reply, not a special case."""
    agent = _agent(tmp_path, templates, assistant(""), final_answer("done now"))

    result = agent.run("say something")

    assert result["completed"] is True
    assert result["steps"][0]["note"] == "unfinished reply"
    reminder = agent.messages[-2]
    assert reminder["role"] == "user"
    assert SENTINEL in reminder["content"]


def test_system_message_renders_the_completion_line(tmp_path, templates):
    """Case 6: the model can only learn the line from the rendered system message."""
    agent = _agent(tmp_path, templates, final_answer("done"), step_limit=1)

    agent.run("finish")

    assert SENTINEL in agent.messages[0]["content"]


def test_session_file_keeps_the_line_and_resume_continues(tmp_path, templates):
    """Case 7: the stored assistant message keeps the line verbatim (Kimi K3
    replays messages as-is); a resumed run picks the session up normally."""
    state_file = tmp_path / "session.json"
    first = _agent(tmp_path, templates, final_answer("Turn one answer."),
                   state_file=str(state_file))
    first.run("turn one")

    stored = [m for m in first.messages if m["role"] == "assistant"]
    assert stored == [{"role": "assistant",
                       "content": f"Turn one answer.\n{SENTINEL}"}]
    assert json.loads(state_file.read_text())["messages"] == first.messages

    resumed = _agent(tmp_path, templates, final_answer("Turn two answer."),
                     state_file=str(state_file), resume=True)
    result = resumed.run("turn two")

    assert result["completed"] is True
    assert result["final_output"] == "Turn two answer."
    assert any(m.get("role") == "assistant"
               and m["content"] == f"Turn one answer.\n{SENTINEL}"
               for m in resumed.messages)


def test_bare_completion_line_without_an_answer_is_unfinished(tmp_path, templates):
    """Case 8: the line alone is not an answer; the same reminder branch runs."""
    agent = _agent(tmp_path, templates, assistant(SENTINEL),
                   final_answer("the real answer"))

    result = agent.run("finish properly")

    assert result["completed"] is True
    assert result["n_steps"] == 2
    assert result["steps"][0]["note"] == "unfinished reply"
    assert result["final_output"] == "the real answer"
    reminder = agent.messages[-2]
    assert reminder["role"] == "user"
    assert SENTINEL in reminder["content"]
