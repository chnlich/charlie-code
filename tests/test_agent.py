"""The tool-calling protocol: what ends the loop, and what must never end it.

Completion is a whitelist — finish_reason `stop`, no tool calls, non-empty text — so
a truncated reply, which is shape-identical to a finished one, cannot end the run.
No network: the model is a scripted stand-in.
"""


import pytest

from agent import BASH_TOOL, Agent, gate_output, render
from conftest import ScriptedModel, assistant, tool_call
from environment import Environment


def _agent(tmp_path, templates, *replies, step_limit=5, emit=None):
    return Agent(
        model=ScriptedModel(*replies),
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                              kill_after_seconds=10, log_dir=str(tmp_path)),
        templates=templates,
        step_limit=step_limit,
        emit=emit,
    )


def test_stop_with_text_completes_and_returns_it(tmp_path, templates):
    result = _agent(tmp_path, templates, assistant("All done: 3 files.")).run("count")

    assert result["completed"] is True
    assert result["final_output"] == "All done: 3 files."
    assert result["n_steps"] == 1


@pytest.mark.parametrize(
    "finish_reason, match",
    [
        ("length", "truncated"),
        ("content_filter", "unexpected finish_reason"),
        ("tool_calls", "unexpected finish_reason"),
        (None, "unexpected finish_reason"),
        ("something_new", "unexpected finish_reason"),
    ],
)
def test_only_stop_may_complete(tmp_path, templates, finish_reason, match):
    """A reply that looks finished must not complete unless the envelope says stop."""
    looks_done = assistant("All done: 3 files.", finish_reason=finish_reason)

    with pytest.raises(RuntimeError, match=match):
        _agent(tmp_path, templates, looks_done).run("count")


def test_assistant_message_is_stored_verbatim(tmp_path, templates):
    """Kimi K3 needs the whole message back, reasoning_content and all."""
    message, finish_reason = assistant(
        "done", reasoning_content="I checked the directory listing."
    )
    agent = _agent(tmp_path, templates, (message, finish_reason))

    agent.run("count")

    stored = [m for m in agent.messages if m.get("role") == "assistant"]
    assert stored == [message]


def test_every_tool_call_gets_its_own_paired_result(tmp_path, templates):
    calls = [
        tool_call(1, command="echo one"),
        tool_call(2, command="echo two"),
        tool_call(3, command="echo three"),
    ]
    agent = _agent(
        tmp_path, templates,
        assistant(tool_calls=calls),
        assistant("ran all three"),
    )

    result = agent.run("run three")

    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == [c["id"] for c in calls]
    assert [step["command"] for step in result["steps"]] == [
        "echo one", "echo two", "echo three",
    ]
    for index, expected in enumerate(("one", "two", "three")):
        assert expected in tool_messages[index]["content"]


def test_tools_are_offered_on_every_call(tmp_path, templates):
    agent = _agent(tmp_path, templates, assistant("done"))

    agent.run("count")

    assert agent.model.seen_tools == [[BASH_TOOL]]


def test_output_carrying_control_markers_is_withheld(tmp_path, templates):
    """Markers reaching the transcript can be echoed back and re-parsed as a call."""
    agent = _agent(
        tmp_path, templates,
        assistant(tool_calls=[tool_call(1, command="printf '<|open|>tools<|sep|>'")]),
        assistant("read it another way"),
    )

    agent.run("cat the file")

    observation = [m for m in agent.messages if m.get("role") == "tool"][0]["content"]
    assert "<|open|>" not in observation
    assert "withheld" in observation
    assert "Exit code: 0" in observation


def test_clean_output_passes_through_untouched(tmp_path, templates):
    assert gate_output("alpha\nbravo") == ("alpha\nbravo", None)


def test_empty_stop_reply_raises_instead_of_completing(tmp_path, templates):
    agent = _agent(tmp_path, templates, assistant(""))

    with pytest.raises(RuntimeError, match="empty reply"):
        agent.run("say something")


@pytest.mark.parametrize(
    "call, expected",
    [
        (tool_call(1, name="python", code="1"), "no tool named"),
        ({"id": "x", "type": "function",
          "function": {"name": "bash", "arguments": "{not json"}}, "not valid JSON"),
        (tool_call(1, command="   "), "non-empty string"),
    ],
)
def test_malformed_calls_are_reported_not_raised(tmp_path, templates, call, expected):
    agent = _agent(tmp_path, templates, assistant(tool_calls=[call]), assistant("ok"))

    result = agent.run("misuse the tool")

    assert result["completed"] is True
    assert expected in result["steps"][0]["observation"]
    assert result["steps"][0]["note"] == "invalid tool call"


def test_step_limit_is_unchanged(tmp_path, templates):
    agent = _agent(
        tmp_path, templates,
        assistant(tool_calls=[tool_call(1, command="true")]),
        step_limit=1,
    )

    with pytest.raises(RuntimeError, match=r"Step limit \(1\) exceeded"):
        agent.run("never finish")


def test_progress_event_reaches_emit_with_the_command_id(tmp_path, templates):
    events = []
    agent = Agent(
        model=ScriptedModel(
            assistant("Waiting.", tool_calls=[tool_call(1, command="sleep 1")]),
            assistant("done"),
        ),
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[0.2],
                                kill_after_seconds=10, log_dir=str(tmp_path)),
        templates=templates,
        step_limit=3,
        emit=events.append,
    )

    agent.run("wait a bit")

    command = next(event for event in events if event["type"] == "command")
    same_call = [event["type"] for event in events if event.get("id") == command["id"]]
    assert same_call == ["command", "command_progress", "observation"]
    progress = next(event for event in events if event["type"] == "command_progress")
    assert progress["step"] == command["step"]
    assert progress["elapsed_seconds"] == 0.2
    assert progress["killed"] is False


def test_system_message_names_the_kill_cap_in_minutes(tmp_path, templates):
    agent = Agent(
        model=ScriptedModel(assistant("done")),
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[60, 300],
                                kill_after_seconds=900, log_dir=str(tmp_path)),
        templates=templates,
        step_limit=1,
    )

    agent.run("finish")

    system = agent.messages[0]["content"]
    assert "still running after 15 minutes is" in system
    assert "{{kill_after_minutes}}" not in system


def test_system_message_orders_the_turn_end_rules_before_the_marker_rule(
        tmp_path, templates):
    agent = _agent(tmp_path, templates, assistant("done"))

    agent.run("finish")

    system = agent.messages[0]["content"]
    assert system.index("Ending your turn:") < system.index("Model control markers:")


def test_fresh_session_appends_cwd_agents_md_to_system_message(tmp_path, templates):
    """The AGENTS.md convention: a fresh session carries the cwd's file in the
    system message, appended after the built-in template."""
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text("Always run the test suite before finishing.\n",
                         encoding="utf-8")
    agent = _agent(tmp_path, templates, assistant("done"))

    agent.run("count")

    template = render(templates["system"], cwd=str(tmp_path), skills="",
                      kill_after_minutes=f"{10 / 60:g}")
    content = agent.messages[0]["content"]
    assert content == template + "\n\n" + agents_md.read_text(encoding="utf-8")
    assert content.index("You have exactly one tool") < content.index(
        "Always run the test suite"
    )
    assert content.index("Ending your turn:") < content.index(
        "Always run the test suite"
    )


def test_fresh_session_skips_missing_or_blank_agents_md(tmp_path, templates):
    template = render(templates["system"], cwd=str(tmp_path), skills="",
                      kill_after_minutes=f"{10 / 60:g}")

    agent = _agent(tmp_path, templates, assistant("done"))
    agent.run("count")
    assert agent.messages[0]["content"] == template

    (tmp_path / "AGENTS.md").write_text("\n   \n", encoding="utf-8")
    blank = _agent(tmp_path, templates, assistant("done"))
    blank.run("count")
    assert blank.messages[0]["content"] == template


def test_undecodable_agents_md_warns_on_stderr_and_session_continues(
        tmp_path, templates, capsys):
    """An unreadable convention file must not kill a session for an unrelated task."""
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_bytes(b"\xff\xfe not utf-8")
    agent = _agent(tmp_path, templates, assistant("done"))

    result = agent.run("count")

    assert result["completed"] is True
    assert agent.messages[0]["content"] == render(
        templates["system"], cwd=str(tmp_path), skills="",
        kill_after_minutes=f"{10 / 60:g}",
    )
    err = capsys.readouterr().err
    assert "warning" in err
    assert str(agents_md) in err


def test_resume_replays_stored_system_text_without_rereading_agents_md(
        tmp_path, templates):
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text("original convention\n", encoding="utf-8")
    state_file = tmp_path / "session.json"
    first = Agent(
        model=ScriptedModel(assistant("done")),
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                              kill_after_seconds=10, log_dir=str(tmp_path)),
        templates=templates,
        step_limit=5,
        state_file=str(state_file),
    )
    first.run("count")
    original_system = first.messages[0]["content"]
    assert "original convention" in original_system

    agents_md.write_text("changed convention\n", encoding="utf-8")
    resumed = Agent(
        model=ScriptedModel(assistant("done again")),
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                              kill_after_seconds=10, log_dir=str(tmp_path)),
        templates=templates,
        step_limit=5,
        state_file=str(state_file),
        resume=True,
    )
    resumed.run("count again")

    assert resumed.messages[0]["content"] == original_system
    assert "changed convention" not in resumed.messages[0]["content"]
