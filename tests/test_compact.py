"""Context reset: the per-command cap, the trigger, the rebuilt context, the
over-window fallback, the startup floor, and the byte-stable prefix the whole
design exists for."""

import copy
import json
from pathlib import Path

import litellm
import pytest
import typer
from litellm.exceptions import ContextWindowExceededError
from typer.testing import CliRunner

import agent as agent_module
import main as cli_main
from agent import STATE_PROTOCOL, Agent, _load_state, load_config, render
from compact import (IMAGE_TOKENS, TRUNCATION_MARKER, est_message_chars,
                     est_messages_tokens, truncate_observation)
from conftest import assistant, tool_call
from environment import Environment
from model import Model
from transcript import Transcript

OVERFLOW = "OVERFLOW"
BAD400 = "BAD400"


class ResetModel:
    """Scripted (reply, prompt_tokens) steps that also keep every prompt sent.

    reply is an (assistant message, finish_reason) pair, OVERFLOW (raises the
    over-window error and leaves usage unset, like the endpoint) or BAD400
    (an unrelated 400). prompt_tokens becomes last_prompt_tokens after a reply.
    """

    def __init__(self, *steps):
        self.model_name = "openai/fake-model"
        self._steps = iter(steps)
        self.last_prompt_tokens = None
        self.last_cached_tokens = 0
        self.seen_prompts = []

    def query(self, messages, tools=None):
        self.seen_prompts.append(copy.deepcopy(messages))
        reply, prompt_tokens = next(self._steps)
        if reply == OVERFLOW:
            raise ContextWindowExceededError(
                "prompt is too long", model="fake", llm_provider="fake"
            )
        if reply == BAD400:
            raise litellm.exceptions.BadRequestError(
                "some other 400", model="fake", llm_provider="fake"
            )
        self.last_prompt_tokens = prompt_tokens
        return reply

    def usage(self):
        return {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}


def _compact(**overrides):
    """The three-key compact block: a 10,000-token window, reset at 5,000."""
    compact = {"context_window": 10000, "threshold_fraction": 0.5,
               "command_observation_chars": 5000}
    compact.update(overrides)
    return compact


def _agent(tmp_path, model, compact=None, emit=None, state_file=None, resume=False,
           images=(), step_limit=6):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    return Agent(
        model=model,
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                                kill_after_seconds=10, log_dir=str(log_dir)),
        templates=load_config()["templates"],
        step_limit=step_limit,
        emit=emit,
        compact=compact or _compact(),
        state_file=str(state_file) if state_file else None,
        resume=resume,
        images=images,
    )


def _big_output(chars, char="x"):
    return f"python3 -c \"print('{char}' * {chars}, end='')\""


def _observation(output):
    return render(load_config()["templates"]["observation"], returncode=0, output=output)


def _pointer(session_dir):
    return render(load_config()["templates"]["reset_pointer"],
                  transcript=str(session_dir / "transcript.md"), session_dir=str(session_dir))


def _marker(output):
    return TRUNCATION_MARKER.format(chars=len(output), lines=output.count("\n") + 1)


def _expiry_compact(**overrides):
    compact = {"context_window": 1_000_000, "threshold_fraction": 0.99,
               "command_observation_chars": 100_000}
    compact.update(overrides)
    return compact


def _call_step(index, command="printf x", **extra):
    return assistant(tool_calls=[tool_call(index, command=command)], **extra)


def _tool_messages(messages):
    return [message for message in messages if message["role"] == "tool"]


# --- the per-command cap ------------------------------------------------------

def test_truncate_observation_keeps_outputs_up_to_the_cap_and_splits_larger_ones():
    assert truncate_observation("abc", 10) == "abc"
    assert truncate_observation("a" * 10, 10) == "a" * 10

    output = "".join(f"line {index}\n" for index in range(100))
    bounded = truncate_observation(output, 101)
    assert bounded == output[:50] + "\n" + _marker(output) + "\n" + output[-51:]
    assert _marker(output) == "[... truncated: 790 chars, 101 lines total ...]"

    marker_only = truncate_observation("x" * 12345, 10).split("\n")[1]
    assert marker_only == "[... truncated: 12,345 chars, 1 lines total ...]"
    assert "/" not in marker_only  # the marker names no file; the transcript stub does


def test_each_command_is_bounded_on_its_own_with_the_header_intact(tmp_path):
    model = ResetModel(
        (assistant(tool_calls=[tool_call(1, command=_big_output(300, "x")),
                               tool_call(2, command=_big_output(300, "y")),
                               tool_call(3, command=_big_output(100, "z"))]), 100),
        (assistant("done"), 100),
    )
    agent = _agent(tmp_path, model, compact=_compact(command_observation_chars=100))

    agent.run("emit three outputs")

    tool_messages = [m for m in model.seen_prompts[1] if m["role"] == "tool"]
    assert [m["content"] for m in tool_messages[:2]] == [
        _observation("x" * 50 + "\n" + _marker("x" * 300) + "\n" + "x" * 50),
        _observation("y" * 50 + "\n" + _marker("y" * 300) + "\n" + "y" * 50),
    ]
    assert tool_messages[2]["content"] == _observation("z" * 100)
    logs = sorted((tmp_path / "logs").glob("s-1-*.log"))
    assert [len(log.read_text()) for log in logs] == [300, 300, 100]


def test_expiry_keeps_old_content_while_its_estimate_is_under_20000_tokens(tmp_path):
    replies = [(_call_step(1, _big_output(79_000)), 100)]
    replies.extend((_call_step(1), 100) for _ in range(4))
    replies.append((assistant("done"), 100))
    model = ResetModel(*replies)
    state_file = tmp_path / "sessions" / "abc.json"
    state_file.parent.mkdir()
    agent = _agent(
        tmp_path, model, compact=_expiry_compact(), state_file=state_file
    )

    agent.run("keep the old output until the expiry batch is full")

    prompts = model.seen_prompts
    for earlier, later in zip(prompts, prompts[1:]):
        assert later[:len(earlier)] == earlier
    first_output = _tool_messages(agent.messages)[0]["content"]
    assert first_output == _observation("x" * 79_000)


def test_expiry_replaces_all_eligible_outputs_in_one_prefix_boundary(tmp_path):
    first_calls = [tool_call(index, command=_big_output(20_000))
                   for index in range(1, 5)]
    replies = [(assistant(tool_calls=first_calls), 100)]
    replies.extend((_call_step(1), 100) for _ in range(4))
    replies.append((assistant("done"), 100))
    model = ResetModel(*replies)
    state_file = tmp_path / "sessions" / "abc.json"
    state_file.parent.mkdir()
    agent = _agent(
        tmp_path, model,
        compact=_expiry_compact(command_observation_chars=25_000),
        state_file=state_file,
    )

    agent.run("replace the old outputs together")

    prompts = model.seen_prompts
    for earlier, later in zip(prompts[1:4], prompts[2:5]):
        assert later[:len(earlier)] == earlier
        assert all(
            message["content"] == _observation("x" * 20_000)
            for message in _tool_messages(earlier)[:4]
        )
    records = Transcript(state_file.with_suffix(".d") / "transcript.md", "abc").step_records()
    expected_stubs = records[0]["tool_stubs"]
    expired_outputs = _tool_messages(prompts[5])[:4]
    assert [message["content"] for message in expired_outputs] == expected_stubs
    for stub in expected_stubs:
        log_path = stub.rstrip("\n").split(" · ")[-1]
        assert Path(log_path).read_text() == "x" * 20_000


def test_expiry_removes_reasoning_and_marks_a_long_command_with_its_transcript_step(
    tmp_path,
):
    long_command = "printf x; #" + " padding" * 35
    replies = [(_call_step(1, long_command, reasoning_content="r" * 80_000), 100)]
    replies.extend((_call_step(1), 100) for _ in range(4))
    replies.append((assistant("done"), 100))
    model = ResetModel(*replies)
    state_file = tmp_path / "sessions" / "abc.json"
    state_file.parent.mkdir()
    agent = _agent(tmp_path, model, compact=_expiry_compact(), state_file=state_file)

    agent.run("expire reasoning and keep a command pointer")

    first_assistant = next(
        message for message in agent.messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert "reasoning_content" not in first_assistant
    command = json.loads(first_assistant["tool_calls"][0]["function"]["arguments"])["command"]
    transcript_path = state_file.with_suffix(".d") / "transcript.md"
    marker = f"\n[full command in transcript step 1 at {transcript_path}]"
    assert command == long_command[:200] + marker
    assert command.count("[full command in transcript step") == 1
    assert long_command in transcript_path.read_text()
    assert "r" * 80_000 in transcript_path.read_text()


def test_a_resumed_run_expires_loaded_output_by_the_same_step_and_batch_rules(tmp_path):
    state_file = tmp_path / "sessions" / "abc.json"
    state_file.parent.mkdir()
    first_replies = [(_call_step(1, _big_output(80_000)), 100)]
    first_replies.extend((_call_step(1), 100) for _ in range(3))
    first = _agent(
        tmp_path, ResetModel(*first_replies),
        compact=_expiry_compact(command_observation_chars=90_000),
        state_file=state_file, step_limit=4,
    )
    with pytest.raises(RuntimeError, match="Step limit"):
        first.run("build four prior steps")
    assert _tool_messages(first.messages)[0]["content"] == _observation("x" * 80_000)

    resumed_model = ResetModel((assistant("continued"), 100))
    resumed = _agent(
        tmp_path, resumed_model,
        compact=_expiry_compact(command_observation_chars=90_000),
        state_file=state_file, resume=True,
    )
    resumed.run("continue the session")

    records = Transcript(state_file.with_suffix(".d") / "transcript.md", "abc").step_records()
    assert _tool_messages(resumed.messages)[0]["content"] == records[0]["tool_stubs"][0]


HEREDOC_WITH_HEADINGS = (
    "cat > notes.md <<'EOF'\n"
    "# Plan\n"
    "## Step one\n"
    + "Padding line for the record.\n" * 8
    + "```python\nprint('two')\n```\n"
    "EOF"
)


def test_a_resumed_session_expires_over_the_batch_with_one_record_per_step(
    tmp_path, monkeypatch
):
    state_file = tmp_path / "sessions" / "abc.json"
    state_file.parent.mkdir()
    transcript_path = state_file.with_suffix(".d") / "transcript.md"
    replies = [(_call_step(1, HEREDOC_WITH_HEADINGS), 100)]
    replies.extend((_call_step(1, _big_output(17_000)), 100) for _ in range(9))
    first = _agent(tmp_path, ResetModel(*replies), compact=_expiry_compact(),
                   state_file=state_file, step_limit=10)
    # The first run keeps its whole batch on disk, so the expiry happens on resume.
    with monkeypatch.context() as patch:
        patch.setattr(agent_module, "EXPIRY_BATCH_TOKEN_THRESHOLD", 10**9)
        with pytest.raises(RuntimeError, match="Step limit"):
            first.run("ten steps")
    messages = _load_state(state_file)
    unmatched = {"role": "assistant", "content": "", "reasoning_content": "r" * 5_000,
                 "tool_calls": [tool_call(99, command="printf unrecorded")]}
    unmatched_tool = {"role": "tool", "tool_call_id": "call-99",
                      "content": _observation("u" * 16_000)}
    messages[2:2] = [unmatched, unmatched_tool]
    state_file.write_text(json.dumps({"protocol": STATE_PROTOCOL, "messages": messages}))
    before = copy.deepcopy([unmatched, unmatched_tool])

    resumed = _agent(tmp_path, ResetModel((assistant("done"), 100)),
                     compact=_expiry_compact(), state_file=state_file, resume=True)
    result = resumed.run("continue")

    assert result["completed"] is True
    records = Transcript(transcript_path, "abc").step_records()
    assert [record["step"] for record in records] == list(range(1, 12))
    by_step = {record["step"]: record for record in records}
    assert by_step[1]["commands"] == [HEREDOC_WITH_HEADINGS]
    heredoc_call = resumed.messages[4]["tool_calls"][0]
    assert json.loads(heredoc_call["function"]["arguments"])["command"] == (
        HEREDOC_WITH_HEADINGS[:200] + f"\n[full command in transcript step 1 at {transcript_path}]")
    # Tool message 0 is the unmatched step; message k is run step k's output.
    tool_messages = _tool_messages(resumed.messages)
    for step in range(1, 7):
        assert tool_messages[step]["content"] == by_step[step]["tool_stubs"][0]
    for step in range(7, 11):
        assert tool_messages[step]["content"] == _observation("x" * 17_000)
    assert resumed.messages[2:4] == before
    assert transcript_path.read_text(encoding="utf-8").count("\n## expiry · ") == 1


def test_a_session_without_a_state_file_expires_nothing_over_the_batch(tmp_path):
    long_command = "printf x; #" + " padding" * 35
    replies = [(_call_step(1, _big_output(80_000), reasoning_content="r" * 80_000), 100),
               (_call_step(1, long_command), 100)]
    replies.extend((_call_step(1), 100) for _ in range(3))
    replies.append((assistant("done"), 100))
    agent = _agent(tmp_path, ResetModel(*replies), compact=_expiry_compact())

    result = agent.run("expire nothing without a transcript")

    assert result["completed"] is True
    assert agent.transcript is None
    first, second = [m for m in agent.messages if m.get("tool_calls")][:2]
    assert first["reasoning_content"] == "r" * 80_000
    assert _tool_messages(agent.messages)[0]["content"] == _observation("x" * 80_000)
    assert json.loads(second["tool_calls"][0]["function"]["arguments"])["command"] == long_command


def test_invalid_tool_call_error_is_bounded_by_the_same_cap(tmp_path):
    model = ResetModel(
        (assistant(tool_calls=[tool_call(1, name="nope", command="ls")]), 100),
        (assistant("done"), 100),
    )
    agent = _agent(tmp_path, model, compact=_compact(command_observation_chars=40))

    result = agent.run("call a tool that does not exist")

    tool_message = [m for m in model.seen_prompts[1] if m["role"] == "tool"][0]
    assert "[... truncated: " in tool_message["content"]
    assert len(tool_message["content"]) < 40 + len(_marker("x" * 100)) + 2
    assert result["steps"][0]["note"] == "invalid tool call"


# --- the trigger --------------------------------------------------------------

@pytest.mark.parametrize("measured, resets", [(4500, True), (4000, False)])
def test_threshold_trigger_uses_the_measured_anchor_plus_what_was_appended(
    tmp_path, measured, resets
):
    s1 = assistant(tool_calls=[tool_call(1, command=_big_output(2000))])
    model = ResetModel((s1, measured), (assistant("done"), 100))
    events = []
    state_file = tmp_path / "sessions" / "abc.json"
    state_file.parent.mkdir()
    agent = _agent(tmp_path, model, emit=events.append, state_file=state_file)

    agent.run("fill the context")

    first, second = model.seen_prompts
    appended = [s1[0], {"role": "tool", "tool_call_id": "call-1",
                        "content": _observation("x" * 2000)}]
    expected = measured + est_messages_tokens(appended)
    compact_events = [e for e in events if e["type"] == "compact"]
    if not resets:
        assert expected < 5000
        assert compact_events == []
        assert second[:len(first)] == first and len(second) == 4
        return
    assert expected >= 5000
    assert compact_events == [{
        "type": "compact", "step": 2, "trigger": "threshold",
        "pre_tokens": expected, "post_tokens_est": est_messages_tokens(second),
    }]
    assert [m["role"] for m in second] == ["system", "user", "user"]
    assert second[0] == first[0]
    assert second[1]["content"] == _pointer(tmp_path / "sessions" / "abc.d")
    assert second[2] == first[1]


def test_without_an_anchor_the_whole_context_is_estimated_and_a_resume_can_reset_first(
    tmp_path,
):
    state_file = tmp_path / "sessions" / "old.json"
    state_file.parent.mkdir()
    history = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "earlier task"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call(1, command="cat big")]},
        {"role": "tool", "tool_call_id": "call-1", "content": "b" * 30000},
        {"role": "assistant", "content": "earlier answer"},
    ]
    state_file.write_text(json.dumps({"protocol": STATE_PROTOCOL, "messages": history}))
    model = ResetModel((assistant("done"), 100))
    events = []
    agent = _agent(tmp_path, model, emit=events.append, state_file=state_file, resume=True)

    agent.run("continue")

    task = {"role": "user", "content": render(load_config()["templates"]["instance"],
                                              task="continue")}
    (prompt,) = model.seen_prompts
    assert [m["role"] for m in prompt] == ["system", "user", "user"]
    assert prompt[0] == history[0] and prompt[2] == task
    assert [e for e in events if e["type"] == "compact"] == [{
        "type": "compact", "step": 1, "trigger": "threshold",
        "pre_tokens": est_messages_tokens(history + [task]),
        "post_tokens_est": est_messages_tokens(prompt),
    }]


def test_est_message_chars_counts_text_reasoning_calls_and_images():
    call = tool_call(1, command="ls")
    assert est_message_chars({"role": "assistant", "content": "abc",
                              "reasoning_content": "de", "tool_calls": [call]}) == (
        5 + len(json.dumps([call])))
    parts = [{"type": "text", "text": "t" * 40},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}}]
    assert est_message_chars({"role": "user", "content": parts}) == 40 + 2 * IMAGE_TOKENS * 4
    assert est_messages_tokens([{"role": "user", "content": "x" * 400}]) == 100


def test_last_query_index_is_cleared_by_a_reset_and_reanchored_by_the_next_query(tmp_path):
    seen = []
    model = ResetModel((OVERFLOW, None), (assistant("done"), 100))
    agent = _agent(tmp_path, model, emit=lambda e: seen.append(agent._last_query_index)
                   if e["type"] == "compact" else None)

    agent.run("finish")

    assert seen == [None]
    assert agent._last_query_index == 3


# --- the rebuilt context ------------------------------------------------------

def test_system_and_task_messages_are_reused_as_they_are_across_resets(tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    s1 = assistant(tool_calls=[tool_call(1, command="echo hi")])
    model = ResetModel((OVERFLOW, None), (s1, 100), (OVERFLOW, None),
                       (assistant("done"), 100))
    agent = _agent(tmp_path, model, images=[str(image)])

    agent.run("look at the picture")

    p0, p1, p2, p3 = model.seen_prompts
    assert [m["role"] for m in p1] == ["system", "user", "user"]
    assert p1 == p3
    assert p1[0] == p0[0] and p1[2] == p0[1]
    assert isinstance(p1[2]["content"], list) and p1[2]["content"][1]["type"] == "image_url"
    assert p2[:3] == p1
    assert agent.messages[2] is agent._task
    assert agent.messages[1]["content"] == _pointer(tmp_path)


def test_state_file_after_a_reset_holds_the_three_messages_and_resumes_from_them(tmp_path):
    state_file = tmp_path / "sessions" / "abc.json"
    state_file.parent.mkdir()
    snapshots = []
    model = ResetModel((OVERFLOW, None), (assistant("first answer"), 100))
    agent = _agent(tmp_path, model, state_file=state_file,
                   emit=lambda e: snapshots.append(_load_state(state_file))
                   if e["type"] == "compact" else None)
    agent.run("first task")

    (snapshot,) = snapshots
    assert [m["role"] for m in snapshot] == ["system", "user", "user"]
    assert snapshot[1]["content"] == _pointer(tmp_path / "sessions" / "abc.d")

    second = ResetModel((assistant("second answer"), 100))
    _agent(tmp_path, second, state_file=state_file, resume=True).run("second task")
    (prompt,) = second.seen_prompts
    assert prompt[:3] == snapshot
    assert [m["role"] for m in prompt] == ["system", "user", "user", "assistant", "user"]
    assert "second task" in prompt[4]["content"]


def test_old_session_state_loads_unchanged_and_the_first_reset_clears_its_placeholders(
    tmp_path,
):
    placeholder = ("[observation masked by context compaction: original was 999 chars. "
                   "Full text: /somewhere/s-1-1.log]")
    summary = "Another model has summarized the conversation so far.\n1. Progress: some."
    history = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": summary},
        {"role": "assistant", "content": "", "tool_calls": [tool_call(1, command="cat big")]},
        {"role": "tool", "tool_call_id": "call-1", "content": placeholder},
        {"role": "assistant", "content": "earlier answer " + "b" * 4000},
    ]
    state_file = tmp_path / "sessions" / "old.json"
    state_file.parent.mkdir()
    state_file.write_text(json.dumps({"protocol": STATE_PROTOCOL, "messages": history}))

    roomy = ResetModel((assistant("done"), 100))
    _agent(tmp_path, roomy, state_file=state_file, resume=True).run("continue")
    (prompt,) = roomy.seen_prompts
    assert prompt[:len(history)] == history  # loaded as-is, placeholders and all

    tight = ResetModel((assistant("done again"), 100))
    _agent(tmp_path, tight, state_file=state_file, resume=True,
           compact=_compact(context_window=1000)).run("continue once more")
    (prompt,) = tight.seen_prompts
    assert len(prompt) == 3
    assert not any("compaction" in (m["content"] or "") for m in prompt)


# --- the over-window fallback -------------------------------------------------

def test_overflow_resets_and_retries_once_and_the_compact_event_precedes_the_context_event(
    tmp_path,
):
    model = ResetModel((OVERFLOW, None), (assistant("finished after retry"), 900))
    events = []
    agent = _agent(tmp_path, model, emit=events.append)

    result = agent.run("overflow at the very first call")

    assert result["completed"] is True
    p0, p1 = model.seen_prompts
    assert [m["role"] for m in p1] == ["system", "user", "user"]
    types = [e["type"] for e in events]
    assert types[:2] == ["compact", "context"]
    assert events[0] == {"type": "compact", "step": 1, "trigger": "overflow",
                         "pre_tokens": est_messages_tokens(p0),
                         "post_tokens_est": est_messages_tokens(p1)}
    assert events[1]["prompt_tokens"] == 900


def test_overflow_on_the_retry_raises_instead_of_looping(tmp_path):
    model = ResetModel((OVERFLOW, None), (OVERFLOW, None))
    agent = _agent(tmp_path, model)

    with pytest.raises(RuntimeError, match="still exceeded after a reset"):
        agent.run("overflow twice")
    assert len(model.seen_prompts) == 2


def test_an_unrelated_400_propagates_unchanged_without_a_reset(tmp_path):
    events = []
    model = ResetModel((BAD400, None))
    agent = _agent(tmp_path, model, emit=events.append)

    with pytest.raises(litellm.exceptions.BadRequestError):
        agent.run("hit a plain 400")
    assert [e for e in events if e["type"] == "compact"] == []
    assert len(model.seen_prompts) == 1


# --- the startup floor --------------------------------------------------------

def test_a_rebuilt_context_that_reaches_the_threshold_refuses_to_start(tmp_path):
    model = ResetModel((assistant("never reached"), 100))
    agent = _agent(tmp_path, model, compact=_compact(context_window=100))

    with pytest.raises(RuntimeError) as excinfo:
        agent.run("finish")
    message = str(excinfo.value)
    assert "system=" in message and "pointer=" in message and "task=" in message
    assert "threshold of 50 tokens" in message
    assert model.seen_prompts == []


# --- the event and the stderr line ---------------------------------------------

def test_compact_event_has_five_keys_and_without_emit_becomes_one_stderr_line(
    tmp_path, capsys
):
    events = []
    _agent(tmp_path, ResetModel((OVERFLOW, None), (assistant("done"), 100)),
           emit=events.append).run("finish")
    (event,) = [e for e in events if e["type"] == "compact"]
    assert set(event) == {"type", "step", "trigger", "pre_tokens", "post_tokens_est"}

    _agent(tmp_path, ResetModel((OVERFLOW, None), (assistant("done"), 100))).run("finish")
    err = capsys.readouterr().err
    assert err.startswith("[compact] step 1 trigger=overflow pre_tokens=")
    assert "post_tokens_est=" in err


# --- prefix stability ------------------------------------------------------------

def test_every_prompt_extends_the_previous_one_until_a_reset(tmp_path):
    model = ResetModel(
        (assistant("first", tool_calls=[tool_call(1, command="echo one")]), 100),
        (assistant("second", tool_calls=[tool_call(1, command="echo two"),
                                         tool_call(2, command="echo three")]), 100),
        (assistant("third", tool_calls=[tool_call(1, command="echo four")]), 100),
        (assistant("done"), 100),
    )
    agent = _agent(tmp_path, model)

    agent.run("three steps")

    prompts = model.seen_prompts
    assert [len(p) for p in prompts] == [2, 4, 7, 9]
    for earlier, later in zip(prompts, prompts[1:]):
        assert later[:len(earlier)] == earlier


def _cli_app():
    app = typer.Typer()
    app.command()(cli_main.run)
    return app


def _run_cli(tmp_path, task_file, monkeypatch, responses, measured, extra_args=()):
    """One CLI run with a scripted Model.query that reports `measured` prompt tokens."""
    replies = iter(responses)
    prompts = []

    def query(self, messages, tools=None):
        prompts.append([dict(message) for message in messages])
        self.last_prompt_tokens = measured
        return next(replies)

    monkeypatch.setattr(Model, "query", query)
    monkeypatch.setattr(Model, "usage",
                        lambda self: {"n_calls": 1, "input_tokens": 2, "output_tokens": 3})
    result = CliRunner().invoke(_cli_app(), [
        "--task-file", task_file, "--json", "--cwd", str(tmp_path),
        "--session-dir", str(tmp_path / "sessions"), "--steps", "4", *extra_args,
    ])
    assert result.exit_code == 0, result.output
    return prompts, [json.loads(line) for line in result.output.splitlines()]


def test_a_resumed_turn_extends_the_previous_turn_prefix_byte_for_byte(
    tmp_path, monkeypatch, task_file
):
    first_prompts, events = _run_cli(tmp_path, task_file("turn one"), monkeypatch, [
        assistant(tool_calls=[tool_call(1, command="printf one")]),
        assistant("Turn one complete."),
    ], measured=100)
    session_id = [e for e in events if e["type"] == "session"][0]["session_id"]
    second_prompts, _ = _run_cli(tmp_path, task_file("turn two"), monkeypatch, [
        assistant("Turn two complete."),
    ], measured=100, extra_args=["--resume", session_id])

    last = first_prompts[-1]
    assert second_prompts[0][:len(last)] == last
    assert [m["role"] for m in second_prompts[0][len(last):]] == ["assistant", "user"]


def test_context_window_flag_scales_the_trigger(tmp_path, monkeypatch, task_file):
    responses = [
        assistant(tool_calls=[tool_call(1, command="printf one")]),
        assistant("done"),
    ]
    prompts, events = _run_cli(tmp_path, task_file("go"), monkeypatch, responses,
                               measured=12990, extra_args=["--context-window", "20000"])
    compact_events = [e for e in events if e["type"] == "compact"]
    assert [(e["step"], e["trigger"]) for e in compact_events] == [(2, "threshold")]
    assert compact_events[0]["pre_tokens"] >= 13000
    assert [e["compact_threshold"] for e in events if e["type"] == "context"] == [13000, 13000]
    assert len(prompts[1]) == 3

    prompts, events = _run_cli(tmp_path, task_file("go"), monkeypatch, list(responses),
                               measured=12990)
    assert [e for e in events if e["type"] == "compact"] == []
    assert len(prompts[1]) == 4
