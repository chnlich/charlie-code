"""Layered context compaction: entry budget, mask layer, summarize layer,
over-window fallback, persistence, and the compact event. All offline: the model
is a scripted stand-in injecting usage values and over-window errors.
"""

import json
import re

import pytest
import typer
from typer.testing import CliRunner

import main as cli_main
from agent import Agent, _load_state, load_config
from compact import (
    MASK_SENTINEL,
    bound_observation,
    drop_oldest_middle_half,
    est_messages_tokens,
    mask_old_observations,
    split_steps,
    verbatim_tail_span,
)
from conftest import assistant, tool_call
from environment import Environment
from litellm.exceptions import BadRequestError, ContextWindowExceededError
from model import Model

OVERFLOW = "OVERFLOW"
BAD400 = "BAD400"


class CompactModel:
    """Scripted (reply, prompt_tokens) steps; the compaction fake.

    reply is either an (assistant message, finish_reason) pair or one of the
    sentinels OVERFLOW (raise a context-window error) / BAD400 (raise an
    unrelated 400). A failed call sets no usage, mirroring the real endpoint.
    """

    def __init__(self, *steps):
        self._steps = list(steps)
        self.last_prompt_tokens = None
        self.seen_tools = []
        self.seen_prompts = []

    def query(self, messages, tools=None):
        self.seen_tools.append(tools)
        self.seen_prompts.append([dict(message) for message in messages])
        reply, prompt_tokens = self._steps.pop(0)
        if reply == OVERFLOW:
            raise ContextWindowExceededError(
                "prompt is too long", model="fake", llm_provider="fake"
            )
        if reply == BAD400:
            raise BadRequestError(
                "some other 400", model="fake", llm_provider="fake"
            )
        self.last_prompt_tokens = prompt_tokens
        return reply

    def usage(self):
        return {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}


def _compact(**overrides):
    cfg = {
        "context_window": 10000,
        "threshold_fraction": 0.5,  # threshold: 5000 tokens
        "mask_keep_steps": 1,
        "tail_budget_tokens": 300,  # 1200 chars
        "step_observation_budget_chars": 40000,
    }
    cfg.update(overrides)
    return cfg


def _agent(tmp_path, templates, *steps, compact=None, emit=None, state_file=None,
           step_limit=20, resume=False):
    return Agent(
        model=CompactModel(*steps),
        environment=Environment(cwd=str(tmp_path), timeout=10),
        templates=templates,
        step_limit=step_limit,
        emit=emit,
        state_file=state_file,
        resume=resume,
        compact=compact if compact is not None else _compact(),
    )


def _big_output(chars, letter):
    return f"head -c {chars} /dev/zero | tr '\\0' {letter}"


# ---------------------------------------------------------------------------
# Acceptance 1: threshold-triggered mask pass keeps assistant messages
# byte-identical, tool_call_id pairing complete, old observations placeholders.
# ---------------------------------------------------------------------------

def test_mask_pass_preserves_assistant_verbatim_and_pairing(tmp_path, templates):
    s1 = assistant(tool_calls=[tool_call(1, command=_big_output(1000, "a"))])
    s2 = assistant(tool_calls=[tool_call(2, command=_big_output(1000, "b"))])
    s3 = assistant("working", tool_calls=[tool_call(3, command=_big_output(1000, "c"))],
                   reasoning_content="step-three thinking")
    s4 = assistant("all done")
    events = []
    agent = _agent(tmp_path, templates,
                   (s1, 100), (s2, 100), (s3, 4900), (s4, 100), emit=events.append)

    result = agent.run("three steps then done")

    assert result["completed"] is True
    # Assistant messages byte-identical, reasoning_content included.
    stored = [m for m in agent.messages if m["role"] == "assistant"]
    assert stored == [s1[0], s2[0], s3[0], s4[0]]
    # tool_call_id pairing complete.
    tool_messages = [m for m in agent.messages if m["role"] == "tool"]
    expected_ids = [c["id"] for s in (s1, s2, s3) for c in s[0]["tool_calls"]]
    assert [m["tool_call_id"] for m in tool_messages] == expected_ids
    # Old observations (outside mask_keep_steps=1) are placeholders keeping the
    # exit code and the original length; the newest step stays verbatim.
    for masked in tool_messages[:2]:
        assert masked["content"].startswith(MASK_SENTINEL)
        assert "exit code was 0" in masked["content"]
        assert re.search(r"original was \d{4} chars", masked["content"])
    assert "c" * 200 in tool_messages[2]["content"]
    assert not tool_messages[2]["content"].startswith(MASK_SENTINEL)
    # Masking alone brought the estimate under the threshold: no summarize.
    compact_events = [e for e in events if e["type"] == "compact"]
    assert len(compact_events) == 1
    event = compact_events[0]
    assert set(event) == {"type", "step", "layer", "trigger", "pre_tokens",
                          "post_tokens_est"}
    assert event["step"] == 4
    assert event["layer"] == "mask"
    assert event["trigger"] == "threshold"
    assert event["pre_tokens"] >= 5000
    assert 0 < event["post_tokens_est"] < event["pre_tokens"]


def test_mask_is_idempotent():
    messages = []
    for index in range(4):
        messages.append({"role": "assistant",
                         "content": "",
                         "tool_calls": [tool_call(index, command="true")],
                         "reasoning_content": f"thinking {index}"})
        messages.append({"role": "tool", "tool_call_id": f"call-{index}",
                         "content": f"Exit code: 0\nOutput:\n{'x' * 500}"})

    first = mask_old_observations(messages, keep_steps=1)
    snapshot = json.dumps(messages, sort_keys=True)
    second = mask_old_observations(messages, keep_steps=1)

    assert first == 3
    assert second == 0
    assert json.dumps(messages, sort_keys=True) == snapshot


# ---------------------------------------------------------------------------
# Acceptance 2: when masking is insufficient the history is exactly
# [system, task, summary, tail] with the tail in whole steps including the last.
# ---------------------------------------------------------------------------

def test_summarize_rebuilds_history_when_masking_is_not_enough(tmp_path, templates):
    s1 = assistant(tool_calls=[tool_call(1, command=_big_output(2000, "a"))])
    s2 = assistant(tool_calls=[tool_call(2, command=_big_output(20000, "b"))])
    summary_text = "1. Progress: steps one and two done. 5. Remaining: finish."
    s_summary = assistant(summary_text)
    s3 = assistant("wrapping up")
    events = []
    agent = _agent(tmp_path, templates,
                   (s1, 100), (s2, 4900), (s_summary, 100), (s3, 100),
                   emit=events.append)

    result = agent.run("two big steps then done")

    assert result["completed"] is True
    # History is exactly [system, task, summary, verbatim tail] plus the final
    # assistant answer appended after compaction.
    assert [m["role"] for m in agent.messages] == [
        "system", "user", "user", "assistant", "tool", "assistant",
    ]
    first_prompt = agent.model.seen_prompts[0]
    assert agent.messages[0] == first_prompt[0]  # system, byte-identical
    assert agent.messages[1] == first_prompt[1]  # task, byte-identical
    summary_message = agent.messages[2]
    assert summary_message["role"] == "user"
    assert summary_message["content"].startswith(
        "Another model has summarized the session so far"
    )
    assert summary_text in summary_message["content"]
    # Tail: whole steps, including the last step, assistant byte-identical.
    tail = agent.messages[3:5]
    assert tail[0] == s2[0]
    assert tail[1]["tool_call_id"] == s2[0]["tool_calls"][0]["id"]
    assert not tail[1]["content"].startswith(MASK_SENTINEL)
    assert "b" * 200 in tail[1]["content"]
    # The tail budget (1200 chars) fits neither the masked middle nor more than
    # the last step; the last step is kept anyway (always at least one).
    assert [m["role"] for m in tail] == ["assistant", "tool"]
    # The summary call used the masked history plus the compact_prompt, no tools.
    assert agent.model.seen_tools[2] is None
    summary_prompt = agent.model.seen_prompts[2]
    assert summary_prompt[-1] == {"role": "user",
                                  "content": templates["compact_prompt"]}
    assert any(m["role"] == "tool" and m["content"].startswith(MASK_SENTINEL)
               for m in summary_prompt[:-1])
    # No placeholder survives into the rebuilt history: the masked middle is gone.
    assert not any(m["role"] == "tool" and m["content"].startswith(MASK_SENTINEL)
                   for m in agent.messages)

    compact_events = [e for e in events if e["type"] == "compact"]
    assert len(compact_events) == 1
    assert compact_events[0]["layer"] == "summarize"
    assert compact_events[0]["trigger"] == "threshold"


# ---------------------------------------------------------------------------
# Acceptance 3: over-window error -> compaction + single retry; second failure
# raises; unrelated 400s propagate.
# ---------------------------------------------------------------------------

def test_overflow_compacts_and_retries_once_then_completes(tmp_path, templates):
    s_summary = assistant("1. Progress: nothing yet. 5. Remaining: everything.")
    s_done = assistant("finished after retry")
    events = []
    agent = _agent(tmp_path, templates,
                   (OVERFLOW, None), (s_summary, 100), (s_done, 100),
                   emit=events.append)

    result = agent.run("overflow at the very first call")

    assert result["completed"] is True
    assert result["final_output"] == "finished after retry"
    assert len(agent.model.seen_prompts) == 3  # failed call, summary, retry
    compact_events = [e for e in events if e["type"] == "compact"]
    assert len(compact_events) == 1
    assert set(compact_events[0]) == {"type", "step", "layer", "trigger",
                                      "pre_tokens", "post_tokens_est"}
    assert compact_events[0]["trigger"] == "overflow"
    assert compact_events[0]["layer"] == "summarize"
    assert [m["role"] for m in agent.messages] == ["system", "user", "user",
                                                   "assistant"]


def test_overflow_retry_failing_again_raises(tmp_path, templates):
    s_summary = assistant("1. Progress: none.")
    agent = _agent(tmp_path, templates,
                   (OVERFLOW, None), (s_summary, 100), (OVERFLOW, None))

    with pytest.raises(RuntimeError, match="context window still exceeded"):
        agent.run("overflow that survives compaction")


def test_over_window_summary_drops_oldest_middle_half_and_retries(tmp_path, templates):
    s1 = assistant(tool_calls=[tool_call(1, command=_big_output(2000, "a"))])
    s2 = assistant(tool_calls=[tool_call(2, command=_big_output(2000, "b"))])
    s3 = assistant(tool_calls=[tool_call(3, command=_big_output(5000, "c"))])
    s_summary = assistant("1. Progress: summarized after dropping middle.")
    s_done = assistant("recovered")
    agent = _agent(tmp_path, templates,
                   (s1, 100), (s2, 100), (s3, 100),
                   (OVERFLOW, None),   # original call at step 4
                   (OVERFLOW, None),   # first summary call: itself over-window
                   (s_summary, 100),   # summary retry after dropping middle
                   (s_done, 100))      # retried original call

    result = agent.run("three steps then overflow")

    assert result["completed"] is True
    # The retried summary (call 5) saw a strictly smaller prompt than the first
    # over-window summary attempt (call 4): the oldest half of the maskable
    # middle (whole steps only) was dropped between the two attempts.
    assert len(agent.model.seen_prompts[5]) < len(agent.model.seen_prompts[4])
    assert [m["role"] for m in agent.messages] == [
        "system", "user", "user", "assistant", "tool", "assistant",
    ]
    assert agent.messages[3] == s3[0]
    assert agent.messages[4]["tool_call_id"] == s3[0]["tool_calls"][0]["id"]


def test_summary_carries_markers_once_retries_then_accepts(tmp_path, templates):
    s_dirty = assistant("look: <|open|> slipped in")
    s_clean = assistant("1. Progress: clean summary.")
    s_done = assistant("done")
    agent = _agent(tmp_path, templates,
                   (OVERFLOW, None), (s_dirty, 100), (s_clean, 100), (s_done, 100))

    result = agent.run("summary marker retry")

    assert result["completed"] is True
    assert "clean summary" in agent.messages[2]["content"]
    assert "<|open|>" not in agent.messages[2]["content"]


def test_summary_with_markers_twice_raises(tmp_path, templates):
    s_dirty = assistant("look: <|open|> slipped in")
    agent = _agent(tmp_path, templates,
                   (OVERFLOW, None), (s_dirty, 100), (s_dirty, 100))

    with pytest.raises(RuntimeError, match="control markers after a retry"):
        agent.run("summary never comes back clean")


def test_unrelated_400_propagates_unchanged(tmp_path, templates):
    events = []
    agent = _agent(tmp_path, templates, (BAD400, None), emit=events.append)

    with pytest.raises(BadRequestError, match="some other 400"):
        agent.run("a different endpoint error")
    assert not any(e["type"] == "compact" for e in events)


# ---------------------------------------------------------------------------
# Acceptance 4: state file is readable right after compaction and --resume works.
# ---------------------------------------------------------------------------

def test_state_file_readable_after_compaction_and_agent_resume(tmp_path, templates):
    s1 = assistant(tool_calls=[tool_call(1, command=_big_output(2000, "a"))])
    s2 = assistant(tool_calls=[tool_call(2, command=_big_output(20000, "b"))])
    s_summary = assistant("1. Progress: two steps summarized.")
    s_done = assistant("turn one done")
    state_file = tmp_path / "session.json"
    agent = _agent(tmp_path, templates,
                   (s1, 100), (s2, 4900), (s_summary, 100), (s_done, 100),
                   state_file=str(state_file))

    agent.run("turn one")

    # Persisted right at compaction time, loadable, and identical to memory.
    state = json.loads(state_file.read_text())
    assert state["protocol"] == "tool-calls-v1"
    assert state["messages"] == agent.messages
    assert _load_state(state_file) == agent.messages

    # An Agent --resume on the compacted state continues the session.
    resumed = _agent(tmp_path, templates, (assistant("turn two done"), 100),
                     state_file=str(state_file), resume=True)
    result = resumed.run("turn two")
    assert result["completed"] is True
    first_prompt = resumed.model.seen_prompts[0]
    assert any("summarized the session so far" in (m.get("content") or "")
               and "two steps summarized" in m["content"]
               for m in first_prompt if m["role"] == "user")
    assert first_prompt[-1]["role"] == "user"
    assert "turn two" in first_prompt[-1]["content"]


def _cli_app():
    app = typer.Typer()
    app.command()(cli_main.run)
    return app


def test_cli_resume_works_on_a_compacted_session(tmp_path, monkeypatch, task_file):
    real_load_config = load_config

    def patched_load_config():
        config = real_load_config()
        config["compact"] = _compact()
        return config

    monkeypatch.setattr(cli_main, "load_config", patched_load_config)
    replies = iter([
        assistant(tool_calls=[tool_call(1, command=_big_output(2000, "a"))]),
        assistant(tool_calls=[tool_call(2, command=_big_output(20000, "b"))]),
        assistant("SUMMARY: steps one and two."),
        assistant("turn one done"),
        assistant("turn two done"),
    ])
    tokens = iter([100, 4900, 100, 100, 100])
    captured = []

    def query(self, messages, tools=None):
        captured.append([dict(message) for message in messages])
        self.last_prompt_tokens = next(tokens)
        return next(replies)

    monkeypatch.setattr(Model, "query", query)

    session_dir = tmp_path / "sessions"
    runner = CliRunner()
    first = runner.invoke(
        _cli_app(),
        ["--task-file", task_file("turn one"), "--json", "--cwd", str(tmp_path),
         "--session-dir", str(session_dir), "--steps", "6"],
    )

    assert first.exit_code == 0, first.output
    events = [json.loads(line) for line in first.stdout.splitlines()]
    compact_events = [e for e in events if e["type"] == "compact"]
    assert len(compact_events) == 1
    assert compact_events[0]["layer"] == "summarize"
    assert compact_events[0]["trigger"] == "threshold"
    session_id = events[0]["session_id"]

    state = json.loads((session_dir / f"{session_id}.json").read_text())
    assert state["protocol"] == "tool-calls-v1"
    assert state["messages"][2]["role"] == "user"
    assert "SUMMARY: steps one and two." in state["messages"][2]["content"]

    second = runner.invoke(
        _cli_app(),
        ["--task-file", task_file("turn two"), "--json", "--resume", session_id,
         "--cwd", str(tmp_path), "--session-dir", str(session_dir), "--steps", "3"],
    )

    assert second.exit_code == 0, second.output
    resumed_prompt = captured[-1]
    assert resumed_prompt[-1]["role"] == "user"
    assert "turn two" in resumed_prompt[-1]["content"]
    assert any("turn one" in (m.get("content") or "") for m in resumed_prompt)
    assert any("SUMMARY: steps one and two." in (m.get("content") or "")
               for m in resumed_prompt)


# ---------------------------------------------------------------------------
# --context-window: the flag overrides compact.context_window per invocation,
# so the trigger is threshold_fraction x the flag's window.
# ---------------------------------------------------------------------------

def _cli_episode_with_window(runner, tmp_path, monkeypatch, window, task_file):
    """One CLI episode over a pinned two-step history, with --context-window.

    Injected prompt_tokens land the step-3 anchored estimate strictly between
    0.5 x 10000 and 0.5 x 20000, so window 10000 compacts and window 20000
    does not under the patched config's threshold_fraction of 0.5.
    """
    replies = iter([
        assistant(tool_calls=[tool_call(1, command=_big_output(8000, "a"))]),
        assistant(tool_calls=[tool_call(2, command=_big_output(8000, "b"))]),
        assistant("done"),
    ])
    tokens = iter([100, 4100, 100])

    def query(self, messages, tools=None):
        self.last_prompt_tokens = next(tokens)
        return next(replies)

    monkeypatch.setattr(Model, "query", query)
    return runner.invoke(
        _cli_app(),
        ["--task-file", task_file("trigger arithmetic"), "--json",
         "--cwd", str(tmp_path),
         "--session-dir", str(tmp_path / "sessions"), "--steps", "5",
         "--context-window", str(window)],
    )


def test_context_window_flag_scales_the_trigger_multiplicatively(
    tmp_path, monkeypatch, task_file
):
    """Same history, same injected prompt_tokens: window W fires a threshold
    compaction, window 2W does not (asserting threshold_fraction x effective
    window, not a single point)."""
    real_load_config = load_config

    def patched_load_config():
        config = real_load_config()
        config["compact"] = _compact()
        return config

    monkeypatch.setattr(cli_main, "load_config", patched_load_config)
    runner = CliRunner()
    window = 10000  # threshold = 0.5 x window

    fired = _cli_episode_with_window(runner, tmp_path, monkeypatch, window, task_file)
    assert fired.exit_code == 0, fired.output
    events = [json.loads(line) for line in fired.stdout.splitlines()]
    compact_events = [e for e in events if e["type"] == "compact"]
    assert len(compact_events) == 1
    event = compact_events[0]
    assert event["trigger"] == "threshold"
    assert event["layer"] == "mask"
    # The estimate sits between the two windows' thresholds: it is what must
    # have decided both directions.
    assert 0.5 * window <= event["pre_tokens"] < 0.5 * 2 * window

    spared = _cli_episode_with_window(runner, tmp_path, monkeypatch, 2 * window, task_file)
    assert spared.exit_code == 0, spared.output
    events = [json.loads(line) for line in spared.stdout.splitlines()]
    assert not any(e["type"] == "compact" for e in events)


@pytest.mark.parametrize("bad_value", ["0", "-7"])
def test_context_window_below_one_is_rejected_naming_the_flag(
    tmp_path, bad_value, task_file
):
    result = CliRunner().invoke(
        _cli_app(),
        ["--task-file", task_file("window validation"), "--json",
         "--cwd", str(tmp_path),
         "--session-dir", str(tmp_path / "sessions"),
         f"--context-window={bad_value}"],
    )
    assert result.exit_code != 0
    assert "--context-window" in result.stderr


# ---------------------------------------------------------------------------
# Acceptance 6: the per-step entry budget, both variants; the control-marker
# gate still runs on the complete output first.
# ---------------------------------------------------------------------------

def test_single_oversized_observation_is_truncated_head_note_tail(tmp_path, templates):
    command = _big_output(30000, "a")
    compact = _compact(context_window=10 ** 9,
                       step_observation_budget_chars=1000)
    s1 = assistant(tool_calls=[tool_call(1, command=command)])
    agent = _agent(tmp_path, templates, (s1, 100), (assistant("done"), 100),
                   compact=compact)

    result = agent.run("one oversized observation")

    assert result["completed"] is True
    observation = [m for m in agent.messages if m["role"] == "tool"][0]["content"]
    expected_original = len("Exit code: 0\nOutput:\n") + 30000 + 1
    assert len(observation) == 1000
    assert observation.startswith("Exit code: 0\nOutput:\n" + "a" * 100)
    assert observation.endswith("a" * 100 + "\n")
    assert (f"observation truncated by the per-step budget: original was "
            f"{expected_original} chars") in observation
    # The session went on: the model saw the bounded observation and finished.
    assert result["steps"][0]["observation"] == observation


def test_multi_call_step_over_budget_gets_note_replacements(tmp_path, templates):
    calls = [tool_call(i + 1, command=_big_output(800, chr(ord("a") + i)))
             for i in range(4)]
    compact = _compact(context_window=10 ** 9,
                       step_observation_budget_chars=1000)
    s1 = assistant(tool_calls=calls)
    agent = _agent(tmp_path, templates, (s1, 100), (assistant("done"), 100),
                   compact=compact)

    result = agent.run("four calls each under budget, aggregate over")

    assert result["completed"] is True
    contents = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert len(contents) == 4
    # Call 1 fits whole (822 chars); call 2 exceeds the remaining 178 and is
    # truncated to exactly the remaining budget; calls 3-4 run normally but
    # enter history as one-line notes stating the original length.
    assert len(contents[0]) == 822
    assert contents[0].startswith("Exit code: 0\nOutput:\na")
    assert len(contents[1]) == 178
    assert "observation truncated by the per-step budget: original was 822" \
           in contents[1]
    for note in contents[2:]:
        assert note.startswith("[observation not recorded: the per-step "
                               "budget was exhausted; original was 822 chars.")
        assert "Re-run the command with filters" in note
    # The step's observation payload entering history stays within the budget
    # (the one-line replacement notes are the sanctioned bounded overrun).
    payload = sum(len(c) for c in contents[:2])
    assert payload == 1000
    assert [step["command"] for step in result["steps"]] == [
        json.loads(c["function"]["arguments"])["command"] for c in calls
    ]


def test_marker_gate_runs_on_complete_output_before_any_budget(tmp_path, templates):
    command = "printf '<|open|>'; " + _big_output(30000, "a")
    compact = _compact(context_window=10 ** 9,
                       step_observation_budget_chars=1000)
    s1 = assistant(tool_calls=[tool_call(1, command=command)])
    agent = _agent(tmp_path, templates, (s1, 100), (assistant("done"), 100),
                   compact=compact)

    agent.run("oversized output carrying a marker")

    observation = [m for m in agent.messages if m["role"] == "tool"][0]["content"]
    assert "withheld" in observation
    assert "a" * 100 not in observation
    assert "truncated by the per-step budget" not in observation


# ---------------------------------------------------------------------------
# Helper-level invariants.
# ---------------------------------------------------------------------------

def test_bound_observation_variants():
    whole, charged = bound_observation("x" * 100, remaining=500)
    assert (whole, charged) == ("x" * 100, 100)

    truncated, charged = bound_observation("y" * 1000, remaining=200)
    assert charged == len(truncated) == 200
    assert truncated.startswith("y" * 40)
    assert truncated.endswith("y" * 40)
    assert "original was 1000 chars" in truncated

    note, charged = bound_observation("z" * 1000, remaining=0)
    assert charged == 0
    assert note.count("\n") == 0  # one line
    assert "original was 1000 chars" in note
    assert "Re-run the command with filters" in note


def test_verbatim_tail_keeps_whole_steps_and_at_least_the_last():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for index in range(3):
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [tool_call(index, command="true")]})
        messages.append({"role": "tool", "tool_call_id": f"call-{index}",
                         "content": "x" * 1600})

    start, end = verbatim_tail_span(messages, tail_budget_tokens=100)
    # The last step alone (1600+ chars) already exceeds the 400-char budget and
    # is still kept whole.
    assert messages[start:end] == messages[-2:]

    start, end = verbatim_tail_span(messages, tail_budget_tokens=1000)
    assert messages[start:end] == messages[-4:]


def test_drop_oldest_middle_half_cuts_on_a_step_boundary():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for index in range(4):
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [tool_call(index, command="true")]})
        messages.append({"role": "tool", "tool_call_id": f"call-{index}",
                         "content": "x" * 100})

    dropped = drop_oldest_middle_half(messages, tail_start=len(messages))

    assert dropped == 4  # half of an 8-message middle, two whole steps
    assert [m.get("tool_call_id") for m in messages if m["role"] == "tool"] == [
        "call-2", "call-3",
    ]
    for (start, end) in split_steps(messages):
        ids = [m["tool_call_id"] for m in messages[start + 1:end]]
        wanted = [c["id"] for c in messages[start]["tool_calls"]]
        assert ids == wanted


def test_split_steps_groups_assistant_with_its_tool_messages():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "t"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "tool", "tool_call_id": "b", "content": "2"},
        {"role": "assistant", "content": "plain"},
        {"role": "user", "content": "reminder"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_call_id": "c", "content": "3"},
    ]
    assert split_steps(messages) == [(2, 5), (5, 6), (7, 9)]


def test_est_messages_tokens_counts_content_reasoning_and_calls():
    messages = [
        {"role": "assistant", "content": "x" * 40,
         "reasoning_content": "y" * 40,
         "tool_calls": [tool_call(1, command="echo hi")]},
        {"role": "tool", "tool_call_id": "call-1", "content": "z" * 40},
    ]
    assert est_messages_tokens(messages) * 4 >= 120
