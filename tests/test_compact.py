"""The compaction ladder: entry budget, the four levels, the target line,
over-window fallback, persistence, and the compact event. All offline: the model
is a scripted stand-in injecting usage values and over-window errors.
"""

import json
import re
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import main as cli_main
from agent import Agent, _load_state, load_config
from compact import (
    COMMAND_ELISION_MARKER,
    ELISION_NOTE,
    MASK_SENTINEL,
    bound_observation,
    drop_old_reasoning,
    drop_oldest_middle_half,
    elide_old_commands,
    est_message_chars,
    est_messages_tokens,
    mask_old_observations,
    split_steps,
    verbatim_tail_span,
)
from conftest import assistant, final_answer, tool_call
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
        self.model_name = "openai/fake-model"
        self.last_prompt_tokens = None
        self.last_cached_tokens = 0
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
        "threshold_fraction": 0.5,  # threshold line: 5000 tokens
        "target_fraction": 0.5,     # target line: 5000 tokens (floor is lower)
        "min_gain_tokens": 0,
        "image_tokens": 1600,
        "ladder": ["mask", "reasoning", "command", "summarize"],
        "keep_tail_tokens": 300,
        "tail_budget_tokens": 300,  # 1200 chars
        "command_head_chars": 200,
        "step_observation_budget_chars": 40000,
    }
    cfg.update(overrides)
    return cfg


def _agent(tmp_path, templates, *steps, compact=None, emit=None, state_file=None,
           step_limit=20, resume=False, images=()):
    return Agent(
        model=CompactModel(*steps),
        environment=Environment(cwd=str(tmp_path), timeout=10,
                                log_dir=str(tmp_path)),
        templates=templates,
        step_limit=step_limit,
        emit=emit,
        state_file=state_file,
        resume=resume,
        images=images,
        compact=compact if compact is not None else _compact(),
    )


def _big_output(chars, letter):
    return f"head -c {chars} /dev/zero | tr '\\0' {letter}"


# ---------------------------------------------------------------------------
# Acceptance 1: a threshold-triggered ladder pass keeps assistant messages
# byte-identical, tool_call_id pairing complete, old observations placeholders.
# ---------------------------------------------------------------------------

def test_mask_pass_preserves_assistant_verbatim_and_pairing(tmp_path, templates):
    s1 = assistant(tool_calls=[tool_call(1, command=_big_output(1000, "a"))])
    s2 = assistant(tool_calls=[tool_call(2, command=_big_output(1000, "b"))])
    s3 = assistant("working", tool_calls=[tool_call(3, command=_big_output(1000, "c"))],
                   reasoning_content="step-three thinking")
    s4 = final_answer("all done")
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
    # Old observations (outside the verbatim tail) are placeholders keeping the
    # exit code, the original length, and the log file; the newest step stays
    # verbatim.
    for masked in tool_messages[:2]:
        assert masked["content"].startswith(MASK_SENTINEL)
        assert "exit code was 0" in masked["content"]
        assert re.search(r"original was \d{4} chars", masked["content"])
        assert "Full text:" in masked["content"]
    assert "c" * 200 in tool_messages[2]["content"]
    assert not tool_messages[2]["content"].startswith(MASK_SENTINEL)
    # Masking alone brought the estimate under the target line: no deeper level.
    compact_events = [e for e in events if e["type"] == "compact"]
    assert len(compact_events) == 1
    event = compact_events[0]
    assert set(event) == {"type", "step", "layers", "layer", "trigger",
                          "pre_tokens", "post_tokens_est", "target_tokens",
                          "floor_tokens", "steps_since_last_compact",
                          "invalidated_from_index"}
    assert event["step"] == 4
    assert event["layers"] == ["mask"]
    assert event["layer"] == "mask"
    assert event["trigger"] == "threshold"
    assert event["pre_tokens"] >= 5000
    assert 0 < event["post_tokens_est"] < event["pre_tokens"]
    assert event["target_tokens"] == 5000
    assert event["floor_tokens"] > 0
    assert event["steps_since_last_compact"] == 4
    assert event["invalidated_from_index"] == 3


def test_mask_is_idempotent():
    messages = []
    for index in range(4):
        messages.append({"role": "assistant",
                         "content": "",
                         "tool_calls": [tool_call(index, command="true")],
                         "reasoning_content": f"thinking {index}"})
        messages.append({"role": "tool", "tool_call_id": f"call-{index}",
                         "content": f"Exit code: 0\nOutput:\n{'x' * 500}"})

    log_paths = {f"call-{index}": f"/logs/s-{index}-1.log" for index in range(4)}
    tail_start, _ = verbatim_tail_span(messages, tail_budget_tokens=300)
    first = mask_old_observations(messages, tail_start, log_paths)
    snapshot = json.dumps(messages, sort_keys=True)
    second = mask_old_observations(messages, tail_start, log_paths)

    assert first > 0
    assert second == 0
    assert json.dumps(messages, sort_keys=True) == snapshot


@pytest.mark.parametrize("old_output_chars, keep_tail_tokens, expected_layer", [
    pytest.param(1000, 300, "summarize", id="insufficient-masking"),
    pytest.param(8000, 300, "mask", id="effective-masking"),
    pytest.param(1000, 4000, "summarize", id="no-eligible-observations"),
    pytest.param(0, 300, "summarize", id="short-observation-not-masked"),
])
def test_threshold_compaction_preserves_measured_token_calibration(
    tmp_path, templates, old_output_chars, keep_tail_tokens, expected_layer
):
    s1 = assistant(tool_calls=[tool_call(1, command=_big_output(old_output_chars, "a"))])
    s2 = assistant(tool_calls=[tool_call(2, command=_big_output(1000, "b"))],
                   reasoning_content="Keep this reasoning in the verbatim tail.")
    s3 = assistant(tool_calls=[tool_call(3, command="true")])
    replies = [(s1, 100), (s2, 6000)]
    if expected_layer == "summarize":
        # The summary measures the old history; only the next regular call can
        # measure the rebuilt history and anchor subsequent step boundaries.
        replies.append((assistant("The earlier observation has been reviewed."), 6200))
    replies.extend([(s3, 100), (final_answer("finished"), 100)])
    events = []
    agent = _agent(tmp_path, templates, *replies, emit=events.append,
                   compact=_compact(keep_tail_tokens=keep_tail_tokens))

    result = agent.run("review observations then finish")

    compact_events = [e for e in events if e["type"] == "compact"]
    assert [e["layer"] for e in compact_events] == [expected_layer]
    event = compact_events[0]
    assert event["step"] == 3
    assert event["trigger"] == "threshold"
    assert event["pre_tokens"] > 6000 > 2 * est_messages_tokens(agent.model.seen_prompts[1])
    assert result["completed"] is True
    assert result["final_output"] == "finished"

    masked_history = agent.model.seen_prompts[2]
    original_observation = agent.model.seen_prompts[1][-1]
    if keep_tail_tokens == 300:
        masked_observation = masked_history[3]
        assert masked_observation["tool_call_id"] == original_observation["tool_call_id"]
        if old_output_chars == 0:
            # A placeholder is longer than a near-empty observation, and masking
            # would grow the history, so the short observation is left as-is.
            assert masked_observation["content"] == original_observation["content"]
            assert not masked_observation["content"].startswith(MASK_SENTINEL)
        else:
            assert masked_observation["content"].startswith(MASK_SENTINEL)
    else:
        assert masked_history[3] == original_observation
        assert not any(m["content"].startswith(MASK_SENTINEL)
                       for m in masked_history if m["role"] == "tool")

    if expected_layer == "summarize":
        assert [tools is None for tools in agent.model.seen_tools] == [
            False, False, True, False, False,
        ]
        assert masked_history[-1] == {"role": "user", "content": templates["compact_prompt"]}
        rebuilt_history = agent.model.seen_prompts[3]
        assert [m["role"] for m in rebuilt_history] == [
            "system", "user", "user", "assistant", "tool",
        ]
        assert event["post_tokens_est"] == est_messages_tokens(rebuilt_history)
        assert event["post_tokens_est"] < 5000
        assert rebuilt_history[-2:] == masked_history[-3:-1]
    else:
        assert all(tools is not None for tools in agent.model.seen_tools)
        assert 5000 > event["post_tokens_est"] > est_messages_tokens(masked_history)
        assert [m for m in agent.messages if m["role"] == "assistant"] == [
            s1[0], s2[0], s3[0], final_answer("finished")[0],
        ]
    assert s2[0] in agent.messages
    assert any(m["role"] == "tool" and m["tool_call_id"] == "call-2"
               and "b" * 1000 in m["content"] for m in agent.messages)
    assert [e["prompt_tokens"] for e in events if e["type"] == "context"] == [
        100, 6000, 100, 100,
    ]


def test_already_masked_observations_escalate_after_next_measurement(tmp_path, templates):
    s1 = assistant(tool_calls=[tool_call(1, command=_big_output(8000, "a"))])
    events = []
    agent = _agent(tmp_path, templates,
                   (s1, 100), (assistant(""), 6000),
                   (assistant("working on"), 6100),
                   (assistant("The observation has been reviewed."), 100),
                   (final_answer("finished"), 100),
                   emit=events.append, compact=_compact(keep_tail_tokens=0))

    result = agent.run("review one observation then finish")

    compact_events = [e for e in events if e["type"] == "compact"]
    assert [e["layer"] for e in compact_events] == ["mask", "summarize"]
    assert [e["step"] for e in compact_events] == [3, 4]
    assert compact_events[0]["post_tokens_est"] < 5000
    assert compact_events[1]["pre_tokens"] > 6000
    assert [tools is None for tools in agent.model.seen_tools] == [
        False, False, False, True, False,
    ]
    after_first_mask = [m for m in agent.model.seen_prompts[2] if m["role"] == "tool"]
    before_summary = [m for m in agent.model.seen_prompts[3] if m["role"] == "tool"]
    assert len(after_first_mask) == 1
    assert after_first_mask[0]["content"].startswith(MASK_SENTINEL)
    assert before_summary == after_first_mask
    assert result["completed"] is True
    assert result["final_output"] == "finished"


def test_mask_estimate_stays_nonnegative_when_character_savings_exceed_usage(
    tmp_path, templates
):
    big_step = assistant(tool_calls=[tool_call(1, command=_big_output(40000, "x"))])
    small_step = assistant(tool_calls=[tool_call(2, command="true")])
    events = []
    agent = _agent(tmp_path, templates,
                   (big_step, 100), (small_step, 4990), (final_answer("finished"), 100),
                   emit=events.append)

    result = agent.run("one huge observation then finish")

    compact_events = [e for e in events if e["type"] == "compact"]
    assert [e["step"] for e in compact_events] == [2, 3]
    # First pass: the only span is the tail, so nothing is eligible and the
    # estimate falls back to the bare measured anchor.
    assert compact_events[0]["layers"] == ["mask"]
    assert compact_events[0]["post_tokens_est"] == 100
    # Second pass: masking the huge observation frees far more characters than
    # the measured prompt tokens account for; the anchored estimate floors at 0.
    assert compact_events[1]["layers"] == ["mask"]
    assert compact_events[1]["pre_tokens"] >= 5000
    assert compact_events[1]["post_tokens_est"] == 0
    assert all(tools is not None for tools in agent.model.seen_tools)
    assert agent.model.seen_prompts[2][3]["content"].startswith(MASK_SENTINEL)
    assert result["completed"] is True
    assert result["final_output"] == "finished"


# ---------------------------------------------------------------------------
# The ladder: stops at the target line; escalates level by level; the summarize
# level backstops; pairing and the non-arguments tool_calls fields survive.
# ---------------------------------------------------------------------------

def test_ladder_stops_at_the_target_line(tmp_path, templates):
    """One compaction buys tens of steps: the ladder stops once the estimate is
    at or below the target line, and the levels it never reached leave no
    trace on the history."""
    s1 = assistant(tool_calls=[tool_call(1, command=_big_output(20000, "a"))],
                   reasoning_content="step-one thinking")
    s2 = assistant(tool_calls=[tool_call(2, command=_big_output(1000, "b"))])
    events = []
    agent = _agent(tmp_path, templates,
                   (s1, 100), (s2, 6000), (final_answer("done"), 100),
                   emit=events.append,
                   compact=_compact(target_fraction=0.3, min_gain_tokens=3000))

    result = agent.run("compact down to thirty percent")

    compact_events = [e for e in events if e["type"] == "compact"]
    assert len(compact_events) == 1
    event = compact_events[0]
    assert event["layers"] == ["mask"]
    assert event["layer"] == "mask"
    assert event["target_tokens"] == 3000
    assert event["floor_tokens"] < event["target_tokens"]
    assert event["post_tokens_est"] <= event["target_tokens"]
    assert event["post_tokens_est"] < event["pre_tokens"]
    # The levels the ladder never reached left no trace.
    stored = [m for m in agent.messages if m["role"] == "assistant"]
    assert stored[0].get("reasoning_content") == "step-one thinking"
    assert stored[0]["tool_calls"][0]["function"]["arguments"] == \
        s1[0]["tool_calls"][0]["function"]["arguments"]
    # The newest step stays verbatim.
    tool_messages = [m for m in agent.messages if m["role"] == "tool"]
    assert "b" * 200 in tool_messages[1]["content"]
    assert result["completed"] is True


def test_ladder_escalates_through_every_level_and_backstops_with_summarize(
    tmp_path, templates
):
    """A target line the first three levels cannot reach: the ladder runs all
    four in order, every lifted-out text round-trips from the session log
    directory, pairing survives, and only `arguments` changes on tool_calls."""
    long_command = "echo " + "y" * 1000
    reasoning_text = "step-one reasoning " * 5
    s1 = assistant(tool_calls=[tool_call(1, command=long_command)],
                   reasoning_content=reasoning_text)
    s2 = assistant(tool_calls=[tool_call(2, command="true")])
    summary_text = "1. Progress: the long command ran. 5. Remaining: finish."
    s_summary = assistant(summary_text)
    s_done = final_answer("wrapped up")
    events = []
    agent = _agent(tmp_path, templates,
                   (s1, 100), (s2, 6000), (s_summary, 100), (s_done, 100),
                   emit=events.append, compact=_compact(target_fraction=0.05))

    result = agent.run("escalate through the whole ladder")

    compact_events = [e for e in events if e["type"] == "compact"]
    assert len(compact_events) == 1
    event = compact_events[0]
    assert event["layers"] == ["mask", "reasoning", "command", "summarize"]
    assert event["layer"] == "summarize"
    assert event["trigger"] == "threshold"
    assert event["invalidated_from_index"] == 2
    assert [tools is None for tools in agent.model.seen_tools] == [
        False, False, True, False,
    ]
    assert result["completed"] is True

    # The summarize prompt saw the ladder-rewritten history.
    summary_prompt = agent.model.seen_prompts[2]
    rewritten = [m for m in summary_prompt if m.get("tool_calls")][0]
    original_call = s1[0]["tool_calls"][0]
    rewritten_call = rewritten["tool_calls"][0]
    # Every tool_calls field except `arguments` is byte-identical: Gemini's
    # thought signature rides on those fields and a request missing it is
    # rejected.
    assert rewritten_call["id"] == original_call["id"]
    assert rewritten_call["type"] == original_call["type"]
    assert rewritten_call["index"] == original_call["index"]
    assert rewritten_call["function"]["name"] == original_call["function"]["name"]
    payload = json.loads(rewritten_call["function"]["arguments"])
    assert payload["command"].startswith("echo ")
    assert COMMAND_ELISION_MARKER in payload["command"]
    # The elided command body round-trips from the file the note names.
    command_note_path = re.search(
        r"full text: (\S+) \.\.\.\]", payload["command"]
    ).group(1)
    assert Path(command_note_path).read_text(encoding="utf-8") == long_command
    # The dropped reasoning round-trips from the session log directory too.
    assert (Path(agent.environment.log_dir) / "s-1.reasoning.txt").read_text(
        encoding="utf-8"
    ) == reasoning_text
    # The masked observation round-trips: the file at the placeholder's path
    # holds the command's full output, and the observation that was replaced is
    # exactly the observation template wrapped around it.
    masked_tool = [m for m in summary_prompt if m["role"] == "tool"
                   and m["tool_call_id"] == "call-1"][0]
    assert masked_tool["content"].startswith(MASK_SENTINEL)
    log_path = re.search(r"Full text: (\S+)\]", masked_tool["content"]).group(1)
    log_text = Path(log_path).read_text(encoding="utf-8")
    replaced = f"Exit code: 0\nOutput:\n{log_text}\n"
    stated = int(re.search(r"original was (\d+) chars",
                           masked_tool["content"]).group(1))
    assert stated == len(replaced)

    # The rebuilt history is [system, task, summary, verbatim tail]; the two
    # rewritten steps are now small enough that both fit the tail budget, and
    # pairing holds end to end.
    assert [m["role"] for m in agent.messages] == [
        "system", "user", "user", "assistant", "tool", "assistant", "tool",
        "assistant",
    ]
    assert [m["tool_call_id"] for m in agent.messages if m["role"] == "tool"] == [
        c["id"] for s in (s1, s2) for c in s[0]["tool_calls"]
    ]
    assert event["post_tokens_est"] == est_messages_tokens(agent.model.seen_prompts[3])


def test_ladder_levels_are_idempotent():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for index in range(3):
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [tool_call(index + 1,
                                                  command="echo " + "y" * 1000)],
                         "reasoning_content": f"thinking {index} " * 10})
        messages.append({"role": "tool", "tool_call_id": f"call-{index + 1}",
                         "content": f"Exit code: 0\nOutput:\n{'x' * 2000}"})
    tail_start, _ = verbatim_tail_span(messages, tail_budget_tokens=100)
    log_paths = {f"call-{index + 1}": f"/logs/s-{index + 1}-1.log"
                 for index in range(3)}
    reasoning_writes = []
    command_writes = []

    def save_reasoning(text, ordinal):
        path = f"/logs/s-{ordinal}.reasoning.txt"
        reasoning_writes.append((path, text))
        return path

    def save_command(text, ordinal, call_index):
        path = f"/logs/s-{ordinal}-{call_index}.command.txt"
        command_writes.append((path, text))
        return path

    assert mask_old_observations(messages, tail_start, log_paths) > 0
    assert drop_old_reasoning(messages, tail_start, save=save_reasoning) > 0
    assert elide_old_commands(messages, tail_start, 200, save=save_command) > 0
    assert len(reasoning_writes) == 2
    assert len(command_writes) == 2
    snapshot = json.dumps(messages, sort_keys=True)

    # A second pass over the rewritten history frees nothing and writes nothing.
    assert mask_old_observations(messages, tail_start, log_paths) == 0
    assert drop_old_reasoning(messages, tail_start, save=save_reasoning) == 0
    assert elide_old_commands(messages, tail_start, 200, save=save_command) == 0
    assert len(reasoning_writes) == 2
    assert len(command_writes) == 2
    assert json.dumps(messages, sort_keys=True) == snapshot


def test_command_level_skips_unparseable_arguments_and_preserves_other_fields():
    def call_with(arguments):
        return {"id": "call-1", "type": "function", "index": 0,
                "extra_content": {"google": {"thoughtSignature": "sig"}},
                "function": {"name": "bash", "arguments": arguments}}

    long_command = "echo " + "y" * 1000
    messages = [{"role": "assistant", "content": "", "tool_calls": [
        call_with("{not json"),
        call_with(json.dumps({"command": long_command})),
        call_with(json.dumps({"command": "short"})),
        call_with(json.dumps({"cwd": "/tmp", "command": long_command})),
    ]}]
    saved = []

    def save(text, ordinal, call_index):
        path = f"/logs/s-{ordinal}-{call_index}.command.txt"
        saved.append((path, text))
        return path

    freed = elide_old_commands(messages, tail_start=1, head_chars=200, save=save)

    calls = messages[0]["tool_calls"]
    # A payload that does not parse is skipped, not rewritten.
    assert calls[0]["function"]["arguments"] == "{not json"
    # A command at or under the head length is untouched, not by a character.
    assert calls[2]["function"]["arguments"] == json.dumps({"command": "short"})
    # The long ones are elided; every field outside `arguments` is byte-identical.
    payload = json.loads(calls[1]["function"]["arguments"])
    assert payload["command"].startswith("echo ")
    assert COMMAND_ELISION_MARKER in payload["command"]
    assert calls[1]["id"] == "call-1"
    assert calls[1]["type"] == "function"
    assert calls[1]["index"] == 0
    assert calls[1]["extra_content"] == {"google": {"thoughtSignature": "sig"}}
    assert calls[1]["function"]["name"] == "bash"
    # Extra payload keys ride along.
    payload4 = json.loads(calls[3]["function"]["arguments"])
    assert payload4["cwd"] == "/tmp"
    assert COMMAND_ELISION_MARKER in payload4["command"]
    assert saved == [(f"/logs/s-1-{call_index}.command.txt", long_command)
                     for call_index in (2, 4)]
    assert freed > 0

    # A second pass is a no-op: the elision marker makes it idempotent.
    saved.clear()
    assert elide_old_commands(messages, tail_start=1, head_chars=200, save=save) == 0
    assert saved == []


# ---------------------------------------------------------------------------
# Acceptance 2: when the first three levels are not enough the history is
# exactly [system, task, summary, tail] with the tail in whole steps.
# ---------------------------------------------------------------------------

def test_summarize_rebuilds_history_when_masking_is_not_enough(tmp_path, templates):
    s1 = assistant(tool_calls=[tool_call(1, command=_big_output(2000, "a"))])
    s2 = assistant(tool_calls=[tool_call(2, command=_big_output(20000, "b"))])
    summary_text = "1. Progress: steps one and two done. 5. Remaining: finish."
    s_summary = assistant(summary_text)
    s3 = final_answer("wrapping up")
    events = []
    agent = _agent(tmp_path, templates,
                   (s1, 100), (s2, 6000), (s_summary, 100), (s3, 100),
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
    # The summary call used the ladder-rewritten history plus the compact
    # prompt, no tools.
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
    assert compact_events[0]["layers"] == ["mask", "reasoning", "command",
                                           "summarize"]
    assert compact_events[0]["trigger"] == "threshold"


# ---------------------------------------------------------------------------
# Acceptance 3: over-window error -> compaction + single retry; second failure
# raises; unrelated 400s propagate.
# ---------------------------------------------------------------------------

def test_overflow_compacts_and_retries_once_then_completes(tmp_path, templates):
    s_summary = assistant("1. Progress: nothing yet. 5. Remaining: everything.")
    s_done = final_answer("finished after retry")
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
    assert set(compact_events[0]) == {"type", "step", "layers", "layer",
                                      "trigger", "pre_tokens",
                                      "post_tokens_est", "target_tokens",
                                      "floor_tokens",
                                      "steps_since_last_compact",
                                      "invalidated_from_index"}
    assert compact_events[0]["trigger"] == "overflow"
    # Overflow goes straight to the summarize level: the endpoint refused the
    # call, so there is no trustworthy measured anchor for the ladder's
    # anchored arithmetic at that moment.
    assert compact_events[0]["layers"] == ["summarize"]
    assert compact_events[0]["layer"] == "summarize"
    assert compact_events[0]["pre_tokens"] == est_messages_tokens(agent.model.seen_prompts[0])
    assert compact_events[0]["post_tokens_est"] == est_messages_tokens(agent.model.seen_prompts[2])
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
    s_done = final_answer("recovered")
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
    s_done = final_answer("done")
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
# The floor: startup preflight, and no per-step compaction when the floor is
# already high.
# ---------------------------------------------------------------------------

def test_floor_above_the_threshold_never_compacts_per_step(tmp_path, templates, capsys):
    """A session whose floor is already high must not pay a full re-prefill on
    every step: the min-gain condition holds the ladder off, the history keeps
    growing, and the over-window path handles the real ceiling."""
    events = []
    agent = _agent(tmp_path, templates,
                   (assistant(tool_calls=[tool_call(1, command="true")]), 6000),
                   (assistant(tool_calls=[tool_call(2, command="true")]), 7000),
                   (final_answer("done"), 100),
                   compact=_compact(keep_tail_tokens=6000, min_gain_tokens=16384),
                   emit=events.append)

    result = agent.run("grow the history past the threshold")

    assert result["completed"] is True
    assert not any(e["type"] == "compact" for e in events)
    err = capsys.readouterr().err
    assert "warning" in err
    assert "floor" in err


def test_floor_at_or_above_the_window_fails_at_startup_naming_the_parts(
    tmp_path, templates
):
    agent = _agent(tmp_path, templates, (final_answer("done"), 100),
                   compact=_compact(keep_tail_tokens=12000))

    with pytest.raises(RuntimeError) as excinfo:
        agent.run("an impossible floor")

    message = str(excinfo.value)
    assert "reaches the context window" in message
    assert "system=" in message
    assert "task=" in message
    assert "tail=12000" in message


# ---------------------------------------------------------------------------
# Acceptance 4: state file is readable right after compaction and --resume works.
# ---------------------------------------------------------------------------

def test_state_file_readable_after_compaction_and_agent_resume(tmp_path, templates):
    s1 = assistant(tool_calls=[tool_call(1, command=_big_output(2000, "a"))])
    s2 = assistant(tool_calls=[tool_call(2, command=_big_output(20000, "b"))])
    s_summary = assistant("1. Progress: two steps summarized.")
    s_done = final_answer("turn one done")
    state_file = tmp_path / "session.json"
    agent = _agent(tmp_path, templates,
                   (s1, 100), (s2, 6000), (s_summary, 100), (s_done, 100),
                   state_file=str(state_file))

    agent.run("turn one")

    # Persisted right at compaction time, loadable, and identical to memory.
    state = json.loads(state_file.read_text())
    assert state["protocol"] == "tool-calls-v1"
    assert state["messages"] == agent.messages
    assert _load_state(state_file) == agent.messages

    # An Agent --resume on the compacted state continues the session.
    resumed = _agent(tmp_path, templates, (final_answer("turn two done"), 100),
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
        final_answer("turn one done"),
        final_answer("turn two done"),
    ])
    tokens = iter([100, 6000, 100, 100, 100])
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
        final_answer("done"),
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
    agent = _agent(tmp_path, templates, (s1, 100), (final_answer("done"), 100),
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
    # The note names the log file holding the full text, and that file exists.
    log_path = re.search(r"Full text: (\S+) \.\.\.\]", observation).group(1)
    assert log_path == str(Path(agent.environment.log_dir) / "s-1-1.log")
    assert len(Path(log_path).read_text()) == 30000
    # The session went on: the model saw the bounded observation and finished.
    assert result["steps"][0]["observation"] == observation


def test_multi_call_step_over_budget_gets_note_replacements(tmp_path, templates):
    calls = [tool_call(i + 1, command=_big_output(800, chr(ord("a") + i)))
             for i in range(4)]
    compact = _compact(context_window=10 ** 9,
                       step_observation_budget_chars=1000)
    s1 = assistant(tool_calls=calls)
    agent = _agent(tmp_path, templates, (s1, 100), (final_answer("done"), 100),
                   compact=compact)

    result = agent.run("four calls each under budget, aggregate over")

    assert result["completed"] is True
    contents = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert len(contents) == 4
    # Call 1 fits whole (822 chars); call 2 exceeds the remaining 178 and is
    # truncated to exactly the remaining budget; calls 3-4 run normally but
    # enter history as one-line notes stating the original length and the log
    # file holding the full text.
    assert len(contents[0]) == 822
    assert contents[0].startswith("Exit code: 0\nOutput:\na")
    # Call 2 exceeds the remaining 178, but the elision note now carries the
    # log path and no longer fits that remainder, so it is replaced too.
    for index, note in enumerate(contents[1:], start=2):
        assert note.startswith("[observation not recorded: the per-step "
                               "budget was exhausted; original was 822 chars.")
        assert note.count("\n") == 0  # one line
        assert "Full text:" in note
        log_path = re.search(r"Full text: (\S+)\.\]", note).group(1)
        assert Path(log_path).read_text() == chr(ord("a") + index - 1) * 800
    # The step's charged payload stays within the budget: call 1 fit whole
    # (822 of the 1000 charged); from call 2 on the elision note no longer
    # fits the remainder, so each observation becomes a one-line replacement
    # note charging 0 (the sanctioned bounded overrun).
    assert [step["command"] for step in result["steps"]] == [
        json.loads(c["function"]["arguments"])["command"] for c in calls
    ]


def test_marker_gate_runs_on_complete_output_before_any_budget(tmp_path, templates):
    command = "printf '<|open|>'; " + _big_output(30000, "a")
    compact = _compact(context_window=10 ** 9,
                       step_observation_budget_chars=1000)
    s1 = assistant(tool_calls=[tool_call(1, command=command)])
    agent = _agent(tmp_path, templates, (s1, 100), (final_answer("done"), 100),
                   compact=compact)

    agent.run("oversized output carrying a marker")

    observation = [m for m in agent.messages if m["role"] == "tool"][0]["content"]
    assert "withheld" in observation
    assert "a" * 100 not in observation
    assert "truncated by the per-step budget" not in observation


# ---------------------------------------------------------------------------
# The floor's estimator: image-bearing task messages.
# ---------------------------------------------------------------------------

def test_est_message_chars_sums_parts_lists_per_part():
    message = {"role": "user", "content": [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}
    assert est_message_chars(message, image_tokens=1600) == len("hello") + 1600 * 4
    # A parts list estimated without the configured constant contributes only
    # its text, so a caller estimating a history that can hold the task message
    # must pass it.
    assert est_message_chars(message) == len("hello")


def test_image_task_message_is_estimated_from_the_configured_constant(
    tmp_path, templates
):
    png = tmp_path / "tiny.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    agent = _agent(tmp_path, templates, (final_answer("done"), 100), images=[png])

    task_message = agent._task_message("look at this")
    assert task_message["content"][1]["type"] == "image_url"
    # Four-chars-per-token on the base64 body would estimate this image at a
    # handful of tokens (the old len()-of-the-list bug estimated it as 1); the
    # configured constant says 1600.
    assert est_message_chars(task_message, image_tokens=1600) >= 1600 * 4
    agent.messages = [{"role": "system", "content": "s"}, task_message]
    assert agent._floor_tokens() >= 1600 + 300


# ---------------------------------------------------------------------------
# The masked observation reads back from the placeholder's path.
# ---------------------------------------------------------------------------

def test_masked_observation_round_trips_through_the_placeholder_path(
    tmp_path, templates
):
    s1 = assistant(tool_calls=[tool_call(1, command=_big_output(20000, "a"))])
    s2 = assistant(tool_calls=[tool_call(2, command="true")])
    agent = _agent(tmp_path, templates, (s1, 100), (s2, 8500),
                   (final_answer("done"), 100),
                   compact=_compact(min_gain_tokens=3000))

    result = agent.run("mask the big observation")

    assert result["completed"] is True
    tool_messages = [m for m in agent.messages if m["role"] == "tool"]
    placeholder = tool_messages[0]["content"]
    assert placeholder.startswith(MASK_SENTINEL)
    log_path = re.search(r"Full text: (\S+)\]", placeholder).group(1)
    # The file at the placeholder's path holds the command's full output, and
    # the observation that was replaced is byte-for-byte the observation
    # template wrapped around it.
    log_text = Path(log_path).read_text(encoding="utf-8")
    replaced = f"Exit code: 0\nOutput:\n{log_text}\n"
    stated = int(re.search(r"original was (\d+) chars", placeholder).group(1))
    assert stated == len(replaced)
    assert replaced == "Exit code: 0\nOutput:\n" + "a" * 20000 + "\n"
    # The placeholder is shorter than what it replaced.
    assert len(placeholder) < len(replaced)


# ---------------------------------------------------------------------------
# Helper-level invariants.
# ---------------------------------------------------------------------------

def test_bound_observation_variants():
    whole, charged = bound_observation("x" * 100, remaining=500,
                                       log_path="/logs/s-1-1.log")
    assert (whole, charged) == ("x" * 100, 100)

    truncated, charged = bound_observation("y" * 1000, remaining=200,
                                           log_path="/logs/s-1-1.log")
    assert charged == len(truncated) == 200
    note = ELISION_NOTE.format(original=1000, path="/logs/s-1-1.log")
    body = 200 - len(note)
    assert truncated.startswith("y" * (body // 2))
    assert truncated.endswith("y" * (body - body // 2))
    assert "original was 1000 chars" in truncated
    assert "/logs/s-1-1.log" in truncated

    note, charged = bound_observation("z" * 1000, remaining=0,
                                      log_path="/logs/s-1-1.log")
    assert charged == 0
    assert note.count("\n") == 0  # one line
    assert "original was 1000 chars" in note
    assert "/logs/s-1-1.log" in note


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
