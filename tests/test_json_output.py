import json

import litellm
import pytest
import typer
from litellm.exceptions import ContextWindowExceededError
from typer.testing import CliRunner

import main as cli_main
import model
from agent import Agent, load_config
from conftest import (FakeCompletionResponse, ScriptedModel, assistant, final_answer,
                      service_unavailable, tool_call)
from environment import Environment
from model import Model


def _cli_app():
    app = typer.Typer()
    app.command()(cli_main.run)
    return app


def _json_lines(output):
    return [json.loads(line) for line in output.splitlines()]


def _patch_model(monkeypatch, *responses, usage=None):
    replies = iter(responses)

    def query(self, messages, tools=None):
        return next(replies)

    monkeypatch.setattr(Model, "query", query)
    monkeypatch.setattr(
        Model,
        "usage",
        lambda self: usage or {"n_calls": 1, "input_tokens": 2, "output_tokens": 3},
    )


OVERFLOW = "OVERFLOW"


class UsageModel:
    """Stands in for Model: scripted (reply, prompt_tokens) steps.

    reply is an (assistant message, finish_reason) pair or the OVERFLOW sentinel
    (raises a context-window error and leaves usage unset, like the endpoint).
    prompt_tokens is that response's usage.prompt_tokens; None means the
    response carried no usage.
    """

    def __init__(self, *steps, model_name="openai/fake-model"):
        self.model_name = model_name
        self._steps = iter(steps)
        self.last_prompt_tokens = None
        self.last_cached_tokens = 0
        self.seen_tools = []

    def query(self, messages, tools=None):
        self.seen_tools.append(tools)
        reply, prompt_tokens = next(self._steps)
        if reply == OVERFLOW:
            raise ContextWindowExceededError(
                "prompt is too long", model="fake", llm_provider="fake"
            )
        self.last_prompt_tokens = prompt_tokens
        return reply

    def usage(self):
        return {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}


def test_json_happy_path_streams_events_and_result(tmp_path, monkeypatch, task_file):
    _patch_model(
        monkeypatch,
        assistant("Writing file.", tool_calls=[tool_call(1, command="printf hi > out.txt")]),
        final_answer("Wrote out.txt."),
    )

    result = CliRunner().invoke(
        _cli_app(),
        [
            "--task-file",
            task_file("write file"),
            "--json",
            "--cwd",
            str(tmp_path),
            "--session-dir",
            str(tmp_path / "sessions"),
            "--steps",
            "3",
        ],
    )

    assert result.exit_code == 0
    events = _json_lines(result.stdout)
    assert [event["type"] for event in events] == [
        "session",
        "context",
        "thought",
        "command",
        "observation",
        "context",
        "thought",
        "result",
    ]
    assert set(events[0]) == {"type", "session_id"}
    assert events[1]["step"] == 1
    assert events[5]["step"] == 2
    assert events[3]["id"] == events[4]["id"]
    assert events[3]["command"] == "printf hi > out.txt"
    assert events[4]["returncode"] == 0
    assert events[-1]["completed"] is True
    assert events[-1]["final_output"] == "Wrote out.txt."
    assert events[-1]["usage"] == {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}
    assert (tmp_path / "out.txt").read_text() == "hi"


def test_json_step_limit_emits_error_and_nonzero_exit(tmp_path, monkeypatch, task_file):
    _patch_model(
        monkeypatch,
        assistant("Still working.", tool_calls=[tool_call(1, command="echo not_done")]),
    )

    result = CliRunner().invoke(
        _cli_app(),
        [
            "--task-file",
            task_file("never complete"),
            "--json",
            "--cwd",
            str(tmp_path),
            "--session-dir",
            str(tmp_path / "sessions"),
            "--steps",
            "1",
        ],
    )

    assert result.exit_code != 0
    events = _json_lines(result.stdout)
    assert events[0]["type"] == "session"
    assert events[-1]["type"] == "error"
    assert "Step limit (1) exceeded" in events[-1]["message"]


def test_json_model_exception_emits_error_and_nonzero_exit(tmp_path, monkeypatch, task_file):
    def query(self, messages, tools=None):
        raise ValueError("model exploded")

    monkeypatch.setattr(Model, "query", query)
    monkeypatch.setattr(Model, "usage", lambda self: {})

    result = CliRunner().invoke(
        _cli_app(),
        [
            "--task-file",
            task_file("fail"),
            "--json",
            "--cwd",
            str(tmp_path),
            "--session-dir",
            str(tmp_path / "sessions"),
            "--steps",
            "3",
        ],
    )

    assert result.exit_code != 0
    events = _json_lines(result.stdout)
    assert events[0]["type"] == "session"
    assert events[1] == {"type": "error", "message": "model exploded"}
    assert "Traceback" not in result.stdout


def test_agent_emit_collects_per_step_events(tmp_path):
    events = []
    agent = Agent(
        model=ScriptedModel(
            assistant(""),
            assistant("Writing file.",
                      tool_calls=[tool_call(1, command="printf hi > out.txt")]),
            final_answer("Wrote out.txt."),
        ),
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                              kill_after_seconds=10, log_dir=str(tmp_path)),
        templates=load_config()["templates"],
        step_limit=3,
        emit=events.append,
    )

    result = agent.run("write out.txt")

    assert result["completed"] is True
    assert [event["type"] for event in events] == [
        "context",
        "context",
        "thought",
        "command",
        "observation",
        "context",
        "thought",
    ]
    assert events[2] == {"type": "thought", "step": 2, "text": "Writing file."}
    assert events[3]["step"] == events[4]["step"] == 2
    assert events[3]["id"] == events[4]["id"]
    assert events[4]["returncode"] == 0
    assert (tmp_path / "out.txt").read_text() == "hi"


def test_agent_without_emit_keeps_return_and_step_limit_behavior(tmp_path):
    success = Agent(
        model=ScriptedModel(final_answer("Nothing to do.")),
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                              kill_after_seconds=10, log_dir=str(tmp_path)),
        templates=load_config()["templates"],
        step_limit=1,
        emit=None,
    ).run("finish")

    assert success["completed"] is True
    assert success["n_steps"] == 1
    assert success["usage"] == {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}

    agent = Agent(
        model=ScriptedModel(
            assistant("No completion.",
                      tool_calls=[tool_call(1, command="echo not_done")]),
        ),
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                              kill_after_seconds=10, log_dir=str(tmp_path)),
        templates=load_config()["templates"],
        step_limit=1,
        emit=None,
    )
    with pytest.raises(RuntimeError, match="Step limit \\(1\\) exceeded"):
        agent.run("do not finish")

    # With per-call usage reported (the context event's source values) and no
    # emit, nothing is emitted, nothing raises, and the run behaves the same.
    with_usage = Agent(
        model=UsageModel(
            (assistant("Working.", tool_calls=[tool_call(1, command="true")]), 4321),
            (final_answer("done"), 8765),
        ),
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                              kill_after_seconds=10, log_dir=str(tmp_path)),
        templates=load_config()["templates"],
        step_limit=3,
        emit=None,
    )
    result = with_usage.run("finish with usage")

    assert result["completed"] is True
    assert result["n_steps"] == 2
    assert result["usage"] == {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}


def test_context_event_per_model_call_reports_that_calls_usage(tmp_path):
    events = []
    agent = Agent(
        model=UsageModel(
            (assistant("First.", tool_calls=[tool_call(1, command="echo one")]), 111),
            (assistant("Second.", tool_calls=[tool_call(2, command="echo two")]), 222),
            (final_answer("all done"), 333),
        ),
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                              kill_after_seconds=10, log_dir=str(tmp_path)),
        templates=load_config()["templates"],
        step_limit=5,
        emit=events.append,
    )

    result = agent.run("two commands then done")

    assert result["completed"] is True
    assert [event["type"] for event in events] == [
        "context", "thought", "command", "observation",
        "context", "thought", "command", "observation",
        "context", "thought",
    ]
    context_events = [event for event in events if event["type"] == "context"]
    assert len(context_events) == 3
    assert [(event["step"], event["prompt_tokens"]) for event in context_events] == [
        (1, 111), (2, 222), (3, 333),
    ]
    for event in context_events:
        assert set(event) == {"type", "step", "prompt_tokens", "cached_tokens",
                              "context_window", "compact_threshold", "model"}
        assert event["cached_tokens"] == 0
        assert event["model"] == "openai/fake-model"


def test_context_event_threshold_is_fraction_times_window_for_the_config_in_use(
    tmp_path,
):
    compact = {
        "context_window": 10000,
        "threshold_fraction": 0.5,
        "target_fraction": 0.3,
        "min_gain_tokens": 16384,
        "image_tokens": 1600,
        "ladder": ["mask", "reasoning", "command", "summarize"],
        "keep_tail_tokens": 300,
        "tail_budget_tokens": 24000,
        "command_head_chars": 200,
        "step_observation_budget_chars": 40000,
    }
    events = []
    agent = Agent(
        model=UsageModel((final_answer("done"), 100)),
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                              kill_after_seconds=10, log_dir=str(tmp_path)),
        templates=load_config()["templates"],
        step_limit=2,
        emit=events.append,
        compact=compact,
    )

    agent.run("finish")

    event = [event for event in events if event["type"] == "context"][0]
    assert event["context_window"] == 10000
    assert event["compact_threshold"] == int(0.5 * 10000) == 5000


def test_cli_context_window_flag_flows_into_the_context_event(
    tmp_path, monkeypatch, task_file
):
    def query(self, messages, tools=None):
        self.last_prompt_tokens = 118234
        return final_answer("done")

    monkeypatch.setattr(Model, "query", query)

    result = CliRunner().invoke(
        _cli_app(),
        [
            "--task-file",
            task_file("show context"),
            "--json",
            "--cwd",
            str(tmp_path),
            "--session-dir",
            str(tmp_path / "sessions"),
            "--context-window",
            "262144",
        ],
    )

    assert result.exit_code == 0, result.output
    events = _json_lines(result.stdout)
    context_events = [event for event in events if event["type"] == "context"]
    assert len(context_events) == 1
    event = context_events[0]
    assert event["step"] == 1
    assert event["prompt_tokens"] == 118234
    assert event["context_window"] == 262144
    assert event["compact_threshold"] == int(0.65 * 262144) == 170393
    assert event["model"] == load_config()["model"]["model_name"]


def test_context_event_carries_the_cached_tokens_the_endpoint_reported(tmp_path):
    events = []
    model = UsageModel((final_answer("done"), 60012))
    model.last_cached_tokens = 58880
    agent = Agent(
        model=model,
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                                kill_after_seconds=10, log_dir=str(tmp_path)),
        templates=load_config()["templates"],
        step_limit=2,
        emit=events.append,
    )

    agent.run("finish with a warm cache")

    event = [event for event in events if event["type"] == "context"][0]
    assert event["prompt_tokens"] == 60012
    assert event["cached_tokens"] == 58880


def test_context_event_on_the_overflow_retry_reports_the_retry_usage(tmp_path):
    s_summary = assistant("1. Progress: nothing yet. 5. Remaining: everything.")
    s_done = final_answer("finished after retry")
    events = []
    agent = Agent(
        model=UsageModel(
            (OVERFLOW, None),   # original call: over-window
            (s_summary, 700),   # compaction summary call
            (s_done, 900),      # retried conversation call
        ),
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                              kill_after_seconds=10, log_dir=str(tmp_path)),
        templates=load_config()["templates"],
        step_limit=3,
        emit=events.append,
    )

    result = agent.run("overflow at the very first call")

    assert result["completed"] is True
    assert result["final_output"] == "finished after retry"
    types = [event["type"] for event in events]
    assert types.count("context") == 1
    context_index = types.index("context")
    event = events[context_index]
    assert event["step"] == 1
    assert event["prompt_tokens"] == 900
    # The overflow compaction's compact event precedes the step's context event.
    assert "compact" in types[:context_index]


def test_context_event_without_usage_reports_null_prompt_tokens(tmp_path):
    events = []
    compact = load_config()["compact"]
    agent = Agent(
        model=UsageModel((final_answer("done"), None)),
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                              kill_after_seconds=10, log_dir=str(tmp_path)),
        templates=load_config()["templates"],
        step_limit=2,
        emit=events.append,
    )

    result = agent.run("finish without usage")

    assert result["completed"] is True
    context_events = [event for event in events if event["type"] == "context"]
    assert len(context_events) == 1
    event = context_events[0]
    assert event["prompt_tokens"] is None
    assert event["step"] == 1
    assert event["context_window"] == compact["context_window"]
    assert event["compact_threshold"] == int(
        compact["threshold_fraction"] * compact["context_window"]
    )
    assert event["model"] == "openai/fake-model"


def test_transient_503_retries_inside_one_call_and_keeps_the_json_stream_clean(
    tmp_path, monkeypatch, task_file
):
    """A 503 recovered inside Model.query is invisible to the event stream: the
    delivered reply accounts the usage, and the retry chatter - status code and
    attempt number only - stays on stderr, never in the stdout JSON events."""
    sentinel = load_config()["agent"]["completion_sentinel"]
    delivered = FakeCompletionResponse(f"Done.\n{sentinel}", prompt_tokens=11,
                                       completion_tokens=5)
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) <= 2:
            raise service_unavailable()
        return delivered

    monkeypatch.setattr(litellm, "completion", fake_completion)
    waits = []
    monkeypatch.setattr(model, "_sleep", waits.append)

    result = CliRunner().invoke(
        _cli_app(),
        [
            "--task-file", task_file("retry me"),
            "--json", "--no-stream",
            "--cwd", str(tmp_path),
            "--session-dir", str(tmp_path / "sessions"),
            "--steps", "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 3
    assert len(waits) == 2
    # The retry lines never touched stdout: the event stream stays parseable and
    # shaped exactly like a clean run's.
    events = _json_lines(result.stdout)
    assert [event["type"] for event in events] == [
        "session", "context", "thought", "result",
    ]
    assert events[-1]["completed"] is True
    assert events[-1]["final_output"] == "Done."
    assert events[-1]["usage"] == {"n_calls": 1, "input_tokens": 11,
                                   "output_tokens": 5, "cached_tokens": 0}
    assert events[1]["prompt_tokens"] == 11
    # One stderr line per retry, naming the status code and the attempt number,
    # never the task or reply content.
    stderr_lines = result.stderr.splitlines()
    assert len(stderr_lines) == 2
    assert "HTTP 503" in stderr_lines[0] and "retry 1 of 2" in stderr_lines[0]
    assert "HTTP 503" in stderr_lines[1] and "retry 2 of 2" in stderr_lines[1]
    assert "retry me" not in result.stderr
    assert "Done." not in result.stderr
