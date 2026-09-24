"""Streaming end to end: Model.query against a loopback fake OpenAI-compatible endpoint.

Tests 1-4 and 7 run the real litellm.completion (never monkeypatched) against a fake
endpoint served by http.server on 127.0.0.1 at an ephemeral port in a daemon thread;
the fake takes a per-test plan of (delay_seconds, chunk_dict), writes
`data: <json>\n\n` lines then `data: [DONE]\n\n`, and touches no external network.
Small timeout_seconds keep the whole module well under 10 s.
"""

import http.server
import json
import socketserver
import threading
import time
from contextlib import contextmanager

import litellm
import pytest
import typer
from litellm.exceptions import (ContextWindowExceededError, InternalServerError,
                                ServiceUnavailableError)
from typer.testing import CliRunner

import main as cli_main
import model
from agent import Agent, load_config, tool_call_command
from conftest import ScriptedModel, assistant, tool_call
from environment import Environment
from model import Model, as_message_dict

litellm.suppress_debug_info = True


def content_chunk(text):
    return {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}


def finish_chunk(finish_reason):
    return {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}


def usage_chunk(prompt_tokens, completion_tokens):
    return {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens}}


def delta_chunk(delta):
    return {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


@contextmanager
def endpoint(plan, *, nonstream_reply=None, before_headers=0.0, status=200, raw_body=None,
             requests=None, first_errors=()):
    """Serve one OpenAI-compatible reply on 127.0.0.1 at an ephemeral port.

    A streamed request is answered with `plan` ((delay, chunk) pairs) as SSE lines
    followed by `data: [DONE]`; a non-streamed request with `nonstream_reply` as one
    JSON body. `before_headers` delays the response status line itself; `status` /
    `raw_body` replace the reply entirely (error-passthrough cases). `first_errors`
    serves (status, body) pairs for the first requests, before the normal reply.
    """
    served = [0]

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length))
            if requests is not None:
                requests.append(request)
            if before_headers:
                time.sleep(before_headers)
            served[0] += 1
            if served[0] <= len(first_errors):
                error_status, error_body = first_errors[served[0] - 1]
                self.send_response(error_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(error_body)))
                self.end_headers()
                self.wfile.write(error_body)
                return
            if status != 200:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw_body)))
                self.end_headers()
                self.wfile.write(raw_body)
                return
            if not request.get("stream"):
                body = json.dumps(nonstream_reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for delay, chunk in plan:
                    time.sleep(delay)
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except BrokenPipeError:
                # The client hit its idle budget and walked away mid-stream; the
                # failure it asserts on is the client side, not this write.
                pass

    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def test_steady_stream_is_never_cut_off_and_reports_usage():
    """Five chunks 0.2 s apart (1.0 s total, twice the idle budget) all arrive."""
    plan = [(0.2, content_chunk(text)) for text in "abcde"]
    plan += [(0.0, finish_chunk("stop")), (0.0, usage_chunk(7, 5))]
    with endpoint(plan) as base:
        model = Model(model_name="openai/fake", api_base=base, api_key="x", timeout_seconds=0.5, stream=True)
        message, finish_reason = model.query([{"role": "user", "content": "hi"}])
    assert message == {"role": "assistant", "content": "abcde"}
    assert finish_reason == "stop"
    assert model.last_prompt_tokens == 7
    assert model.input_tokens == 7
    assert model.output_tokens == 5
    assert model.n_calls == 1


def test_mid_stream_silence_ends_the_call_at_the_idle_budget():
    plan = [(0.05, content_chunk("a")), (1.5, content_chunk("b"))]
    with endpoint(plan) as base:
        model = Model(model_name="openai/fake", api_base=base, api_key="x", timeout_seconds=0.5, stream=True)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="no output for 0.5s"):
            model.query([{"role": "user", "content": "hi"}])
        assert time.monotonic() - started < 0.5 + 1.0


def test_silence_before_response_headers_ends_the_call_at_the_idle_budget():
    with endpoint([], before_headers=1.5) as base:
        model = Model(model_name="openai/fake", api_base=base, api_key="x", timeout_seconds=0.5, stream=True)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="no output for 0.5s"):
            model.query([{"role": "user", "content": "hi"}])
        assert time.monotonic() - started < 0.5 + 1.0


TOOL_ARGS = json.dumps({"command": "echo hi"})
TOOLCALL_REPLY = {
    "role": "assistant", "content": None, "reasoning_content": "think think",
    "tool_calls": [{"id": "call-1", "type": "function",
                    "function": {"name": "bash", "arguments": TOOL_ARGS}}],
}
TOOLCALL_USAGE = {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
TOOLCALL_NONSTREAM = {
    "id": "x", "object": "chat.completion", "created": 0, "model": "m",
    "choices": [{"index": 0, "message": TOOLCALL_REPLY, "finish_reason": "tool_calls"}],
    "usage": TOOLCALL_USAGE,
}


def toolcall_stream_plan():
    """One tool call as deltas: identity+name first, then the arguments in two
    fragments, then finish, then the usage chunk."""
    return [
        (0.0, delta_chunk({"role": "assistant", "reasoning_content": "think "})),
        (0.0, delta_chunk({"reasoning_content": "think"})),
        (0.0, delta_chunk({"tool_calls": [{"index": 0, "id": "call-1", "type": "function",
                                           "function": {"name": "bash", "arguments": ""}}]})),
        (0.0, delta_chunk({"tool_calls": [{"index": 0, "function": {"arguments": TOOL_ARGS[:9]}}]})),
        (0.0, delta_chunk({"tool_calls": [{"index": 0, "function": {"arguments": TOOL_ARGS[9:]}}]})),
        (0.0, finish_chunk("tool_calls")),
        (0.0, usage_chunk(TOOLCALL_USAGE["prompt_tokens"], TOOLCALL_USAGE["completion_tokens"])),
    ]


def test_tool_call_deltas_reassemble_to_the_non_stream_shape():
    with endpoint(toolcall_stream_plan(), nonstream_reply=TOOLCALL_NONSTREAM) as base:
        reference = litellm.completion(
            model="openai/fake", api_base=base, api_key="x",
            messages=[{"role": "user", "content": "hi"}], timeout=5, num_retries=0,
        )
        model = Model(model_name="openai/fake", api_base=base, api_key="x", timeout_seconds=5, stream=True)
        message, finish_reason = model.query([{"role": "user", "content": "hi"}])
    expected = as_message_dict(reference.choices[0].message)
    # provider_specific_fields is litellm-internal, not endpoint content: the
    # non-stream path attaches it, the reassembled stream does not.
    assert "provider_specific_fields" in expected
    del expected["provider_specific_fields"]
    assert message == expected
    assert finish_reason == "tool_calls"
    assert tool_call_command(message["tool_calls"][0]) == ("echo hi", False, None)


def test_reasoning_deltas_survive_reassembly():
    with endpoint(toolcall_stream_plan()) as base:
        model = Model(model_name="openai/fake", api_base=base, api_key="x", timeout_seconds=5, stream=True)
        message, _ = model.query([{"role": "user", "content": "hi"}])
    assert message["reasoning_content"] == "think think"
    assert not message.get("content")


def signature_delta(index, tool_id, arguments, signature):
    """A tool-call delta fragment carrying `extra_content` - the field outside
    the rebuild's fixed list where Gemini's OpenAI-compatible endpoint rides the
    thought signature."""
    return delta_chunk({"tool_calls": [{
        "index": index, "id": tool_id, "type": "function",
        "function": {"name": "bash", "arguments": arguments},
        "extra_content": {"google": {"thought_signature": signature}},
    }]})


def test_fields_outside_the_rebuild_list_survive_on_the_returned_tool_call():
    """The rebuild keeps a fixed field list; anything else the raw deltas carried
    must reach the caller verbatim, and `index` must not leak in with it."""
    plan = [
        (0.0, delta_chunk({"role": "assistant", "content": None})),
        (0.0, signature_delta(0, "call-1", "", "SIG")),
        (0.0, delta_chunk({"tool_calls": [
            {"index": 0, "function": {"arguments": TOOL_ARGS}}]})),
        (0.0, finish_chunk("tool_calls")),
        (0.0, usage_chunk(11, 5)),
    ]
    with endpoint(plan) as base:
        model = Model(model_name="openai/fake", api_base=base, api_key="x",
                      timeout_seconds=5, stream=True)
        message, finish_reason = model.query([{"role": "user", "content": "hi"}])
    tool_call = message["tool_calls"][0]
    assert tool_call["extra_content"] == {"google": {"thought_signature": "SIG"}}
    assert "index" not in tool_call
    assert tool_call_command(tool_call) == ("echo hi", False, None)
    assert finish_reason == "tool_calls"


def test_non_contiguous_delta_indexes_align_by_index_not_position():
    """Indexes 0 and 2: positional alignment would hunt for extras at position 1
    and lose the second call's field; index alignment keeps both."""
    plan = [
        (0.0, delta_chunk({"role": "assistant", "content": None})),
        (0.0, signature_delta(0, "call-a", "", "SIG-A")),
        (0.0, signature_delta(2, "call-b", TOOL_ARGS, "SIG-B")),
        (0.0, finish_chunk("tool_calls")),
        (0.0, usage_chunk(11, 5)),
    ]
    with endpoint(plan) as base:
        model = Model(model_name="openai/fake", api_base=base, api_key="x",
                      timeout_seconds=5, stream=True)
        message, _ = model.query([{"role": "user", "content": "hi"}])
    by_id = {tc["id"]: tc for tc in message["tool_calls"]}
    assert by_id["call-a"]["extra_content"] == {"google": {"thought_signature": "SIG-A"}}
    assert by_id["call-b"]["extra_content"] == {"google": {"thought_signature": "SIG-B"}}
    assert all("index" not in tc for tc in message["tool_calls"])


A = '{"background":true,"command":"sleep 20; echo done"}'
B = '{"command":"ls -la"}'


def test_parallel_calls_sharing_one_index_come_back_as_separate_calls():
    """Gemini's endpoint streams parallel calls all at index 0, one id per call;
    the rebuild groups fragments by index alone, so without the id-based split
    the two calls fuse into one whose arguments are two JSON objects end to end."""
    plan = [
        (0.0, delta_chunk({"role": "assistant", "content": None})),
        (0.0, signature_delta(0, "call_237700", A, "SIG-A")),
        (0.0, delta_chunk({"tool_calls": [
            {"index": 0, "id": "call_237702", "type": "function",
             "function": {"name": "bash", "arguments": B}}]})),
        (0.0, finish_chunk("tool_calls")),
        (0.0, usage_chunk(11, 5)),
    ]
    with endpoint(plan) as base:
        model = Model(model_name="openai/fake", api_base=base, api_key="x",
                      timeout_seconds=5, stream=True)
        message, _ = model.query([{"role": "user", "content": "hi"}])
    calls = message["tool_calls"]
    assert [tc["id"] for tc in calls] == ["call_237700", "call_237702"]
    assert [tc["function"]["arguments"] for tc in calls] == [A, B]
    assert calls[0]["extra_content"] == {"google": {"thought_signature": "SIG-A"}}
    assert "extra_content" not in calls[1]
    assert all("index" not in tc for tc in calls)
    assert [tool_call_command(tc) for tc in calls] == [
        ("sleep 20; echo done", True, None), ("ls -la", False, None)]


def test_openai_shaped_parallel_calls_are_unchanged_by_the_split():
    """OpenAI gives each parallel call its own index and rides its continuations
    on that index; the split must leave that shape's index sequence untouched."""
    plan = [
        (0.0, delta_chunk({"role": "assistant", "content": None})),
        (0.0, delta_chunk({"tool_calls": [
            {"index": 0, "id": "call-1", "type": "function",
             "function": {"name": "bash", "arguments": ""}}]})),
        (0.0, delta_chunk({"tool_calls": [
            {"index": 0, "function": {"arguments": A[:9]}}]})),
        (0.0, delta_chunk({"tool_calls": [
            {"index": 0, "function": {"arguments": A[9:]}}]})),
        (0.0, delta_chunk({"tool_calls": [
            {"index": 1, "id": "call-2", "type": "function",
             "function": {"name": "bash", "arguments": ""}}]})),
        (0.0, delta_chunk({"tool_calls": [
            {"index": 1, "function": {"arguments": B}}]})),
        (0.0, delta_chunk({"tool_calls": [
            {"index": 0, "function": {"arguments": ""}}]})),
        (0.0, finish_chunk("tool_calls")),
        (0.0, usage_chunk(11, 5)),
    ]
    with endpoint(plan) as base:
        model = Model(model_name="openai/fake", api_base=base, api_key="x",
                      timeout_seconds=5, stream=True)
        message, _ = model.query([{"role": "user", "content": "hi"}])
    calls = message["tool_calls"]
    assert [tc["id"] for tc in calls] == ["call-1", "call-2"]
    assert [tc["function"]["arguments"] for tc in calls] == [A, B]
    assert all("index" not in tc for tc in calls)


def test_separate_tool_call_fragments_assigns_one_index_per_id():
    """Unit view of the split: one index per distinct id, id-less continuations
    follow the id current at their original index, and the OpenAI shape's index
    sequence passes through unchanged."""
    from litellm.types.utils import ModelResponseStream

    def fragment(index, tool_id=None, name=None, arguments=""):
        call = {"index": index, "function": {"arguments": arguments}}
        if name is not None:
            call["function"]["name"] = name
        if tool_id is not None:
            call["id"] = tool_id
            call["type"] = "function"
        return call

    def rewritten_indexes(deltas):
        chunks = [ModelResponseStream(**delta_chunk({"tool_calls": [d]})) for d in deltas]
        assert model.separate_tool_call_fragments(chunks) is chunks
        return [tc.index for c in chunks for ch in c.choices
                for tc in (ch.delta.tool_calls or [])]

    # Gemini: two parallel calls, both streamed at index 0.
    assert rewritten_indexes([fragment(0, "call-a"), fragment(0, "call-b")]) == [0, 1]
    # Mixed: the id-less fragment joins the id now current at index 0.
    assert rewritten_indexes([
        fragment(0, "call-a"), fragment(0, "call-b"), fragment(0)]) == [0, 1, 1]
    # Name and arguments arrive before the id; the inherited placeholder keeps
    # its slot, so new indexes need not be contiguous.
    assert rewritten_indexes([
        fragment(0, name="bash", arguments='{"fir'),
        fragment(0, "call-nf", arguments='st":1}'),
        fragment(1, "call-2", name="bash", arguments=B),
    ]) == [0, 0, 2]
    # A fragment that never gets an id keeps its own new index.
    assert rewritten_indexes([fragment(0, "call-a"), fragment(1)]) == [0, 1]
    # OpenAI shape: each call its own index, id only on its first fragment.
    assert rewritten_indexes([
        fragment(0, "call-1"),
        fragment(0, arguments=A[:9]),
        fragment(0, arguments=A[9:]),
        fragment(1, "call-2"),
        fragment(1, arguments=B),
        fragment(0, arguments=""),
    ]) == [0, 0, 0, 1, 1, 0]


TOOLCALL_REPLY_WITH_SIGNATURE = {
    "role": "assistant", "content": None, "reasoning_content": "think think",
    "tool_calls": [{"id": "call-1", "type": "function",
                    "function": {"name": "bash", "arguments": TOOL_ARGS},
                    "extra_content": {"google": {"thought_signature": "SIG"}}}],
}
TOOLCALL_NONSTREAM_WITH_SIGNATURE = {
    "id": "x", "object": "chat.completion", "created": 0, "model": "m",
    "choices": [{"index": 0, "message": TOOLCALL_REPLY_WITH_SIGNATURE,
                 "finish_reason": "tool_calls"}],
    "usage": TOOLCALL_USAGE,
}


def test_non_streaming_path_returns_the_endpoint_message_and_never_streams():
    requests = []
    with endpoint([], nonstream_reply=TOOLCALL_NONSTREAM_WITH_SIGNATURE,
                  requests=requests) as base:
        model = Model(model_name="openai/fake", api_base=base, api_key="x",
                      timeout_seconds=5, stream=False)
        message, finish_reason = model.query([{"role": "user", "content": "hi"}])
    # No rebuild runs on this path: the endpoint's message comes through intact.
    tool_call = message["tool_calls"][0]
    assert tool_call["extra_content"] == {"google": {"thought_signature": "SIG"}}
    assert tool_call["id"] == "call-1"
    assert tool_call["function"] == {"name": "bash", "arguments": TOOL_ARGS}
    assert message["reasoning_content"] == "think think"
    assert "index" not in tool_call
    assert finish_reason == "tool_calls"
    # One non-streamed request was sent.
    assert not requests[0].get("stream")
    # Both paths account usage the same way.
    assert model.last_prompt_tokens == 11
    assert model.input_tokens == 11
    assert model.output_tokens == 5
    assert model.n_calls == 1


class SleepingModel:
    """Scripted stand-in that sleeps per query, so elapsed run time is observable."""

    def __init__(self, replies, seconds):
        self._replies = iter(replies)
        self._seconds = seconds
        self.last_prompt_tokens = 0

    def query(self, messages, tools=None):
        time.sleep(self._seconds)
        return next(self._replies)

    def usage(self):
        return {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}


def test_removed_agent_time_budget_is_rejected(tmp_path, templates):
    removed_kwarg = "wall" "_seconds"
    with pytest.raises(TypeError):
        Agent(
            model=ScriptedModel(),
            environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                              kill_after_seconds=10, log_dir=str(tmp_path)),
            templates=templates,
            step_limit=5,
            **{removed_kwarg: 0},
        )
    assert not hasattr(Agent, "_check" "_wall")


def test_a_slow_but_productive_run_finishes(tmp_path, templates):
    replies = [
        assistant(tool_calls=[tool_call(1, command="echo one")]),
        assistant(tool_calls=[tool_call(2, command="echo two")]),
        assistant("done"),
    ]
    agent = Agent(
        model=SleepingModel(replies, 0.2),
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                              kill_after_seconds=10, log_dir=str(tmp_path)),
        templates=templates,
        step_limit=5,
    )
    started = time.monotonic()
    result = agent.run("do three things")
    assert result["completed"] is True
    assert time.monotonic() - started >= 0.6


def test_removed_cli_option_exits_nonzero(task_file):
    removed_flag = "--wall" "-seconds"
    app = typer.Typer()
    app.command()(cli_main.run)
    result = CliRunner().invoke(
        app, ["--task-file", task_file("do it"), removed_flag, "0"]
    )
    assert result.exit_code != 0
    assert "No such option" in result.output


def test_nonpositive_or_noninteger_timeout_seconds_is_a_parameter_error(monkeypatch, task_file):
    def bomb(**kwargs):
        raise AssertionError("model called despite an invalid --timeout-seconds")

    monkeypatch.setattr(litellm, "completion", bomb)
    app = typer.Typer()
    app.command()(cli_main.run)
    for value in ("0", "-5", "abc"):
        result = CliRunner().invoke(
            app, ["--task-file", task_file("do it"), "--timeout-seconds", value]
        )
        assert result.exit_code == 2
        assert "--timeout-seconds" in result.output


def test_cli_stream_and_timeout_defaults_and_overrides_reach_the_model(tmp_path, task_file, monkeypatch):
    recorded = {}
    real_init = Model.__init__

    def init(self, **kwargs):
        recorded.update(kwargs)
        real_init(self, **kwargs)

    monkeypatch.setattr(Model, "__init__", init)
    monkeypatch.setattr(Model, "query",
                        lambda self, messages, tools=None: assistant("done"))
    monkeypatch.setattr(
        Model, "usage",
        lambda self: {"n_calls": 1, "input_tokens": 2, "output_tokens": 3},
    )
    app = typer.Typer()
    app.command()(cli_main.run)
    argv = ["--task-file", task_file("do it"), "--json", "--cwd", str(tmp_path),
            "--session-dir", str(tmp_path / "sessions"), "--steps", "1"]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0
    config = load_config()["model"]
    assert recorded["stream"] == config["stream"]
    assert recorded["timeout_seconds"] == config["timeout_seconds"]

    result = CliRunner().invoke(app, argv + ["--no-stream", "--timeout-seconds", "90"])
    assert result.exit_code == 0
    assert recorded["stream"] is False
    assert recorded["timeout_seconds"] == 90


OVER_WINDOW_BODY = json.dumps({
    "error": {"message": "This model's maximum context length is 8 tokens.",
              "type": "invalid_request_error", "code": "context_length_exceeded"},
}).encode()


def test_over_window_error_passes_through_as_context_window_exceeded():
    with endpoint([], status=400, raw_body=OVER_WINDOW_BODY) as base:
        model = Model(model_name="openai/fake", api_base=base, api_key="x", timeout_seconds=5, stream=True)
        with pytest.raises(ContextWindowExceededError):
            model.query([{"role": "user", "content": "hi"}])


# The 503 body an overloaded OpenAI-compatible endpoint sends; litellm maps the
# status line to ServiceUnavailableError regardless of the body's wording.
OVERLOADED_BODY = json.dumps(
    {"error": {"message": "UNAVAILABLE", "code": 503}}
).encode()


def _recorded_waits(monkeypatch):
    """Record the backoff waits instead of spending 90 real seconds sleeping."""
    waits = []
    monkeypatch.setattr(model, "_sleep", waits.append)
    return waits


def test_transient_503s_are_retried_and_the_stream_completes(monkeypatch):
    """503, 503, then a normal stream: three requests, one delivered response,
    usage accounted once, waits 30 s then 60 s plus 0-1 s jitter."""
    plan = [(0.0, content_chunk("hello")), (0.0, finish_chunk("stop")),
            (0.0, usage_chunk(9, 4))]
    requests = []
    with endpoint(plan, requests=requests,
                  first_errors=[(503, OVERLOADED_BODY)] * 2) as base:
        waits = _recorded_waits(monkeypatch)
        m = Model(model_name="openai/fake", api_base=base, api_key="x",
                  timeout_seconds=5, stream=True)
        message, finish_reason = m.query([{"role": "user", "content": "hi"}])

    assert message == {"role": "assistant", "content": "hello"}
    assert finish_reason == "stop"
    assert len(requests) == 3
    assert m.n_calls == 1
    assert m.input_tokens == 9
    assert m.output_tokens == 4
    assert 30 <= waits[0] <= 31
    assert 60 <= waits[1] <= 61


def test_persistent_503_propagates_after_exactly_three_requests(monkeypatch):
    requests = []
    with endpoint([], first_errors=[(503, OVERLOADED_BODY)] * 5,
                  requests=requests) as base:
        waits = _recorded_waits(monkeypatch)
        m = Model(model_name="openai/fake", api_base=base, api_key="x",
                  timeout_seconds=5, stream=True)
        with pytest.raises(ServiceUnavailableError):
            m.query([{"role": "user", "content": "hi"}])

    assert len(requests) == 3
    assert len(waits) == 2


def test_non_streaming_transient_503_is_retried(monkeypatch):
    """The non-streaming path shares the policy: 503 once, then the endpoint's
    message arrives whole and accounted."""
    requests = []
    with endpoint([], nonstream_reply=TOOLCALL_NONSTREAM, requests=requests,
                  first_errors=[(503, OVERLOADED_BODY)]) as base:
        waits = _recorded_waits(monkeypatch)
        m = Model(model_name="openai/fake", api_base=base, api_key="x",
                  timeout_seconds=5, stream=False)
        message, finish_reason = m.query([{"role": "user", "content": "hi"}])

    assert message["tool_calls"][0]["id"] == "call-1"
    assert finish_reason == "tool_calls"
    assert len(requests) == 2
    assert all(not request.get("stream") for request in requests)
    assert m.n_calls == 1
    assert 30 <= waits[0] <= 31


def test_mid_stream_silence_is_never_retried(monkeypatch):
    """A break after the first chunk keeps the single-attempt failure path:
    exactly one request, no backoff wait, the same RuntimeError."""
    plan = [(0.05, content_chunk("a")), (1.5, content_chunk("b"))]
    requests = []
    with endpoint(plan, requests=requests) as base:
        waits = _recorded_waits(monkeypatch)
        m = Model(model_name="openai/fake", api_base=base, api_key="x",
                  timeout_seconds=0.5, stream=True)
        with pytest.raises(RuntimeError, match="no output for 0.5s"):
            m.query([{"role": "user", "content": "hi"}])

    assert len(requests) == 1
    assert waits == []


def test_non_503_error_status_is_never_retried(monkeypatch):
    requests = []
    with endpoint([], status=500, raw_body=b'{"error": {"message": "boom"}}',
                  requests=requests) as base:
        waits = _recorded_waits(monkeypatch)
        m = Model(model_name="openai/fake", api_base=base, api_key="x",
                  timeout_seconds=5, stream=True)
        with pytest.raises(InternalServerError):
            m.query([{"role": "user", "content": "hi"}])

    assert len(requests) == 1
    assert waits == []
