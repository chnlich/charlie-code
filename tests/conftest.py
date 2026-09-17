"""Shared helpers for the tool-calling protocol tests."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Prefer this checkout's own src/ and repo root over whatever an editable install's
# .pth file happens to point at (it may resolve to a different checkout of the same
# repo), so `python -m pytest` always exercises the code actually under test here.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

_UNSET = object()


def tool_call(index, name="bash", **arguments):
    """One OpenAI-shaped tool call. The id is opaque, as the real ones are."""
    return {
        "id": f"call-{index}",
        "index": index - 1,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def assistant(content="", tool_calls=None, finish_reason=_UNSET, **extra):
    """An (assistant message, finish_reason) pair shaped like an endpoint reply."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = list(tool_calls)
    message.update(extra)
    if finish_reason is _UNSET:
        finish_reason = "tool_calls" if tool_calls else "stop"
    return message, finish_reason


def final_answer(content, **extra):
    """A reply that completes the run: an answer closed by the completion line."""
    from agent import load_config

    sentinel = load_config()["agent"]["completion_sentinel"]
    return assistant(f"{content}\n{sentinel}", **extra)


def service_unavailable():
    """A litellm explicit-HTTP-503: the only exception the request layer retries.

    A fresh instance per call, so tests can assert the last one propagates.
    """
    import litellm

    return litellm.exceptions.ServiceUnavailableError(
        "UNAVAILABLE", llm_provider="openai", model="fake"
    )


class _FakeChoice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class FakeCompletionResponse:
    """A non-streamed endpoint reply, shaped as Model.query consumes it."""

    def __init__(self, content, prompt_tokens=11, completion_tokens=5):
        self.choices = [_FakeChoice({"role": "assistant", "content": content}, "stop")]
        self.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=None,
        )


class ScriptedModel:
    """Stands in for Model: replays canned (message, finish_reason) pairs."""

    def __init__(self, *replies):
        self._replies = iter(replies)
        self.model_name = "openai/fake-model"
        self.seen_tools = []
        # Zero tokens: the reset trigger never fires in these tests.
        self.last_prompt_tokens = 0
        self.last_cached_tokens = 0

    def query(self, messages, tools=None):
        self.seen_tools.append(tools)
        return next(self._replies)

    def usage(self):
        return {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}


@pytest.fixture
def templates():
    from agent import load_config

    return load_config()["templates"]


@pytest.fixture
def task_file(tmp_path):
    """Deliver a CLI task through a real temp file, never argv or stdin.

    --task-file accepts only a real readable file, so CLI tests write the
    task text with `task_file("...")` and pass the returned path. Rewrites
    reuse the same path within one test.
    """

    def write(text):
        path = tmp_path / "task.md"
        path.write_text(text, encoding="utf-8")
        return str(path)

    return write
