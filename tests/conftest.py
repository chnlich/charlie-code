"""Shared helpers for the tool-calling protocol tests."""

import json

import pytest

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


class ScriptedModel:
    """Stands in for Model: replays canned (message, finish_reason) pairs."""

    def __init__(self, *replies):
        self._replies = iter(replies)
        self.seen_tools = []

    def query(self, messages, tools=None):
        self.seen_tools.append(tools)
        return next(self._replies)

    def usage(self):
        return {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}


@pytest.fixture
def templates():
    from agent import load_config

    return load_config()["templates"]
