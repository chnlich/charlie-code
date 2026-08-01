"""Shared helpers for the tool-calling protocol tests."""

import json
import sys
from pathlib import Path

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
