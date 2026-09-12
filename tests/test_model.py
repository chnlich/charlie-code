"""Unit tests for client-side reasoning-leak stripping and the model-call gate.

No network is touched: litellm.completion is monkeypatched throughout.
"""

import time

import litellm
import pytest

from model import Model, strip_leaked_reasoning


def test_orphan_closing_tag_is_dropped():
    assert strip_leaked_reasoning("</think>\nreal answer") == "real answer"


def test_full_think_block_is_dropped():
    assert strip_leaked_reasoning("<think>reasoning</think>real") == "real"


def test_plain_content_without_tag_is_unchanged():
    assert strip_leaked_reasoning("just a normal answer") == "just a normal answer"


def test_bash_block_is_preserved_after_leaked_prefix():
    content = "<think>leaked thought</think>Let me do it.\n```bash\necho hi\n```"
    assert strip_leaked_reasoning(content) == "Let me do it.\n```bash\necho hi\n```"


def test_query_passes_timeout_and_disables_litellms_own_retries(monkeypatch):
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        raise TimeoutError("endpoint stalled")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    model = Model(model_name="m", api_base="http://x/v1", api_key="k",
                  timeout_seconds=7, stream=True)

    with pytest.raises(TimeoutError):
        model.query([{"role": "user", "content": "hi"}])

    assert seen["timeout"] == 7
    assert seen["num_retries"] == 0
    assert seen["stream"] is True
    assert seen["stream_options"] == {"include_usage": True}


def test_non_streaming_call_passes_timeout_and_omits_stream_options(monkeypatch):
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        raise TimeoutError("endpoint stalled")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    model = Model(model_name="m", api_base="http://x/v1", api_key="k",
                  timeout_seconds=7, stream=False)

    with pytest.raises(TimeoutError):
        model.query([{"role": "user", "content": "hi"}])

    assert seen["timeout"] == 7
    assert seen["num_retries"] == 0
    assert "stream" not in seen
    assert "stream_options" not in seen


def test_num_retries_zero_bounds_a_stalled_call_to_one_attempt(monkeypatch):
    """Without num_retries=0, litellm's own handler retries (default max_retries=2),
    tripling the cost of a stalled call. `fake_completion` mimics that internal
    retry loop, driven by the same `num_retries` kwarg litellm itself reads."""

    def fake_completion(**kwargs):
        attempts = kwargs.get("num_retries", 2) + 1
        for _ in range(attempts):
            time.sleep(kwargs["timeout"])
        raise TimeoutError("endpoint stalled")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    timeout_seconds = 0.1
    model = Model(model_name="m", api_base="http://x/v1", api_key="k",
                  timeout_seconds=timeout_seconds, stream=True)

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        model.query([{"role": "user", "content": "hi"}])
    elapsed = time.monotonic() - start

    # 1 attempt, not the 3 a hidden default max_retries=2 would cost.
    assert elapsed < timeout_seconds * 2


def test_query_sends_no_request_side_reasoning_fields(monkeypatch):
    """Endpoints separate reasoning server-side and strict OpenAI-compatible layers
    reject unknown request fields, so nothing like the removed `extra_body` /
    `separate_reasoning` knob may reach litellm.completion."""
    sent = {}

    def fake_completion(**kwargs):
        sent.update(kwargs)
        raise TimeoutError("endpoint stalled")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    model = Model(model_name="m", api_base="http://x/v1", api_key="k",
                  timeout_seconds=7, stream=True)

    with pytest.raises(TimeoutError):
        model.query([{"role": "user", "content": "hi"}])

    def walk(value):
        """Yield every dict key and scalar anywhere inside the call kwargs."""
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from walk(item)
        else:
            yield value

    keys_and_scalars = list(walk(sent))
    assert "extra_body" not in keys_and_scalars
    assert "separate_reasoning" not in keys_and_scalars
