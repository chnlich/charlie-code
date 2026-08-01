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
    model = Model(model_name="m", api_base="http://x/v1", api_key="k", model_timeout=7)

    with pytest.raises(TimeoutError):
        model.query([{"role": "user", "content": "hi"}])

    assert seen["timeout"] == 7
    assert seen["num_retries"] == 0


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
    model_timeout = 0.1
    model = Model(model_name="m", api_base="http://x/v1", api_key="k",
                  model_timeout=model_timeout)

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        model.query([{"role": "user", "content": "hi"}])
    elapsed = time.monotonic() - start

    # 1 attempt, not the 3 a hidden default max_retries=2 would cost.
    assert elapsed < model_timeout * 2
