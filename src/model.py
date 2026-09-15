"""litellm wrapper around the configured OpenAI-compatible endpoint.

`query` hands back the assistant message exactly as the endpoint sent it, plus the
envelope's finish reason. The whole message is what goes back on the next turn:
Kimi K3 is trained in preserved-thinking-history mode and needs `reasoning_content`
and `tool_calls` returned as-is, not just `content`. The request itself carries no
reasoning knobs: endpoints separate reasoning server-side and strict OpenAI-compatible
layers reject unknown fields, which is why the old `extra_body={"separate_reasoning":
True}` was removed.

The call runs in one of two modes, selected by `stream`. Streaming (the
default) reads the chunks to the end here and reassembles them with
`litellm.stream_chunk_builder`, so the caller still receives one whole message;
`timeout` then bounds the silence between chunks (httpx's per-read timeout), not
the whole call, so a model that keeps producing is never cut off. Non-streaming
takes the endpoint's message directly and `timeout` bounds the whole call. Either
way the number means "how long a silent call may stay silent", and both paths
account usage and propagate errors identically.

The streamed rebuild is lossy by design: it reconstructs tool calls from a fixed
field list and silently drops anything else - Gemini's OpenAI-compatible
endpoint, for one, carries the thought signature on
`tool_call.extra_content.google`, and losing it turns the next turn into a 400.
So after every rebuild `merge_stream_tool_call_fields` merges back every field
the raw deltas carried that the rebuild does not produce. No option or config
value gates this: handing back exactly what the endpoint sent is the invariant,
not a mode.

Some endpoints still leak reasoning into `content` as an orphan closing `</think>`
(SGLang issue #4711). `strip_leaked_reasoning` cleans that up for display text only;
the message stored in the conversation stays byte-for-byte what the endpoint sent.
"""

import litellm


def strip_leaked_reasoning(content):
    """Drop a leaked reasoning prefix ending in the LAST `</think>`, then lstrip."""
    marker = "</think>"
    idx = content.rfind(marker)
    if idx == -1:
        return content
    return content[idx + len(marker):].lstrip()


def as_message_dict(message):
    """Plain dict of an assistant message, keeping every field the endpoint sent.

    Unset fields are dropped so a `"tool_calls": null` never travels back out.
    """
    for attr in ("model_dump", "dict"):
        dump = getattr(message, attr, None)
        if callable(dump):
            return {key: value for key, value in dump().items() if value is not None}
    return {key: value for key, value in dict(message).items() if value is not None}


# Tool-call fields litellm's stream_chunk_builder already produces. `index` is in
# the set too - as the builder's fragment-grouping key, never as message content:
# merging it would inject streaming position into the returned message.
_REBUILT_TOOL_CALL_KEYS = frozenset(
    {"id", "type", "function", "custom", "provider_specific_fields", "index"}
)


def _delta_tool_call_extras(chunks):
    """Tool-call content in the raw stream deltas that the rebuild will drop.

    Returns two maps keyed by the builder's fragment index (a missing index
    defaults to 0, matching the builder's own grouping rule): `extras` holds, per
    index, the non-None fields outside the rebuild's fixed list, and
    `index_by_id` maps each fragment id to its index, because the rebuilt tool
    calls carry no index of their own.
    """
    extras = {}
    index_by_id = {}
    for chunk in chunks:
        for choice in chunk.model_dump().get("choices") or []:
            for tool_call in (choice.get("delta") or {}).get("tool_calls") or []:
                index = tool_call.get("index", 0)
                spare = {
                    key: value
                    for key, value in tool_call.items()
                    if key not in _REBUILT_TOOL_CALL_KEYS and value is not None
                }
                if spare:
                    extras.setdefault(index, {}).update(spare)
                if tool_call.get("id"):
                    index_by_id.setdefault(tool_call["id"], index)
    return extras, index_by_id


def prompt_cached_tokens(usage):
    """cached_tokens from the endpoint's prompt_tokens_details, 0 when absent.

    Endpoints disagree on the shape: some send the details object, some send
    nothing at all. Absence is the normal case for a cold cache, not an error,
    so it accounts as zero.
    """
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    if isinstance(details, dict):
        value = details.get("cached_tokens")
    else:
        value = getattr(details, "cached_tokens", None)
    return value or 0


def merge_stream_tool_call_fields(message, chunks):
    """Merge back onto each rebuilt tool call every field the raw deltas carried
    that the rebuild does not produce, aligned by fragment index.

    Write-if-absent, so the step stays idempotent if litellm starts carrying
    these fields itself. The alignment key is the fragment index the builder
    grouped by, never the list position of the rebuilt calls.
    """
    extras, index_by_id = _delta_tool_call_extras(chunks)
    for tool_call in message.get("tool_calls") or []:
        for field, value in extras.get(index_by_id[tool_call["id"]], {}).items():
            tool_call.setdefault(field, value)
    return message


class Model:
    def __init__(self, model_name, api_base, api_key, timeout_seconds, stream):
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        # Passed to litellm as `timeout`: the silence bound between streamed
        # chunks when streaming, the whole-call bound when not.
        self.timeout_seconds = timeout_seconds
        self.stream = stream
        self.n_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        # prompt_tokens of the most recent call; the compaction trigger's anchor.
        self.last_prompt_tokens = None
        # cached_tokens of the most recent call, for the per-step context event.
        self.last_cached_tokens = 0

    def query(self, messages, tools=None):
        """Send the conversation and return (assistant message, finish_reason).

        Streaming reads the stream to the end here and reassembles one whole
        message with `litellm.stream_chunk_builder`; `timeout` is the idle bound
        between chunks, so the call lasts as long as chunks keep arriving.
        Silence before the response headers raises `litellm.Timeout`; silence
        after them (including before the first chunk) raises
        `MidStreamFallbackError`; both become one RuntimeError naming the budget.
        Non-streaming takes the endpoint's message directly and `timeout` bounds
        the whole call. Every other error, `ContextWindowExceededError` included,
        propagates unchanged for the agent's own handling, and both paths account
        usage the same way.

        The streamed rebuild reconstructs tool calls from a fixed field list and
        drops anything else, so every field the raw deltas carried that the
        rebuild does not produce is merged back onto the returned tool calls -
        unconditionally, on every call: the message handed back and replayed must
        be exactly what the endpoint sent.

        `num_retries=0` is explicit: litellm's OpenAI-compatible handler otherwise
        retries internally (default `max_retries=2`), which would silently triple
        the cost of a stalled call.
        """
        try:
            if self.stream:
                stream = litellm.completion(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    timeout=self.timeout_seconds,
                    num_retries=0,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                chunks = list(stream)
                response = litellm.stream_chunk_builder(chunks, messages=messages)
            else:
                response = litellm.completion(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    timeout=self.timeout_seconds,
                    num_retries=0,
                )
        except (litellm.Timeout, litellm.exceptions.MidStreamFallbackError) as exc:
            raise RuntimeError(
                f"model produced no output for {self.timeout_seconds}s"
            ) from exc
        self.n_calls += 1
        usage = response.usage
        self.input_tokens += usage.prompt_tokens
        self.output_tokens += usage.completion_tokens
        self.cached_tokens += prompt_cached_tokens(usage)
        self.last_prompt_tokens = usage.prompt_tokens
        self.last_cached_tokens = prompt_cached_tokens(usage)
        choice = response.choices[0]
        message = as_message_dict(choice.message)
        if self.stream:
            merge_stream_tool_call_fields(message, chunks)
        return message, choice.finish_reason

    def usage(self):
        return {
            "n_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
        }
