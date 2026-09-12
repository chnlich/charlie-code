"""litellm wrapper around the configured OpenAI-compatible endpoint.

`query` hands back the assistant message exactly as the endpoint sent it, plus the
envelope's finish reason. The whole message is what goes back on the next turn:
Kimi K3 is trained in preserved-thinking-history mode and needs `reasoning_content`
and `tool_calls` returned as-is, not just `content`. The request itself carries no
reasoning knobs: endpoints separate reasoning server-side and strict OpenAI-compatible
layers reject unknown fields, which is why the old `extra_body={"separate_reasoning":
True}` was removed.

The call is streamed (`stream=True`) and the chunks are reassembled with
`litellm.stream_chunk_builder`, so the caller still receives one whole message.
Streaming is what makes "the model is still producing" observable: litellm's
`timeout` then bounds the silence between chunks (httpx's per-read timeout), not
the whole call, so a model that keeps producing is never cut off and only
`idle_seconds` of silence ends the call.

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


class Model:
    def __init__(self, model_name, api_base, api_key, idle_seconds):
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        # Seconds without a streamed chunk before the call fails. Passed to litellm
        # as `timeout`, which under streaming is the per-read (inter-chunk) bound.
        self.idle_seconds = idle_seconds
        self.n_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        # prompt_tokens of the most recent call; the compaction trigger's anchor.
        self.last_prompt_tokens = None

    def query(self, messages, tools=None):
        """Send the conversation and return (assistant message, finish_reason).

        The completion is streamed and read to the end here; `timeout` is the idle
        bound between chunks, so the call lasts as long as chunks keep arriving.
        Silence before the response headers raises `litellm.Timeout`; silence after
        them (including before the first chunk) raises `MidStreamFallbackError`;
        both become one RuntimeError naming the idle budget. Every other error,
        `ContextWindowExceededError` included, propagates unchanged for the agent's
        own handling.

        `num_retries=0` is explicit: litellm's OpenAI-compatible handler otherwise
        retries internally (default `max_retries=2`), which would silently triple
        the cost of a stalled call.
        """
        try:
            stream = litellm.completion(
                model=self.model_name,
                messages=messages,
                tools=tools,
                api_base=self.api_base,
                api_key=self.api_key,
                timeout=self.idle_seconds,
                num_retries=0,
                stream=True,
                stream_options={"include_usage": True},
            )
            chunks = list(stream)
        except (litellm.Timeout, litellm.exceptions.MidStreamFallbackError) as exc:
            raise RuntimeError(
                f"model produced no output for {self.idle_seconds}s"
            ) from exc
        response = litellm.stream_chunk_builder(chunks, messages=messages)
        self.n_calls += 1
        usage = response.usage
        self.input_tokens += usage.prompt_tokens
        self.output_tokens += usage.completion_tokens
        self.last_prompt_tokens = usage.prompt_tokens
        choice = response.choices[0]
        return as_message_dict(choice.message), choice.finish_reason

    def usage(self):
        return {
            "n_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
