"""litellm wrapper around the configured OpenAI-compatible endpoint.

`query` hands back the assistant message exactly as the endpoint sent it, plus the
envelope's finish reason. The whole message is what goes back on the next turn:
Kimi K3 is trained in preserved-thinking-history mode and needs `reasoning_content`
and `tool_calls` returned as-is, not just `content`.

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
    def __init__(self, model_name, api_base, api_key):
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.n_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def query(self, messages, tools=None):
        """Send the conversation and return (assistant message, finish_reason)."""
        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            tools=tools,
            api_base=self.api_base,
            api_key=self.api_key,
            extra_body={"separate_reasoning": True},
        )
        self.n_calls += 1
        usage = response.usage
        self.input_tokens += usage.prompt_tokens
        self.output_tokens += usage.completion_tokens
        choice = response.choices[0]
        return as_message_dict(choice.message), choice.finish_reason

    def usage(self):
        return {
            "n_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
