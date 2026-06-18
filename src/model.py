"""litellm wrapper around the your-model SGLang endpoint with simple usage tracking.

your-model returns its chain-of-thought in a separate `reasoning_content` field; we
deliberately read ONLY the main message `content` and ignore the reasoning.
"""

import litellm


class Model:
    def __init__(self, model_name, api_base, api_key):
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.n_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def query(self, messages):
        """Send the conversation and return the assistant's message content."""
        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            api_base=self.api_base,
            api_key=self.api_key,
        )
        self.n_calls += 1
        usage = response.usage
        self.input_tokens += usage.prompt_tokens
        self.output_tokens += usage.completion_tokens
        # Use only the main content; reasoning_content is ignored on purpose.
        return response.choices[0].message.content or ""

    def usage(self):
        return {
            "n_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
