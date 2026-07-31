"""Core linear-history agent loop, modeled on mini-swe-agent's ~100-line core.

The conversation is a flat message list. Each step the model answers with an
assistant message; when it carries tool calls we run them and feed one tool message
back per call. When it carries none, the envelope's `finish_reason` decides: only
`stop` with non-empty text ends the session. The message shape alone never proves
completion, because a truncated reply looks exactly like a finished one.
"""

import json
import os
import tempfile
from pathlib import Path

import yaml

from model import strip_leaked_reasoning

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"

# Stamped into every session state file. A session recorded under the old bash-block
# protocol carries no stamp, and resuming it would feed the model a history that
# tells it to answer with fenced commands, so those sessions are refused outright.
STATE_PROTOCOL = "tool-calls-v1"

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run one bash command in a fresh subprocess and return its combined "
            "stdout/stderr and exit code."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

# Structure markers of the model families we drive. Command output carrying any of
# these is withheld rather than fed back: the serving stack parses the model's
# generated text back into tool calls, so a marker that reaches the transcript can
# be echoed by the model and promoted from data into a real, executed call.
CONTROL_MARKERS = (
    "<tool_call>", "<arg_key>", "<arg_value>",
    "<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>",
    "<|user|>", "<|assistant|>", "<|system|>", "<|observation|>",
)


def load_config(path=DEFAULT_CONFIG_PATH):
    return yaml.safe_load(Path(path).read_text())


def render(template, **values):
    """Fill {{name}} placeholders. Double braces never collide with shell syntax."""
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def find_control_markers(text):
    """Control markers present in text, in declaration order."""
    return [marker for marker in CONTROL_MARKERS if marker in text]


def _marker_label(marker):
    """Name a marker without spelling it, so the note itself stays inert."""
    return marker.strip("<>|/")


def gate_output(output):
    """Return (output to feed back, note). Withhold anything carrying a marker."""
    hits = find_control_markers(output)
    if not hits:
        return output, None
    note = (f"output withheld: contains {len(hits)} model control marker(s) "
            f"[{', '.join(_marker_label(hit) for hit in hits)}]; {len(output)} bytes")
    replacement = (
        "The command ran and its exit code above is unchanged. Its output is "
        "withheld because it contains model control markers, which must not enter "
        "the conversation. Read the content through a transform instead, for "
        "example `base64 <file>` or `tr -d '<>|'`."
    )
    return replacement, note


def tool_call_command(tool_call):
    """Return (command, error); exactly one of the two is None."""
    function = tool_call.get("function") or {}
    name = function.get("name")
    if name != "bash":
        return None, f"Error: there is no tool named {name!r}. The only tool is `bash`."
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError as exc:
        return None, f"Error: tool arguments are not valid JSON ({exc})."
    command = arguments.get("command") if isinstance(arguments, dict) else None
    if not isinstance(command, str) or not command.strip():
        return None, "Error: the bash tool needs a non-empty string `command`."
    return command, None


def _load_state(state_path):
    """Messages from a session file, refusing anything not on this protocol."""
    state = json.loads(state_path.read_text())
    if not isinstance(state, dict) or state.get("protocol") != STATE_PROTOCOL:
        raise RuntimeError(
            f"{state_path} was recorded under an older protocol and cannot be "
            f"resumed; start a new session."
        )
    return state["messages"]


class Agent:
    def __init__(
        self,
        model,
        environment,
        templates,
        step_limit,
        skills_catalog="",
        emit=None,
        state_file=None,
        resume=False,
    ):
        self.model = model
        self.environment = environment
        self.templates = templates
        self.step_limit = step_limit
        self.skills_catalog = skills_catalog
        self.emit = emit
        self.state_file = state_file
        self.resume = resume
        self.messages = []

    def _initial_messages(self, task):
        if self.resume and self.state_file and Path(self.state_file).exists():
            messages = _load_state(Path(self.state_file))
            messages.append({
                "role": "user",
                "content": render(self.templates["instance"], task=task),
            })
            return messages

        return [
            {"role": "system", "content": render(
                self.templates["system"],
                cwd=self.environment.cwd,
                skills=self.skills_catalog,
            )},
            {"role": "user", "content": render(self.templates["instance"], task=task)},
        ]

    def _persist_messages(self):
        if self.state_file is None:
            return

        state_path = Path(self.state_file)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=str(state_path.parent),
            prefix=f".{state_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump({"protocol": STATE_PROTOCOL, "messages": self.messages}, tmp)
            tmp.write("\n")
            tmp_path = tmp.name

        os.replace(tmp_path, state_path)

    def _run_tool_calls(self, step_idx, thought, tool_calls):
        """Run every call in order, appending one tool message per call."""
        records = []
        for index, tool_call in enumerate(tool_calls, start=1):
            step_thought = thought if index == 1 else ""
            command, error = tool_call_command(tool_call)
            if error is not None:
                self.messages.append({"role": "tool",
                                      "tool_call_id": tool_call.get("id"),
                                      "content": error})
                records.append({"thought": step_thought, "command": None,
                                "observation": error, "note": "invalid tool call"})
                continue

            event_id = f"s-{step_idx}-{index}"
            if self.emit:
                self.emit({"type": "command", "step": step_idx,
                           "id": event_id, "command": command})

            result = self.environment.execute(command)
            output, note = gate_output(result["output"])
            if self.emit:
                self.emit({"type": "observation", "step": step_idx,
                           "id": event_id, "returncode": result["returncode"],
                           "output": output})

            observation = render(
                self.templates["observation"],
                returncode=result["returncode"],
                output=output or "<no output>",
            )
            if note:
                observation = f"[{note}]\n{observation}"
            self.messages.append({"role": "tool",
                                  "tool_call_id": tool_call.get("id"),
                                  "content": observation})
            records.append({"thought": step_thought, "command": command,
                            "observation": observation,
                            "returncode": result["returncode"], "note": note})
        return records

    def run(self, task):
        self.messages = self._initial_messages(task)
        steps = []
        try:
            for step_idx in range(1, self.step_limit + 1):
                message, finish_reason = self.model.query(self.messages, tools=[BASH_TOOL])
                self.messages.append(message)
                thought = strip_leaked_reasoning(message.get("content") or "").strip()
                if self.emit and thought:
                    self.emit({"type": "thought", "step": step_idx, "text": thought})

                if finish_reason == "length":
                    raise RuntimeError(
                        f"Step {step_idx}: generation was truncated "
                        "(finish_reason=length); raise the endpoint's output budget."
                    )

                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    steps.extend(self._run_tool_calls(step_idx, thought, tool_calls))
                    continue

                if finish_reason != "stop":
                    raise RuntimeError(
                        f"Step {step_idx}: unexpected finish_reason={finish_reason!r}."
                    )

                if thought:
                    return {"task": task, "steps": steps, "completed": True,
                            "n_steps": step_idx, "final_output": thought,
                            "usage": self.model.usage()}

                observation = self.templates["empty_response_reminder"]
                self.messages.append({"role": "user", "content": observation})
                steps.append({"thought": "", "command": None,
                              "observation": observation, "note": "empty response"})

            raise RuntimeError(
                f"Step limit ({self.step_limit}) exceeded without task completion."
            )
        finally:
            self._persist_messages()
