"""Core linear-history agent loop, modeled on mini-swe-agent's ~100-line core.

The conversation is a flat message list. Each step the model answers with an
assistant message; when it carries tool calls we run them and feed one tool message
back per call. When it carries none, the envelope's `finish_reason` decides: only
`stop` with non-empty text ends the session. The message shape alone never proves
completion, because a truncated reply looks exactly like a finished one.

To keep long sessions alive near the context window, the loop compacts history in
layers (see compact.py): a per-step entry budget bounds every step's observations
before they enter history; at a step boundary whose measured-plus-estimated size
reaches the threshold, old observations are masked (lossless: re-run to re-read),
and when that is not enough a structured summary plus a verbatim tail replaces the
history. An over-window error from the endpoint runs the same compaction and
retries the call once instead of killing the session.

Every message enters history through `Agent._append_message`, which rewrites the
session state file atomically on each append: a process killed at any moment
(SIGTERM from a manual stop, SIGKILL, OOM) loses at most the message in flight,
and resuming such a file first answers any tool call the kill left unanswered.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import yaml
from litellm.exceptions import ContextWindowExceededError

from compact import (
    bound_observation,
    drop_oldest_middle_half,
    est_messages_tokens,
    mask_old_observations,
    verbatim_tail_span,
)
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


INTERRUPTED_TOOL_RESULT = (
    "[interrupted: the session was stopped before this tool call returned]"
)


def repair_dangling_tool_calls(messages):
    """Answer every tool call that has no tool message, in place.

    Per-message persistence writes an assistant's tool calls before their
    results, so a session killed mid-step leaves calls without answers, and a
    history like that is invalid for the model API. Each unanswered call gets a
    placeholder tool message stating the interruption, placed after the step's
    existing tool messages; no result is ever invented. Returns the count added.
    """
    added = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") != "assistant":
            index += 1
            continue
        end = index + 1
        while end < len(messages) and messages[end].get("role") == "tool":
            end += 1
        answered = {messages[k].get("tool_call_id") for k in range(index + 1, end)}
        placeholders = [
            {"role": "tool", "tool_call_id": call.get("id"),
             "content": INTERRUPTED_TOOL_RESULT}
            for call in (message.get("tool_calls") or [])
            if call.get("id") not in answered
        ]
        messages[end:end] = placeholders
        added += len(placeholders)
        index = end + len(placeholders)
    return added


def read_agents_md(cwd):
    """The cwd's AGENTS.md text for a fresh session's system message, or "".

    The file is an environment convention scanned at startup, like CLAUDE.md for
    Claude Code, not something the run selected -- so a missing or blank file is
    simply no convention (silent), while an unreadable one must not kill a
    session launched for an unrelated task: one stderr warning, then continue
    without it. Resumed sessions never get here; their history already carries
    whatever the file said when they were fresh.
    """
    path = Path(cwd) / "AGENTS.md"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except (UnicodeDecodeError, OSError) as exc:
        print(f"warning: skipping unreadable {path}: {exc}", file=sys.stderr)
        return ""
    return text if text.strip() else ""


class Agent:
    def __init__(
        self,
        model,
        environment,
        templates,
        step_limit,
        wall_seconds=3600,
        skills_catalog="",
        emit=None,
        state_file=None,
        resume=False,
        compact=None,
    ):
        self.model = model
        self.environment = environment
        self.templates = templates
        self.step_limit = step_limit
        self.wall_seconds = wall_seconds
        self.skills_catalog = skills_catalog
        self.emit = emit
        self.state_file = state_file
        self.resume = resume
        self.compact = compact if compact is not None else load_config()["compact"]
        self.messages = []
        self._start_time = None
        # len(self.messages) at the last successful model query: messages after it
        # are what the last measured prompt_tokens does not yet account for. None
        # right after a compaction, until the next query re-anchors the estimate.
        self._last_query_index = None

    def _check_wall(self):
        """Raise once elapsed run time exceeds the wall-clock budget.

        Called at step top, right after `model.query` returns, and after each
        single tool call inside `_run_tool_calls` -- a reply may carry several
        calls, so the budget must also be enforced between them, not only at step
        boundaries.
        """
        elapsed = time.monotonic() - self._start_time
        if elapsed > self.wall_seconds:
            raise RuntimeError(
                f"Wall-clock budget ({self.wall_seconds}s) exceeded after "
                f"{elapsed:.1f}s."
            )

    def _initial_messages(self):
        """History before this turn's task message.

        A resumed session's file is loaded and repaired; a fresh session starts
        from its system message alone. The task message itself enters through
        `_append_message` in `run`, so it reaches the state file before the
        first model call. `--resume` names an explicit target, so a missing
        state file is an error, never a silent fresh start.
        """
        if self.resume:
            state_path = Path(self.state_file) if self.state_file else None
            if state_path is None or not state_path.exists():
                raise RuntimeError(
                    f"Cannot resume: session state file {state_path} does not exist."
                )
            messages = _load_state(state_path)
            repair_dangling_tool_calls(messages)
            return messages

        system = render(
            self.templates["system"],
            cwd=self.environment.cwd,
            skills=self.skills_catalog,
        )
        agents_md = read_agents_md(self.environment.cwd)
        if agents_md:
            system += "\n\n" + agents_md
        return [{"role": "system", "content": system}]

    def _append_message(self, message):
        """The one way a message enters history: append, then persist.

        Persisting on every append bounds what a kill at any moment can lose to
        the message being produced; the file on disk is otherwise always the
        complete history so far.
        """
        self.messages.append(message)
        self._persist_messages()

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
        """Run every call in order, appending one tool message per call.

        Every observation passes the control-marker gate on its complete output
        first, then the step's cumulative entry budget: bounding decides what
        enters history, never whether the command runs.
        """
        records = []
        remaining = self.compact["step_observation_budget_chars"]
        for index, tool_call in enumerate(tool_calls, start=1):
            step_thought = thought if index == 1 else ""
            command, error = tool_call_command(tool_call)
            if error is not None:
                bounded, charged = bound_observation(error, remaining)
                remaining -= charged
                self._append_message({"role": "tool",
                                      "tool_call_id": tool_call.get("id"),
                                      "content": bounded})
                records.append({"thought": step_thought, "command": None,
                                "observation": bounded, "note": "invalid tool call"})
                self._check_wall()
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
            bounded, charged = bound_observation(observation, remaining)
            remaining -= charged
            self._append_message({"role": "tool",
                                  "tool_call_id": tool_call.get("id"),
                                  "content": bounded})
            records.append({"thought": step_thought, "command": command,
                            "observation": bounded,
                            "returncode": result["returncode"], "note": note})
            self._check_wall()
        return records

    def _threshold_tokens(self):
        return int(self.compact["threshold_fraction"] * self.compact["context_window"])

    def _anchored_estimate(self):
        """prompt_tokens measured by the last call plus chars/4 of what was
        appended since. The measured value is the anchor; the chars/4 part only
        spans messages the measurement does not know about. None without anchor."""
        measured = self.model.last_prompt_tokens
        if measured is None or self._last_query_index is None:
            return None
        return measured + est_messages_tokens(self.messages[self._last_query_index:])

    def _query_step(self, step_idx):
        """One model call with the over-window fallback.

        An over-window error runs one compaction and the call is retried once; a
        second over-window failure raises. Any other error propagates unchanged.
        """
        self._last_query_index = len(self.messages)
        try:
            return self.model.query(self.messages, tools=[BASH_TOOL])
        except ContextWindowExceededError:
            self._compact(step_idx, trigger="overflow")
        self._last_query_index = len(self.messages)
        try:
            return self.model.query(self.messages, tools=[BASH_TOOL])
        except ContextWindowExceededError as exc:
            raise RuntimeError(
                f"Step {step_idx}: context window still exceeded after one "
                f"compaction and retry; giving up."
            ) from exc

    def _maybe_compact(self, step_idx):
        """Threshold trigger, checked at step boundaries."""
        estimate = self._anchored_estimate()
        if estimate is None or estimate < self._threshold_tokens():
            return
        self._compact(step_idx, trigger="threshold", pre_tokens=estimate)

    def _compact(self, step_idx, trigger, pre_tokens=None):
        """One compaction pass: mask layer, summarize layer when masking is not
        enough (or the endpoint already reported overflow), then persist and
        emit. Compaction rewrites self.messages in place; the state protocol is
        untouched."""
        before_mask_est = est_messages_tokens(self.messages)
        if pre_tokens is None:
            estimate = self._anchored_estimate()
            pre_tokens = (estimate if estimate is not None
                          else before_mask_est)

        mask_old_observations(self.messages, self.compact["mask_keep_steps"])
        # Keep measured token calibration: chars/4 estimates only the masking
        # delta, since a fresh whole-history estimate can hide an overfull prompt.
        post_tokens_est = max(
            0, pre_tokens + est_messages_tokens(self.messages) - before_mask_est
        )
        layer = "mask"
        if (trigger == "overflow"
                or post_tokens_est >= self._threshold_tokens()):
            self._summarize_layer(step_idx)
            layer = "summarize"
            post_tokens_est = est_messages_tokens(self.messages)

        self._persist_messages()
        self._last_query_index = None
        event = {
            "type": "compact",
            "step": step_idx,
            "layer": layer,
            "trigger": trigger,
            "pre_tokens": int(pre_tokens),
            "post_tokens_est": post_tokens_est,
        }
        if self.emit:
            self.emit(event)
        else:
            print(
                f"[compact] step {step_idx} layer={layer} trigger={trigger} "
                f"pre_tokens={event['pre_tokens']} "
                f"post_tokens_est={event['post_tokens_est']}",
                file=sys.stderr,
            )

    def _summarize_layer(self, step_idx):
        """Rebuild history as [system, task, summary, verbatim tail]."""
        summary = self._summary_text(step_idx)
        tail_start, tail_end = verbatim_tail_span(
            self.messages, self.compact["tail_budget_tokens"]
        )
        prefix = self.templates["compact_summary_prefix"].rstrip("\n") + "\n\n"
        self.messages = (
            self.messages[:2]
            + [{"role": "user", "content": prefix + summary}]
            + self.messages[tail_start:tail_end]
        )

    def _summary_text(self, step_idx):
        """One same-model, tool-less summary call over the masked full history.

        Summary output passes the control-marker gate: markers mean one retry,
        then refusal. An over-window summary call drops the oldest half of the
        maskable middle and retries once.
        """
        dropped = False
        marked = False
        while True:
            prompt = self.messages + [
                {"role": "user", "content": self.templates["compact_prompt"]}
            ]
            try:
                message, finish_reason = self.model.query(prompt, tools=None)
            except ContextWindowExceededError:
                if dropped:
                    raise RuntimeError(
                        f"Step {step_idx}: compaction summary still exceeds the "
                        f"context window after dropping the oldest half of the "
                        f"maskable middle."
                    )
                tail_start, _ = verbatim_tail_span(
                    self.messages, self.compact["tail_budget_tokens"]
                )
                drop_oldest_middle_half(self.messages, tail_start)
                dropped = True
                continue
            if finish_reason == "length":
                raise RuntimeError(
                    f"Step {step_idx}: compaction summary was truncated "
                    f"(finish_reason=length); raise the endpoint's output budget."
                )
            text = strip_leaked_reasoning(message.get("content") or "").strip()
            if find_control_markers(text):
                if marked:
                    raise RuntimeError(
                        f"Step {step_idx}: compaction summary contains model "
                        f"control markers after a retry; refusing to rebuild "
                        f"history on top of it."
                    )
                marked = True
                continue
            return text

    def run(self, task):
        self.messages = self._initial_messages()
        self._append_message({
            "role": "user",
            "content": render(self.templates["instance"], task=task),
        })
        self._start_time = time.monotonic()
        steps = []
        try:
            for step_idx in range(1, self.step_limit + 1):
                self._check_wall()
                self._maybe_compact(step_idx)
                message, finish_reason = self._query_step(step_idx)
                if self.emit:
                    self.emit({
                        "type": "context",
                        "step": step_idx,
                        "prompt_tokens": self.model.last_prompt_tokens,
                        "context_window": self.compact["context_window"],
                        "compact_threshold": self._threshold_tokens(),
                        "model": self.model.model_name,
                    })
                self._append_message(message)
                self._check_wall()
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
                self._append_message({"role": "user", "content": observation})
                steps.append({"thought": "", "command": None,
                              "observation": observation, "note": "empty response"})

            raise RuntimeError(
                f"Step limit ({self.step_limit}) exceeded without task completion."
            )
        finally:
            self._persist_messages()
            self.environment.sweep()
