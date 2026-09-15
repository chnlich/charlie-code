"""Core linear-history agent loop, modeled on mini-swe-agent's ~100-line core.

The conversation is a flat message list. Each step the model answers with an
assistant message; when it carries tool calls we run them and feed one tool message
back per call. When it carries none, the session ends only if the model said so:
the reply's last line is the completion sentinel and an answer stands above it.
Shape alone never proves completion, because a truncated reply looks exactly like
a finished one; a sentinel cannot be truncated into existence.

To keep long sessions alive near the context window, the loop compacts history
with a ladder (see compact.py): a per-step entry budget bounds every step's
observations before they enter history; at a step boundary whose
measured-plus-estimated size reaches the threshold line and buys at least
min_gain_tokens above the target line, the ladder runs mask old observations,
drop old reasoning, elide old command bodies, and summarize in order, stopping
once the estimate is at or below the target line. Every rewritten text is saved
under the run's session log directory and its placeholder names that file, so
nothing is lost and no placeholder asks the model to re-run a command. An
over-window error from the endpoint goes straight to the summarize level and
retries the call once instead of killing the session.

Every message enters history through `Agent._append_message`, which rewrites the
session state file atomically on each append: a process killed at any moment
(SIGTERM from a manual stop, SIGKILL, OOM) loses at most the message in flight,
and resuming such a file first answers any tool call the kill left unanswered.
"""

import base64
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml
from litellm.exceptions import ContextWindowExceededError

from compact import (
    bound_observation,
    drop_old_reasoning,
    drop_oldest_middle_half,
    elide_old_commands,
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

# Image suffixes a task message may carry as image parts, with the MIME type
# each becomes in its data URL. main.py validates --image against these keys,
# so the suffix whitelist and the parts' MIME types share one source of truth.
IMAGE_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}

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


def _rewrite_fingerprints(messages):
    """References to the values a ladder level can rewrite, one entry per
    message: (content, reasoning_content, [tool_calls arguments]).

    The levels mutate messages in place, so comparing message dicts against a
    snapshot would compare a dict with itself. Comparing these value
    references instead: an untouched message holds the same objects, a
    rewritten one holds new strings or has lost the reasoning field. Holding
    the pre-pass references keeps them alive, so identity is decisive.
    """
    return [
        (
            message.get("content"),
            message.get("reasoning_content"),
            [
                call["function"].get("arguments")
                if isinstance(call.get("function"), dict) else None
                for call in (message.get("tool_calls") or [])
            ],
        )
        for message in messages
    ]


def render(template, **values):
    """Fill {{name}} placeholders. Double braces never collide with shell syntax."""
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def split_completion(text, sentinel):
    """Split a final answer from its completion line: (answer, completed).

    Completion is the model's own declaration: the last non-empty line is the
    sentinel and an answer stands above it. Matching tolerates surrounding
    whitespace and Markdown emphasis, which costs no safety, because a cut-off
    reply can only lose characters and never gain the line. A reply that is the
    bare sentinel carries no answer, so it reads as unfinished.
    """
    lines = text.rstrip().splitlines()
    if not lines:
        return text, False
    last = lines[-1].strip()
    while True:
        undecorated = last.strip("`*").strip()
        if undecorated == last:
            break
        last = undecorated
    if last != sentinel:
        return text, False
    answer = "\n".join(lines[:-1]).rstrip()
    if not answer:
        return text, False
    return answer, True


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
        skills_catalog="",
        emit=None,
        state_file=None,
        resume=False,
        compact=None,
        images=(),
        completion_sentinel=None,
        unfinished_reply_limit=None,
    ):
        self.model = model
        self.environment = environment
        self.templates = templates
        self.step_limit = step_limit
        agent_config = load_config()["agent"]
        self.completion_sentinel = (
            completion_sentinel if completion_sentinel is not None
            else agent_config["completion_sentinel"]
        )
        self.unfinished_reply_limit = (
            unfinished_reply_limit if unfinished_reply_limit is not None
            else agent_config["unfinished_reply_limit"]
        )
        # Consecutive tool-less replies that did not complete; any tool call resets it.
        self._unfinished_replies = 0
        self.skills_catalog = skills_catalog
        self.emit = emit
        self.state_file = state_file
        self.resume = resume
        self.compact = compact if compact is not None else load_config()["compact"]
        self.images = list(images)
        self.messages = []
        # len(self.messages) at the last successful model query: messages after it
        # are what the last measured prompt_tokens does not yet account for. None
        # right after a compaction, until the next query re-anchors the estimate.
        self._last_query_index = None
        # tool_call_id -> log file holding the command's full output, for the
        # observations this run appended. Masking points placeholders at these
        # files; observations inherited from an earlier run have no entry here.
        self._observation_logs = {}
        # Step number of the last compaction, for steps_since_last_compact.
        self._last_compact_step = 0

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
            completion_sentinel=self.completion_sentinel,
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
                # An invalid call never ran, so it has no command log; when its
                # text will not fit the budget whole, save it as the full text
                # the budget note names.
                log_path = None
                if len(error) > remaining:
                    log_path = self._save_full_text(f"s-{step_idx}-{index}.log", error)
                bounded, charged = bound_observation(error, remaining, log_path)
                remaining -= charged
                self._append_message({"role": "tool",
                                      "tool_call_id": tool_call.get("id"),
                                      "content": bounded})
                records.append({"thought": step_thought, "command": None,
                                "observation": bounded, "note": "invalid tool call"})
                continue

            event_id = f"s-{step_idx}-{index}"
            if self.emit:
                self.emit({"type": "command", "step": step_idx,
                           "id": event_id, "command": command})

            result = self.environment.execute(command, step_idx, index)
            log_path = result["log_path"]
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
            bounded, charged = bound_observation(observation, remaining, log_path)
            remaining -= charged
            if tool_call.get("id") is not None:
                self._observation_logs[tool_call["id"]] = log_path
            self._append_message({"role": "tool",
                                  "tool_call_id": tool_call.get("id"),
                                  "content": bounded})
            records.append({"thought": step_thought, "command": command,
                            "observation": bounded,
                            "returncode": result["returncode"], "note": note})
        return records

    def _save_full_text(self, filename, text):
        """Write one text into the run's session log directory; return its path.

        This is how the texts the compaction ladder lifts out of history - a
        dropped reasoning block, an elided command body, an observation the
        entry budget could not record - stay readable at the path a placeholder
        names. Nothing is ever deleted.
        """
        path = Path(self.environment.log_dir) / filename
        path.write_text(text, encoding="utf-8")
        return str(path)

    def _save_reasoning(self, text, span_ordinal):
        return self._save_full_text(f"s-{span_ordinal}.reasoning.txt", text)

    def _save_command(self, text, span_ordinal, call_index):
        return self._save_full_text(
            f"s-{span_ordinal}-{call_index}.command.txt", text
        )

    def _threshold_tokens(self):
        return int(self.compact["threshold_fraction"] * self.compact["context_window"])

    def _floor_parts(self):
        """(system, task, tail) token estimates, the floor's three parts.

        The floor is the incompressible part of the history: the system prompt
        and task the summarize level preserves, plus the verbatim tail the
        ladder's first three levels keep. Compaction can never push the
        estimate below it.
        """
        image_tokens = self.compact["image_tokens"]
        return (
            est_messages_tokens(self.messages[:1], image_tokens),
            est_messages_tokens(self.messages[1:2], image_tokens),
            self.compact["keep_tail_tokens"],
        )

    def _floor_tokens(self):
        return sum(self._floor_parts())

    def _target_tokens(self):
        """The line compaction stops on: the configured fraction of the window,
        but never below the floor - the floor cannot be compacted away, so a
        target under it would only make the ladder run to its end in vain."""
        return max(
            int(self.compact["target_fraction"] * self.compact["context_window"]),
            self._floor_tokens(),
        )

    def _preflight_floor(self):
        """Startup check on the floor, run once the initial history exists.

        A floor at or above the window kills the run at startup, naming each
        part's token count, because no compaction can ever reach the target
        line. A floor between the threshold line and the window gets one
        warning: compaction cannot free enough space to matter, so the session
        will run on the over-window path instead.
        """
        system_tokens, task_tokens, tail_tokens = self._floor_parts()
        floor = system_tokens + task_tokens + tail_tokens
        window = self.compact["context_window"]
        if floor >= window:
            raise RuntimeError(
                f"session floor of {floor} tokens reaches the context window of "
                f"{window} tokens: system={system_tokens}, task={task_tokens}, "
                f"tail={tail_tokens}; compaction can never get under the target "
                f"line, so the session cannot run."
            )
        if floor >= self._threshold_tokens():
            print(
                f"warning: session floor of {floor} tokens is above the "
                f"compaction threshold of {self._threshold_tokens()} tokens "
                f"(system={system_tokens}, task={task_tokens}, "
                f"tail={tail_tokens}); compaction cannot buy enough space, so "
                f"the history will grow until the endpoint reports over-window.",
                file=sys.stderr,
            )

    def _anchored_estimate(self):
        """prompt_tokens measured by the last call plus chars/4 of what was
        appended since. The measured value is the anchor; the chars/4 part only
        spans messages the measurement does not know about. None without anchor."""
        measured = self.model.last_prompt_tokens
        if measured is None or self._last_query_index is None:
            return None
        return measured + est_messages_tokens(
            self.messages[self._last_query_index:], self.compact["image_tokens"]
        )

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
        """Threshold trigger, checked at step boundaries.

        Two conditions must hold: the anchored estimate reaches the threshold
        line, and it exceeds the target line by at least min_gain_tokens. The
        second condition keeps a session whose floor is already high from
        paying a full re-prefill every step: compaction would buy less than the
        rewrite costs, so the history keeps growing and the over-window path
        handles the real ceiling.
        """
        estimate = self._anchored_estimate()
        if estimate is None or estimate < self._threshold_tokens():
            return
        if estimate < self._target_tokens() + self.compact["min_gain_tokens"]:
            return
        self._compact(step_idx, trigger="threshold", pre_tokens=estimate)

    def _compact(self, step_idx, trigger, pre_tokens=None):
        """One compaction pass over the configured ladder, then persist and emit.

        The first three levels (mask, reasoning, command) rewrite history in
        place outside the one verbatim tail all four levels share; the
        summarize level rebuilds the history as [system, task, summary, tail].
        The ladder stops as soon as the estimate is at or below the target
        line. An over-window endpoint error goes straight to the summarize
        level: the endpoint refused the call, so there is no trustworthy
        measured anchor for the first three levels' arithmetic at that moment.

        After each of the first three levels the estimate is anchored on the
        last measured prompt_tokens minus only the characters the pass freed
        (chars/4); a fresh whole-history estimate could hide an overfull
        prompt. After the summarize level the rebuilt history is estimated
        directly. Compaction rewrites self.messages in place; the state
        protocol is untouched.
        """
        image_tokens = self.compact["image_tokens"]
        ladder = list(self.compact["ladder"])
        target_tokens = self._target_tokens()
        floor_tokens = self._floor_tokens()
        if pre_tokens is None:
            estimate = self._anchored_estimate()
            pre_tokens = (estimate if estimate is not None
                          else est_messages_tokens(self.messages, image_tokens))
        anchor = self.model.last_prompt_tokens
        if trigger == "overflow" and "summarize" in ladder:
            ladder = ladder[ladder.index("summarize"):]
        tail_start, _ = verbatim_tail_span(
            self.messages, self.compact["keep_tail_tokens"]
        )
        fingerprints = _rewrite_fingerprints(self.messages)

        freed_chars = 0
        layers_run = []
        for level in ladder:
            if level == "summarize":
                self._summarize_layer(step_idx)
                layers_run.append(level)
                post_tokens_est = est_messages_tokens(self.messages, image_tokens)
                break
            if level == "mask":
                freed = mask_old_observations(
                    self.messages, tail_start, self._observation_logs
                )
            elif level == "reasoning":
                freed = drop_old_reasoning(
                    self.messages, tail_start, save=self._save_reasoning
                )
            elif level == "command":
                freed = elide_old_commands(
                    self.messages, tail_start,
                    self.compact["command_head_chars"], save=self._save_command,
                )
            else:
                raise RuntimeError(f"unknown compaction ladder level {level!r}")
            layers_run.append(level)
            freed_chars += freed
            post_tokens_est = max(
                0, int((anchor if anchor is not None else pre_tokens)
                       - freed_chars // 4)
            )
            if post_tokens_est <= target_tokens:
                break

        invalidated_from_index = None
        for index in range(min(len(fingerprints), len(self.messages))):
            if _rewrite_fingerprints(self.messages[index:index + 1])[0] != fingerprints[index]:
                invalidated_from_index = index
                break

        self._persist_messages()
        self._last_query_index = None
        steps_since_last_compact = step_idx - self._last_compact_step
        self._last_compact_step = step_idx
        event = {
            "type": "compact",
            "step": step_idx,
            "layers": layers_run,
            "layer": layers_run[-1],
            "trigger": trigger,
            "pre_tokens": int(pre_tokens),
            "post_tokens_est": post_tokens_est,
            "target_tokens": target_tokens,
            "floor_tokens": floor_tokens,
            "steps_since_last_compact": steps_since_last_compact,
            "invalidated_from_index": invalidated_from_index,
        }
        if self.emit:
            self.emit(event)
        else:
            print(
                f"[compact] step {step_idx} layers={','.join(layers_run)} "
                f"trigger={trigger} pre_tokens={event['pre_tokens']} "
                f"post_tokens_est={event['post_tokens_est']} "
                f"target_tokens={event['target_tokens']}",
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

    def _task_message(self, task):
        """The task user message: plain text, or text plus image parts.

        With images attached, content becomes a parts list: the rendered
        instance template first, then one image_url part per image in flag
        order, each a base64 data URL. Each file is read here, at message-build
        time; the CLI already validated existence, readability, suffix, and
        size before the Agent existed. Without images the message stays
        exactly the string-content form. Either way it persists verbatim:
        compaction retires it like any other message, resume replays it
        as-is.
        """
        text = render(self.templates["instance"], task=task)
        if not self.images:
            return {"role": "user", "content": text}
        parts = [{"type": "text", "text": text}]
        for path in self.images:
            raw = Path(path).read_bytes()
            mime = IMAGE_MIME[Path(path).suffix.lower().lstrip(".")]
            encoded = base64.b64encode(raw).decode("ascii")
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            })
        return {"role": "user", "content": parts}

    def run(self, task):
        self.messages = self._initial_messages()
        self._append_message(self._task_message(task))
        self._preflight_floor()
        steps = []
        try:
            for step_idx in range(1, self.step_limit + 1):
                self._maybe_compact(step_idx)
                message, finish_reason = self._query_step(step_idx)
                if self.emit:
                    self.emit({
                        "type": "context",
                        "step": step_idx,
                        "prompt_tokens": self.model.last_prompt_tokens,
                        "cached_tokens": self.model.last_cached_tokens,
                        "context_window": self.compact["context_window"],
                        "compact_threshold": self._threshold_tokens(),
                        "model": self.model.model_name,
                    })
                self._append_message(message)
                reply = strip_leaked_reasoning(message.get("content") or "").strip()
                thought, completed = split_completion(reply, self.completion_sentinel)
                if self.emit and thought:
                    self.emit({"type": "thought", "step": step_idx, "text": thought})

                if finish_reason == "length":
                    raise RuntimeError(
                        f"Step {step_idx}: generation was truncated "
                        "(finish_reason=length); raise the endpoint's output budget."
                    )

                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    self._unfinished_replies = 0
                    steps.extend(self._run_tool_calls(step_idx, thought, tool_calls))
                    continue

                if finish_reason != "stop":
                    raise RuntimeError(
                        f"Step {step_idx}: unexpected finish_reason={finish_reason!r}."
                    )

                if completed:
                    return {"task": task, "steps": steps, "completed": True,
                            "n_steps": step_idx, "final_output": thought,
                            "usage": self.model.usage()}

                self._unfinished_replies += 1
                if self._unfinished_replies >= self.unfinished_reply_limit:
                    raise RuntimeError(
                        f"Step {step_idx}: {self.unfinished_reply_limit} consecutive replies "
                        "without a tool call or the completion line; giving up."
                    )
                observation = render(self.templates["unfinished_reply_reminder"],
                                     completion_sentinel=self.completion_sentinel)
                self._append_message({"role": "user", "content": observation})
                steps.append({"thought": thought, "command": None,
                              "observation": observation, "note": "unfinished reply"})

            raise RuntimeError(
                f"Step limit ({self.step_limit}) exceeded without task completion."
            )
        finally:
            self._persist_messages()
            self.environment.sweep()
