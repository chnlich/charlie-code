"""Core linear-history agent loop, modeled on mini-swe-agent's ~100-line core.

A bash call can run its command in the background: the call returns at once
with the task id, pid and log path while the command keeps running under the
kill cap, and its exit record rides appended to a later tool result, at most
four records per result. Whatever still runs when the run ends is killed.

The conversation is a flat message list. Each step the model answers with an
assistant message; when it carries tool calls we run them and feed one tool message
back per call. A reply that carries no tool call ends the session and is
delivered to the user as the answer, as written. A reply with neither tool
calls nor text is an error.

History is append-only: a message that has been sent is never rewritten, so the
prompt prefix stays byte-identical from one call to the next and the endpoint's
prefix cache keeps hitting. Each command's output is bounded on its own before it
enters the context (see compact.py). Before every model call the agent estimates
the context from the last measured prompt_tokens plus chars/4 of what was appended
since; at or above threshold_fraction * context_window it rebuilds the context as
three messages, the system message, a one-line pointer to the session's transcript
and command log directory, and this turn's task, and the older material stays on
disk: the transcript (transcript.py) holds every message the model saw, the
per-command log files hold the full outputs, and the model reads them back with
rg. An over-window error from the endpoint runs the same reset and retries the
call once instead of killing the session.

Every message enters history through `Agent._append_message`, which appends the
transcript record and then rewrites the session state file atomically: a process
killed at any moment (SIGTERM from a manual stop, SIGKILL, OOM) loses at most the
message in flight, and resuming such a file first answers any tool call the kill
left unanswered.
"""

import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import yaml
from litellm.exceptions import ContextWindowExceededError

from compact import est_messages_tokens, truncate_observation
from model import strip_leaked_reasoning
from transcript import Transcript

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
            "stdout/stderr and exit code. With background=true the call returns at "
            "once with the task id, pid and log path while the command keeps "
            "running; its exit code and output arrive appended to a later tool "
            "result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "background": {
                    "type": "boolean",
                    "description": "Start the command and return at once; default false.",
                },
            },
            "required": ["command"],
        },
    },
}

# Exited background tasks delivered per tool result. Each record's output is
# bounded by command_observation_chars (5000) on its own, so four records add at
# most about 20k characters, about 5k tokens: a burst of finished tasks cannot push
# the context estimate over the reset threshold in one step. The rest wait for the
# next tool result, in exit order.
BACKGROUND_DELIVERY_LIMIT = 4

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
    """Return (command, background, error); exactly one of command and error is None.

    `background` is False when the argument is absent; when present it must be a
    JSON boolean, and anything else is an error that starts nothing.
    """
    function = tool_call.get("function") or {}
    name = function.get("name")
    if name != "bash":
        return (None, False,
                f"Error: there is no tool named {name!r}. The only tool is `bash`.")
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError as exc:
        return None, False, f"Error: tool arguments are not valid JSON ({exc})."
    command = arguments.get("command") if isinstance(arguments, dict) else None
    if not isinstance(command, str) or not command.strip():
        return None, False, "Error: the bash tool needs a non-empty string `command`."
    background = arguments.get("background", False)
    if not isinstance(background, bool):
        return None, False, "Error: the bash tool's `background` must be true or false."
    return command, background, None


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
    ):
        self.model = model
        self.environment = environment
        self.templates = templates
        self.step_limit = step_limit
        self.skills_catalog = skills_catalog
        self.emit = emit
        # The executor reports command progress into the same event stream.
        if emit is not None:
            self.environment.emit = emit
        self.state_file = state_file
        self.resume = resume
        self.compact = compact if compact is not None else load_config()["compact"]
        self.images = list(images)
        self.messages = []
        # This turn's task message: the third message of every rebuilt context.
        self._task = None
        # len(self.messages) at the last successful model query: messages after it
        # are what the last measured prompt_tokens does not yet account for. None
        # right after a reset, until the next query re-anchors the estimate.
        self._last_query_index = None
        # The session's disk directory <session id>.d (command logs per run, the
        # transcript at its top) derives from the state file; the pointer message
        # names both. Without a state file there is nothing to resume and no
        # transcript; the directory then follows the executor's log directory.
        if state_file is not None:
            state_path = Path(state_file)
            self.session_dir = state_path.with_suffix(".d")
            self.transcript = Transcript(self.session_dir / "transcript.md", state_path.stem)
        else:
            self.session_dir = Path(self.environment.log_dir).parent
            self.transcript = None

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
            added = repair_dangling_tool_calls(messages)
            for _ in range(added):
                self._record(Transcript.interrupted_stub())
            return messages

        system = render(
            self.templates["system"],
            cwd=self.environment.cwd,
            skills=self.skills_catalog,
            kill_after_minutes=f"{self.environment.kill_after_seconds / 60:g}",
        )
        agents_md = read_agents_md(self.environment.cwd)
        if agents_md:
            system += "\n\n" + agents_md
        return [{"role": "system", "content": system}]

    def _record(self, record):
        """Append one transcript record, when this session keeps a transcript."""
        if self.transcript is not None and record:
            self.transcript.append(record)

    def _append_message(self, message, record=None):
        """The one way a message enters history: append, record, then persist.

        The transcript record goes first, so the transcript is never behind the
        context a kill leaves on disk. Persisting on every append bounds what a
        kill at any moment can lose to the message being produced; the file on
        disk is otherwise always the complete history so far.
        """
        self.messages.append(message)
        self._record(record)
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

    def _pointer_message(self):
        """The one-line pointer a rebuilt context carries: the transcript path
        and the command log directory, rendered the same way for the whole
        session so the rebuilt prefix is stable across resets."""
        return {
            "role": "user",
            "content": render(
                self.templates["reset_pointer"],
                transcript=str(self.session_dir / "transcript.md"),
                session_dir=str(self.session_dir),
            ),
        }

    def _collect_deliveries(self):
        """Exited background tasks' records for the result being assembled.

        Pops at most BACKGROUND_DELIVERY_LIMIT tasks, in exit order, and renders
        one record per task: the full output is gated like any observation and
        then bounded on its own, so a burst of finished tasks cannot flood the
        context. Returns (text, stubs): `text` is the block appended to the tool
        result that triggered the collection, `stubs` the matching transcript
        lines. Records beyond the limit stay queued for the next collection.
        """
        tasks = self.environment.pop_finished(BACKGROUND_DELIVERY_LIMIT)
        if not tasks:
            return "", ""
        cap = self.compact["command_observation_chars"]
        records = []
        stubs = ""
        for task in tasks:
            raw = Path(task.log_path).read_text(errors="replace")
            gated, note = gate_output(raw)
            bounded = truncate_observation(gated, cap) if gated else "<no output>"
            if note:
                bounded = f"[{note}]\n{bounded}"
            if task.killed:
                outcome = (f"terminated at the {self.environment.cap_label()} cap: "
                           f"code {task.returncode}")
            else:
                outcome = f"exited: code {task.returncode}"
            command = task.command.replace("\n", " ")
            if len(command) > 80:
                command = command[:80] + "…"
            record = render(
                self.templates["background_exit"],
                task_id=task.id,
                outcome=outcome,
                seconds=round(task.duration),
                command=command,
                log=task.log_path,
                output=bounded,
            ).rstrip("\n")
            records.append(record)
            stubs += Transcript.background_exit_stub(task.id, task.returncode, raw,
                                                     task.log_path)
        return "\n\n".join(records), stubs

    def _running_phrase(self, tasks):
        """One phrase naming the background tasks still running, for a reminder."""
        cap = self.environment.cap_label()

        def desc(task):
            return (f"(pid {task.pid}, {round(time.monotonic() - task.started)} s of "
                    f"{cap}; log {task.log_path})")

        if len(tasks) == 1:
            return f"background task {tasks[0].id} is still running {desc(tasks[0])}"
        items = [f"{task.id} {desc(task)}" for task in tasks]
        return ("background tasks " + ", ".join(items[:-1]) + " and " + items[-1]
                + " are still running")

    def _run_tool_calls(self, step_idx, thought, tool_calls):
        """Run every call in order, appending one tool message per call.

        Every observation passes the control-marker gate on its complete output
        first, then the per-command cap: bounding decides what enters the
        context, never whether the command runs. The transcript gets one stub
        line per call naming the log file that holds the full output. A
        background call returns at once; after each call's own work is done, the
        records of background tasks that exited meanwhile ride appended to the
        same result, at most BACKGROUND_DELIVERY_LIMIT of them.
        """
        records = []
        cap = self.compact["command_observation_chars"]
        for index, tool_call in enumerate(tool_calls, start=1):
            step_thought = thought if index == 1 else ""
            command, background, error = tool_call_command(tool_call)
            if error is not None:
                bounded = truncate_observation(error, cap)
                text, stubs = self._collect_deliveries()
                if text:
                    bounded += "\n\n" + text
                self._append_message(
                    {"role": "tool", "tool_call_id": tool_call.get("id"), "content": bounded},
                    record=Transcript.invalid_call_stub(error) + stubs,
                )
                records.append({"thought": step_thought, "command": None,
                                "observation": bounded, "note": "invalid tool call"})
                continue

            event_id = f"s-{step_idx}-{index}"
            if self.emit:
                self.emit({"type": "command", "step": step_idx,
                           "id": event_id, "command": command})

            if background:
                task = self.environment.start_background(command, step_idx, index)
                body = render(self.templates["background_started"],
                              task_id=task.id, pid=task.pid,
                              log=task.log_path).rstrip("\n")
                text, stubs = self._collect_deliveries()
                content = body + ("\n\n" + text if text else "")
                if self.emit:
                    self.emit({"type": "observation", "step": step_idx,
                               "id": event_id, "returncode": None,
                               "background": True, "output": content})
                self._append_message(
                    {"role": "tool", "tool_call_id": tool_call.get("id"), "content": content},
                    record=Transcript.background_start_stub(task.pid, task.log_path) + stubs,
                )
                records.append({"thought": step_thought, "command": command,
                                "observation": content, "returncode": None,
                                "note": "background"})
                continue

            result = self.environment.execute(command, step_idx, index)
            output, note = gate_output(result["output"])
            text, stubs = self._collect_deliveries()
            observation = render(
                self.templates["observation"],
                returncode=result["returncode"],
                output=truncate_observation(output, cap) if output else "<no output>",
            )
            if note:
                observation = f"[{note}]\n{observation}"
            if text:
                observation += "\n\n" + text
            if self.emit:
                self.emit({"type": "observation", "step": step_idx,
                           "id": event_id, "returncode": result["returncode"],
                           "output": output + ("\n\n" + text if text else "")})
            self._append_message(
                {"role": "tool", "tool_call_id": tool_call.get("id"), "content": observation},
                record=Transcript.tool_stub(result["returncode"], result["output"],
                                            result["log_path"]) + stubs,
            )
            records.append({"thought": step_thought, "command": command,
                            "observation": observation,
                            "returncode": result["returncode"], "note": note})
        return records

    def _threshold_tokens(self):
        return int(self.compact["threshold_fraction"] * self.compact["context_window"])

    def _estimate(self):
        """Tokens the next call would send: the last measured prompt_tokens plus
        chars/4 of what was appended since, or chars/4 of the whole context when
        no measurement anchors it (the first call of a run, or after a reset)."""
        measured = self.model.last_prompt_tokens
        if measured is None or self._last_query_index is None:
            return est_messages_tokens(self.messages)
        return measured + est_messages_tokens(self.messages[self._last_query_index:])

    def _check_reset_floor(self):
        """Startup check: the rebuilt context alone must sit under the threshold.

        A session whose system message, pointer and task already reach the
        threshold would reset before every call and keep no memory between
        steps, so it refuses to start, naming the three parts.
        """
        parts = (
            est_messages_tokens(self.messages[:1]),
            est_messages_tokens([self._pointer_message()]),
            est_messages_tokens([self._task]),
        )
        floor = sum(parts)
        threshold = self._threshold_tokens()
        if floor >= threshold:
            raise RuntimeError(
                f"the rebuilt context alone is {floor} tokens (system={parts[0]}, "
                f"pointer={parts[1]}, task={parts[2]}), at or above the reset "
                f"threshold of {threshold} tokens: the session would reset at "
                f"every step, so it cannot run."
            )

    def _maybe_reset(self, step_idx):
        """Threshold trigger, checked before every model call (the step boundary)."""
        estimate = self._estimate()
        if estimate >= self._threshold_tokens():
            self._reset_context(step_idx, trigger="threshold", pre_tokens=estimate)

    def _reset_context(self, step_idx, trigger, pre_tokens):
        """Rebuild the context as [system, pointer, task], then persist and emit.

        The system message object and this turn's task message are reused as
        they are, so their bytes do not change; the pointer is rendered the same
        way every time. The old messages leave the context but not the disk:
        the transcript already holds them, and this reset gets its own record
        there. The anchor is cleared so the next query re-measures.
        """
        self.messages = [self.messages[0], self._pointer_message(), self._task]
        self._persist_messages()
        post_tokens_est = est_messages_tokens(self.messages)
        self._record(Transcript.reset_record(trigger, int(pre_tokens), post_tokens_est))
        self._last_query_index = None
        event = {
            "type": "compact",
            "step": step_idx,
            "trigger": trigger,
            "pre_tokens": int(pre_tokens),
            "post_tokens_est": post_tokens_est,
        }
        if self.emit:
            self.emit(event)
        else:
            print(
                f"[compact] step {step_idx} trigger={trigger} "
                f"pre_tokens={event['pre_tokens']} post_tokens_est={post_tokens_est}",
                file=sys.stderr,
            )

    def _query_step(self, step_idx):
        """One model call with the over-window fallback.

        An over-window error runs one reset and the call is retried once; a
        second over-window failure raises. Any other error propagates unchanged.
        """
        pre_tokens = self._estimate()
        self._last_query_index = len(self.messages)
        try:
            return self.model.query(self.messages, tools=[BASH_TOOL])
        except ContextWindowExceededError:
            self._reset_context(step_idx, trigger="overflow", pre_tokens=pre_tokens)
        self._last_query_index = len(self.messages)
        try:
            return self.model.query(self.messages, tools=[BASH_TOOL])
        except ContextWindowExceededError as exc:
            raise RuntimeError(
                f"Step {step_idx}: context window still exceeded after a reset "
                f"and retry; giving up."
            ) from exc

    def _task_message(self, task):
        """The task user message: plain text, or text plus image parts.

        With images attached, content becomes a parts list: the rendered
        instance template first, then one image_url part per image in flag
        order, each a base64 data URL. Each file is read here, at message-build
        time; the CLI already validated existence, readability, suffix, and
        size before the Agent existed. Without images the message stays
        exactly the string-content form. Either way it persists verbatim: a
        reset carries it over as it is, resume replays it as-is.
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
        self._task = self._task_message(task)
        self._append_message(self._task, record=Transcript.user_record(self._task))
        self._check_reset_floor()
        steps = []
        try:
            for step_idx in range(1, self.step_limit + 1):
                self._maybe_reset(step_idx)
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
                self._append_message(
                    message, record=Transcript.assistant_record(message, step_idx)
                )
                reply = strip_leaked_reasoning(message.get("content") or "").strip()
                if self.emit and reply:
                    self.emit({"type": "thought", "step": step_idx, "text": reply})

                if finish_reason == "length":
                    raise RuntimeError(
                        f"Step {step_idx}: generation was truncated "
                        "(finish_reason=length); raise the endpoint's output budget."
                    )

                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    steps.extend(self._run_tool_calls(step_idx, reply, tool_calls))
                    continue

                if finish_reason != "stop":
                    raise RuntimeError(
                        f"Step {step_idx}: unexpected finish_reason={finish_reason!r}."
                    )

                if not reply:
                    raise RuntimeError(f"Step {step_idx}: empty reply with no "
                                       "tool call; nothing to deliver.")

                running = self.environment.running_background()
                if running or self.environment.finished_pending():
                    text, stubs = self._collect_deliveries()
                    if running:
                        body = render(self.templates["background_running_reminder"],
                                      tasks=self._running_phrase(running))
                    else:
                        body = render(self.templates["background_delivered_reminder"])
                    content = text + "\n\n" + body.rstrip() if text else body.rstrip()
                    reminder = {"role": "user", "content": content}
                    self._append_message(
                        reminder, record=Transcript.user_record(reminder) + stubs
                    )
                    steps.append({"thought": reply, "command": None,
                                  "observation": content, "note": "background pending"})
                    continue

                return {"task": task, "steps": steps, "completed": True,
                        "n_steps": step_idx, "final_output": reply,
                        "usage": self.model.usage()}

            raise RuntimeError(
                f"Step limit ({self.step_limit}) exceeded without task completion."
            )
        finally:
            # Every exit path (completion, step limit, model error, SIGTERM and
            # KeyboardInterrupt through main.py) leaves no background process:
            # the main.py handlers kill what runs under them, and a task that
            # outlives the loop is killed here, before the state is persisted.
            if self.environment.running_background():
                self.environment.kill_running()
            self._persist_messages()
