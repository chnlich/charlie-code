"""The transcript: a plain-text, append-only record of every message the model
saw, one file per session, written as each message enters the context.

The context itself is rebuilt from scratch when it reaches the reset threshold
(see agent.py), so the transcript is where the model finds what it did before a
reset: user messages, its own replies and commands, and one stub line per
command result naming the log file that holds the full output. Command outputs
never enter the transcript; they live only in those log files, so `rg` over the
transcript answers "what did I do" and `rg` over the log directory answers
"what was in the output". The system message and the reset pointer are not
recorded: the first is in the state file unchanged for the whole session, the
second is constant and a `reset` record marks where each reset happened.

Record grammar: every header line is preceded by one blank line; stub lines
have no header and no blank line, so they attach to the assistant record above.

    ## user · <UTC time>
    <message text verbatim>

    ## assistant · step <n> · v2 · <UTC time>
    ```text
    <reasoning, when the endpoint returned any>
    ```
    ```text
    <reply text>
    ```
    ```
    $ <command>
    ```
    -> exit <code> · <chars> chars · <lines> lines · <log file>
    -> background · pid <pid> · <log file>
    -> background <task id> exit <code> · <chars> chars · <lines> lines · <log file>

    ## reset · <UTC time> · <trigger> · pre <n> tokens · post <n> tokens

    ## expiry · <UTC time> · kept <n> steps without a matching transcript step

A v2 assistant record fences each reasoning, reply and command body in backticks
longer than any backtick run that opens a line of that body, so no line inside a
fence is record structure. Reasoning and reply carry the info string `text`; a
command block has none and its first line is `$ `. An assistant header without
the `v2` token is a legacy record: read by line rules alone, where a `$ ` line
opens a command and any `## ` line ends the record.
"""

import base64
import datetime
import json
import re
from pathlib import Path

from compact import line_count

ASSISTANT_HEADER = "## assistant · step "

# An opening fence of a v2 block: backticks, then `text` for reasoning and reply.
FENCE_OPEN = re.compile(r"(`{3,})(text)?")

PREAMBLE = (
    "# charlie-code transcript · session {session_id}\n"
    "# One record per message the model saw, appended as it entered the context; the system message\n"
    '# is not recorded. "$ " opens a command; "-> " is its result: exit code, output size, and the\n'
    "# log file holding the full output. Command outputs live only in those log files.\n"
)


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _body(text):
    """A record body: the text verbatim, closed by exactly one newline."""
    return text.rstrip("\n") + "\n"


def _fenced(body, info=""):
    """`body` in a backtick fence of at least three backticks, longer than any
    backtick run that opens a line of `body`: no line of the body can close it."""
    longest = max((len(run) for run in re.findall(r"^`+", body, re.MULTILINE)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{info}\n{body}\n{ticks}\n"


def _image_bytes(url):
    """Decoded size of a base64 data URL's payload; 0 when the URL is not one."""
    if not isinstance(url, str) or "," not in url:
        return 0
    try:
        return len(base64.b64decode(url.split(",", 1)[1]))
    except (ValueError, TypeError):
        return 0


def _image_mime(url):
    if not isinstance(url, str) or not url.startswith("data:"):
        return "unknown"
    return url[5:].split(";", 1)[0] or "unknown"


class Transcript:
    """Append-only writer for one session's transcript file.

    The file is created with the preamble on the first append, so a session
    recorded under earlier code starts its transcript at the first message the
    new code appends. Each append opens, writes, and closes the file, so a kill
    at any moment loses at most the record being written.
    """

    def __init__(self, path, session_id):
        self.path = Path(path)
        self.session_id = session_id

    def append(self, record):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(PREAMBLE.format(session_id=self.session_id), encoding="utf-8")
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(record)

    @staticmethod
    def user_record(message):
        """A user message: the task (in its template wrapping) or a reminder."""
        content = message.get("content")
        if isinstance(content, list):
            lines = []
            image_index = 0
            for part in content:
                if part.get("type") == "image_url":
                    image_index += 1
                    url = (part.get("image_url") or {}).get("url")
                    lines.append(
                        f"[image {image_index}: {_image_mime(url)}, {_image_bytes(url):,} bytes]"
                    )
                else:
                    lines.append((part.get("text") or "").rstrip("\n"))
            text = "\n".join(lines)
        else:
            text = content or ""
        return f"\n## user · {_utc_now()}\n" + _body(text)

    @staticmethod
    def assistant_record(message, step):
        """An assistant message: reasoning (when present), the reply text, and
        one command block per tool call. An invalid call shows its raw arguments.
        Every body is fenced, so step_records reads it back verbatim."""
        parts = []
        reasoning = message.get("reasoning_content")
        if reasoning:
            parts.append(_fenced(reasoning.rstrip("\n"), "text"))
        content = message.get("content")
        if content:
            parts.append(_fenced(content.rstrip("\n"), "text"))
        for text in Transcript.command_texts(message):
            parts.append(_fenced("$ " + text.rstrip("\n")))
        return f"\n## assistant · step {step} · v2 · {_utc_now()}\n" + "".join(parts)

    @staticmethod
    def command_texts(message):
        """The exact command or raw arguments written for each tool call."""
        texts = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            raw = function.get("arguments") or ""
            text = None
            if function.get("name") == "bash":
                try:
                    payload = json.loads(raw or "{}")
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict) and isinstance(payload.get("command"), str):
                    text = payload["command"]
            texts.append(text if text is not None else raw)
        return texts

    def step_records(self):
        """Step numbers, full commands and output stubs recorded on disk.

        Lines split on "\\n" alone and keep every other byte, so a command
        carrying \\x0c or \\r reads back as written. An assistant header whose
        step is not an integer drops that record. Nothing here raises.
        """
        if not self.path.exists():
            return []

        records = []
        current = None

        def finish_record():
            if current is None:
                return
            stubs = current.pop("stubs")
            tool_stubs = []
            stub_index = 0
            for _ in current["commands"]:
                if stub_index == len(stubs):
                    tool_stubs.append(None)
                    continue
                group = [stubs[stub_index]]
                stub_index += 1
                while (
                    stub_index < len(stubs)
                    and stubs[stub_index].startswith("-> background ")
                    and " exit " in stubs[stub_index]
                ):
                    group.append(stubs[stub_index])
                    stub_index += 1
                tool_stubs.append("".join(group))
            current["tool_stubs"] = tool_stubs
            records.append(current)

        with self.path.open(encoding="utf-8", newline="") as handle:
            lines = handle.read().split("\n")
        if lines[-1] == "":
            lines.pop()

        fenced = False
        reading_command = False
        index = 0
        while index < len(lines):
            line = lines[index]
            index += 1
            if line.startswith(ASSISTANT_HEADER):
                finish_record()
                fields = line[len(ASSISTANT_HEADER):].split(" · ")
                current = None
                if fields[0].isdecimal():
                    current = {"step": int(fields[0]), "commands": [], "stubs": []}
                fenced = fields[1:2] == ["v2"]
                reading_command = False
            elif line.startswith("## "):
                finish_record()
                current = None
                reading_command = False
            elif current is None:
                continue
            elif fenced:
                opener = FENCE_OPEN.fullmatch(line)
                if opener is not None:
                    ticks = opener.group(1)
                    end = index
                    while end < len(lines) and lines[end] != ticks:
                        end += 1
                    body = "\n".join(lines[index:end])
                    index = end + 1
                    if opener.group(2) is None and body.startswith("$ "):
                        current["commands"].append(body[2:])
                elif line.startswith("-> ") and current["commands"]:
                    current["stubs"].append(line + "\n")
            elif line.startswith("$ "):
                current["commands"].append(line[2:])
                reading_command = True
            elif line.startswith("-> ") and current["commands"]:
                current["stubs"].append(line + "\n")
                reading_command = False
            elif reading_command:
                current["commands"][-1] += "\n" + line
        finish_record()
        return records

    @staticmethod
    def tool_stub(returncode, output, log_path):
        """One line per command that ran: exit code, size of the full output,
        and the log file that holds it."""
        return (
            f"-> exit {returncode} · {len(output):,} chars · {line_count(output):,} lines"
            f" · {log_path}\n"
        )

    @staticmethod
    def background_start_stub(pid, log_path):
        """One line per background call: the pid and the log that will hold the
        full output."""
        return f"-> background · pid {pid} · {log_path}\n"

    @staticmethod
    def background_exit_stub(task_id, returncode, output, log_path):
        """One line per exited background task, like tool_stub but naming the
        task id instead of an exit the model watched happen."""
        return (
            f"-> background {task_id} exit {returncode} · {len(output):,} chars"
            f" · {line_count(output):,} lines · {log_path}\n"
        )

    @staticmethod
    def invalid_call_stub(error):
        """A tool call that never ran: the first line of the error the model got."""
        first_line = (error or "").splitlines()[0] if error else ""
        return f"-> invalid call · {first_line}\n"

    @staticmethod
    def interrupted_stub():
        """The placeholder answer a resume gives a call the kill left unanswered."""
        return "-> interrupted\n"

    @staticmethod
    def reset_record(trigger, pre_tokens, post_tokens):
        """Where a reset happened: what triggered it and the estimates around it."""
        return (
            f"\n## reset · {_utc_now()} · {trigger} · pre {pre_tokens:,} tokens"
            f" · post {post_tokens:,} tokens\n"
        )

    @staticmethod
    def expiry_record(kept_steps):
        """Old content kept because its steps have no matching transcript record."""
        return (
            f"\n## expiry · {_utc_now()} · kept {kept_steps} steps"
            " without a matching transcript step\n"
        )
