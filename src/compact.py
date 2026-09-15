"""Layered context compaction helpers.

Pure functions over the flat message list, used by agent.Agent:

- Entry budget: bound every step's observations before they enter history, so no
  single-step tool burst can blow the window on its own.
- Compaction ladder levels: mask old observations, drop old reasoning, elide old
  command bodies. Each rewrites history in place outside the retained tail and
  returns the characters it freed; the summarize level (rebuilt history) lives
  in agent.py because it calls the model. All four share one verbatim tail.
- Tail cutter: whole steps, newest first, up to a budget; an assistant message
  and its tool messages are inseparable.

Token estimates here use chars // 4. The agent anchors both the trigger and the
post-level estimates to the endpoint's measured usage.prompt_tokens, subtracting
only the characters a level freed. Rebuilt history gets a fresh estimate.
"""

import json
import re

#: Sentinel prefix marking a tool message whose observation was masked. Its presence
#: is what makes re-masking idempotent.
MASK_SENTINEL = "[observation masked by context compaction:"

#: Marker inside an elided command body. Its presence is what makes re-eliding
#: idempotent: an already-elided command is left exactly as it is.
COMMAND_ELISION_MARKER = "[... command body elided;"

#: Note appended to an elided command body. States the file holding the full
#: text, so reading a command back never means re-running it.
COMMAND_ELISION_NOTE = "\n[... command body elided; full text: {path} ...]"

#: Notes inserted by the per-step entry budget. Both state the original length
#: and the file holding the full text; nothing asks the model to re-run a
#: command, which side effects make false.
ELISION_NOTE = (
    "\n[... observation truncated by the per-step budget: original was "
    "{original} chars; head and tail kept. Full text: {path} ...]\n"
)
REPLACEMENT_NOTE = (
    "[observation not recorded: the per-step budget was exhausted; original was "
    "{original} chars. Full text: {path}.]"
)

_EXIT_CODE = re.compile(r"Exit code: (-?\d+)")
_ORIGINAL_CHARS = re.compile(r"original was (\d+) chars")


def est_message_chars(message, image_tokens=0):
    """Estimated character volume of one message (content + reasoning + calls).

    A parts-list content (a task message carrying images) is summed per part:
    text parts contribute their length, image parts contribute image_tokens * 4
    characters - the base64 length of an image says nothing about its token
    count, so each image counts as the configured constant. Callers estimating
    a history that can hold the task message pass the configured image_tokens;
    the default 0 covers step spans, which never carry parts.
    """
    content = message.get("content")
    if isinstance(content, list):
        total = 0
        for part in content:
            if part.get("type") == "image_url":
                total += image_tokens * 4
            else:
                total += len(part.get("text") or "")
    else:
        total = len(content or "")
    total += len(message.get("reasoning_content") or "")
    tool_calls = message.get("tool_calls")
    if tool_calls:
        total += len(json.dumps(tool_calls))
    return total


def est_messages_tokens(messages, image_tokens=0):
    """chars/4 estimate of a message list."""
    return sum(est_message_chars(message, image_tokens) for message in messages) // 4


def original_observation_chars(content):
    """Original length of a masked observation, stated in its placeholder."""
    match = _ORIGINAL_CHARS.search(content)
    if match is None:
        raise RuntimeError(f"masked observation without an original length: {content!r}")
    return int(match.group(1))


def split_steps(messages):
    """Index spans (start, end) of steps: each assistant message + its tool messages.

    Non-step messages (system, user) never appear in a span. Spans are in message
    order and never overlap.
    """
    spans = []
    index = 0
    while index < len(messages):
        if messages[index].get("role") == "assistant":
            end = index + 1
            while end < len(messages) and messages[end].get("role") == "tool":
                end += 1
            spans.append((index, end))
            index = end
        else:
            index += 1
    return spans


def span_ordinals(messages):
    """Message index -> 1-based step ordinal, for every step's assistant message.

    Every assistant message starts a span, so the map is total over assistant
    messages; the ordinals are the step numbers the session log directory names
    files by.
    """
    return {
        start: ordinal
        for ordinal, (start, _) in enumerate(split_steps(messages), start=1)
    }


def bound_observation(observation, remaining, log_path):
    """Bound one observation to the step's remaining character budget.

    Returns (text_for_history, chars_charged). Fits whole when it fits; over the
    remaining budget keeps head and tail halves with an elision note stating the
    original length and the log file holding the full text, charging exactly
    `remaining`; when nothing (or too little) is left, the observation is
    replaced by a one-line note that states the original length and the log
    file, charging 0. The replacement note lines are the one sanctioned overrun
    of the step budget: bounded by the number of tool calls in the step, one
    short line each.
    """
    if len(observation) <= remaining:
        return observation, len(observation)
    if log_path is None:
        raise RuntimeError(
            "bounding an observation past the step budget needs the log file "
            "holding its full text"
        )
    note = ELISION_NOTE.format(original=len(observation), path=log_path)
    if remaining > len(note):
        body = remaining - len(note)
        head = body // 2
        tail = body - head
        text = observation[:head] + note + observation[len(observation) - tail:]
        return text, len(text)
    return REPLACEMENT_NOTE.format(original=len(observation), path=log_path), 0


def mask_placeholder(content, log_path):
    """Placeholder for a masked observation: exit code, original length, and the
    log file holding the full text, so reading it back never means re-running
    the command."""
    match = _EXIT_CODE.search(content)
    exit_part = f"; exit code was {match.group(1)}" if match else ""
    return (
        f"{MASK_SENTINEL} original was {len(content)} chars{exit_part}.\n"
        f" Full text: {log_path}]"
    )


def mask_old_observations(messages, tail_start, log_paths):
    """Mask tool observations outside the retained tail, in place.

    tail_start is where the verbatim tail begins (verbatim_tail_span at
    keep_tail_tokens); only tool messages before it are eligible. An observation
    is replaced only when the placeholder is shorter than what it would replace
    and the log file holding the full text is known (log_paths maps tool_call_id
    to path; calls that never executed have no log and stay verbatim, as do
    observations from earlier runs). Already-masked messages are skipped, so
    repeated passes are idempotent. Returns the characters freed.
    """
    freed = 0
    for index in range(min(tail_start, len(messages))):
        message = messages[index]
        if message.get("role") != "tool":
            continue
        content = message.get("content") or ""
        if content.startswith(MASK_SENTINEL):
            continue
        log_path = log_paths.get(message.get("tool_call_id"))
        if log_path is None:
            continue
        placeholder = mask_placeholder(content, log_path)
        if len(placeholder) >= len(content):
            continue
        message["content"] = placeholder
        freed += len(content) - len(placeholder)
    return freed


def drop_old_reasoning(messages, tail_start, save=None):
    """Delete reasoning_content from assistant messages outside the tail, in place.

    The field is removed outright and no placeholder is left in the body: a
    placeholder could only go into content, which would pollute the text sent
    back to the model. `save`, when given, is called as save(text, span_ordinal)
    with the original text before each drop, so the caller can persist it.
    Returns the characters freed.
    """
    ordinals = span_ordinals(messages)
    freed = 0
    for index in range(min(tail_start, len(messages))):
        message = messages[index]
        if message.get("role") != "assistant":
            continue
        text = message.get("reasoning_content")
        if not text:
            continue
        if save is not None:
            save(text, ordinals[index])
        freed += len(text)
        del message["reasoning_content"]
    return freed


def elide_old_commands(messages, tail_start, head_chars, save):
    """Elide command bodies in tool_calls of assistant messages outside the tail.

    Every field of a tool_calls entry except `arguments` is preserved
    byte-for-byte (Gemini signs its thought signature on those fields; a request
    missing it is rejected). Only the `command` value of a parsed arguments
    payload is rewritten, keeping its first head_chars characters plus an
    elision note naming the file holding the full text; every other payload key
    is carried over. A payload that does not parse, carries no string `command`,
    was elided by an earlier pass, or would not shrink, is skipped. save is
    called as save(full_command, span_ordinal, call_index) before each rewrite
    and returns the path the note names. Returns the characters freed.
    """
    ordinals = span_ordinals(messages)
    freed = 0
    for index in range(min(tail_start, len(messages))):
        message = messages[index]
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        for call_index, call in enumerate(tool_calls, start=1):
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            raw = function.get("arguments")
            if not isinstance(raw, str):
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            command = payload.get("command")
            if not isinstance(command, str):
                continue
            if COMMAND_ELISION_MARKER in command:
                continue
            if len(command) <= head_chars:
                continue
            path = save(command, ordinals[index], call_index)
            elided = command[:head_chars] + COMMAND_ELISION_NOTE.format(path=path)
            new_raw = json.dumps({**payload, "command": elided})
            if len(new_raw) >= len(raw):
                continue
            function["arguments"] = new_raw
            freed += len(raw) - len(new_raw)
    return freed


def verbatim_tail_span(messages, tail_budget_tokens):
    """(start, end) slice of the newest whole steps filling the tail budget.

    Steps are taken from newest backwards (an assistant message and all its tool
    messages are inseparable) until the budget (chars/4) is filled; the last whole
    step is always included, even if it alone exceeds the budget. (len, len) when
    there are no steps at all.
    """
    spans = split_steps(messages)
    budget_chars = tail_budget_tokens * 4
    kept_start = len(messages)
    total = 0
    for start, end in reversed(spans):
        span_chars = sum(est_message_chars(message) for message in messages[start:end])
        if kept_start < len(messages) and total + span_chars > budget_chars:
            break
        kept_start = start
        total += span_chars
    return kept_start, len(messages)


def drop_oldest_middle_half(messages, tail_start, head_size=2):
    """Drop the oldest half of the maskable middle, cutting on a step boundary.

    The middle is messages[head_size:tail_start]: everything the summarize layer
    would fold into the summary. The cut is rounded up to a step start so an
    assistant message is never separated from its tool messages. Mutates the list
    in place; returns the number of messages dropped.
    """
    middle = tail_start - head_size
    if middle <= 2:
        return 0
    cut = head_size + middle // 2
    boundaries = {head_size + start for start, _ in split_steps(messages[head_size:tail_start])}
    boundaries.add(tail_start)
    cut = min(boundary for boundary in boundaries if boundary >= cut)
    del messages[head_size:cut]
    return cut - head_size
