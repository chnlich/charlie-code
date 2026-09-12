"""Layered context compaction helpers.

Pure functions over the flat message list, used by agent.Agent:

- Entry budget: bound every step's observations before they enter history, so no
  single-step tool burst can blow the window on its own.
- Mask layer (lossless): replace old tool observations with placeholders that keep
  the exit code and original length; the command can be re-run to re-read.
- Tail cutter for the summarize layer: whole steps, newest first, up to a budget;
  an assistant message and its tool messages are inseparable.

Token estimates here use chars // 4. The agent anchors both the trigger and the
post-mask estimate to the endpoint's measured usage.prompt_tokens, estimating only
appended content and the masking delta. Rebuilt history gets a fresh estimate.
"""

import json
import re

#: Sentinel prefix marking a tool message whose observation was masked. Its presence
#: is what makes re-masking idempotent.
MASK_SENTINEL = "[observation masked by context compaction:"

#: Notes inserted by the per-step entry budget. Both state the original length; the
#: replacement note also says how to re-read the output.
ELISION_NOTE = (
    "\n[... observation truncated by the per-step budget: original was "
    "{original} chars; head and tail kept ...]\n"
)
REPLACEMENT_NOTE = (
    "[observation not recorded: the per-step budget was exhausted; original was "
    "{original} chars. Re-run the command with filters to re-read it.]"
)

_EXIT_CODE = re.compile(r"Exit code: (-?\d+)")


def est_message_chars(message):
    """Estimated character volume of one message (content + reasoning + calls)."""
    total = len(message.get("content") or "")
    total += len(message.get("reasoning_content") or "")
    tool_calls = message.get("tool_calls")
    if tool_calls:
        total += len(json.dumps(tool_calls))
    return total


def est_messages_tokens(messages):
    """chars/4 estimate of a message list."""
    return sum(est_message_chars(message) for message in messages) // 4


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


def bound_observation(observation, remaining):
    """Bound one observation to the step's remaining character budget.

    Returns (text_for_history, chars_charged). Fits whole when it fits; over the
    remaining budget keeps head and tail halves with an elision note stating the
    original length and charges exactly `remaining`; when nothing (or too little)
    is left, the observation is replaced by a one-line note that states the
    original length and how to re-read it, charging 0. The replacement note lines
    are the one sanctioned overrun of the step budget: bounded by the number of
    tool calls in the step, one short line each.
    """
    if len(observation) <= remaining:
        return observation, len(observation)
    note = ELISION_NOTE.format(original=len(observation))
    if remaining > len(note):
        body = remaining - len(note)
        head = body // 2
        tail = body - head
        text = observation[:head] + note + observation[len(observation) - tail:]
        return text, len(text)
    return REPLACEMENT_NOTE.format(original=len(observation)), 0


def mask_placeholder(content):
    """Placeholder for a masked observation: exit code, length, re-read hint."""
    match = _EXIT_CODE.search(content)
    exit_part = f"; exit code was {match.group(1)}" if match else ""
    return (
        f"{MASK_SENTINEL} original was {len(content)} chars{exit_part}. "
        f"Re-run the command (with filters) to re-read it.]"
    )


def mask_old_observations(messages, keep_steps):
    """Mask tool observations outside the most recent `keep_steps` steps, in place.

    Only role=tool `content` fields are rewritten; assistant messages stay
    byte-identical and tool_call_id pairing is untouched. Already-masked messages
    are skipped, so repeated passes are idempotent. Returns the number of
    observations masked in this pass.
    """
    spans = split_steps(messages)
    old_spans = spans[:-keep_steps] if keep_steps else spans
    masked = 0
    for start, end in old_spans:
        for index in range(start, end):
            message = messages[index]
            if message.get("role") != "tool":
                continue
            content = message.get("content") or ""
            if content.startswith(MASK_SENTINEL):
                continue
            message["content"] = mask_placeholder(content)
            masked += 1
    return masked


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
