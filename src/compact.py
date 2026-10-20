"""Context estimates and bounded content for the agent's message history.

Token estimates use chars // 4. Before each model call, the agent adds that
estimate for messages appended since the last measured prompt, or estimates the
whole context when no measurement anchors it. Old step content leaves the
context in batches; the transcript and command logs keep the full records.
Between batches, the message list stays byte-identical for the endpoint cache.
At the reset threshold, the agent rebuilds the context as [system, pointer, task].
"""

import json

#: Tokens charged per image part in an estimate. The base64 length of an image
#: says nothing about its token count, so every image counts as this constant.
IMAGE_TOKENS = 1600

# Three recent steps stay available for follow-up work on the current task.
EXPIRY_KEEP_STEPS = 3

# A batch waits for enough old content to amortize one prefix-cache invalidation.
EXPIRY_BATCH_TOKEN_THRESHOLD = 20_000

# Long commands keep a readable prefix while the transcript remains authoritative.
EXPIRY_COMMAND_PREFIX_CHARS = 200

#: The one line that replaces the middle of an over-cap command output. It
#: states the totals of the full output and names no file: the transcript's stub
#: line for the same command carries the log path.
TRUNCATION_MARKER = "[... truncated: {chars:,} chars, {lines:,} lines total ...]"


def line_count(text):
    """Lines in `text` as the marker and the transcript stubs count them: one
    more than the newlines it contains, zero for empty text."""
    return text.count("\n") + 1 if text else 0


def est_message_chars(message):
    """Estimated character volume of one message (content + reasoning + calls).

    A parts-list content (a task message carrying images) is summed per part:
    text parts contribute their length, image parts contribute IMAGE_TOKENS * 4
    characters each.
    """
    content = message.get("content")
    if isinstance(content, list):
        total = 0
        for part in content:
            if part.get("type") == "image_url":
                total += IMAGE_TOKENS * 4
            else:
                total += len(part.get("text") or "")
    else:
        total = len(content or "")
    total += len(message.get("reasoning_content") or "")
    tool_calls = message.get("tool_calls")
    if tool_calls:
        total += len(json.dumps(tool_calls))
    return total


def est_messages_tokens(messages):
    """chars // 4 estimate of a message list."""
    return sum(est_message_chars(message) for message in messages) // 4


def truncate_observation(output, cap):
    """Bound one command output to `cap` characters before it enters the context.

    Output at or under the cap enters whole. Over the cap, the first cap // 2
    characters and the last cap - cap // 2 characters enter, separated by the
    marker line stating the full output's total characters and lines. Each
    command is bounded on its own; there is no per-step aggregate. The full
    output is untouched in the command's log file.
    """
    if len(output) <= cap:
        return output
    head = cap // 2
    tail = cap - head
    marker = TRUNCATION_MARKER.format(chars=len(output), lines=line_count(output))
    return output[:head] + "\n" + marker + "\n" + output[len(output) - tail:]


def command_expiry_preview(command, transcript_step, transcript_path):
    """Keep a long command's prefix and point to its full transcript record."""
    if len(command) <= EXPIRY_COMMAND_PREFIX_CHARS:
        return command
    marker = (
        f"\n[full command in transcript step {transcript_step} at {transcript_path}]"
    )
    return command[:EXPIRY_COMMAND_PREFIX_CHARS] + marker
