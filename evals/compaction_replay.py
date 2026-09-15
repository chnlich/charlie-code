#!/usr/bin/env python3
"""Offline replay of the compaction ladder over stored session histories.

  python evals/compaction_replay.py --sessions ~/.charlie-code/sessions --min-tokens 100000

Reads every session state file under --sessions, restores masked observations
to the original length their placeholders state, and selects the histories
whose chars/4 estimate is at or above --min-tokens. For each one it replays the
ladder's first three levels (mask, reasoning, command) with the packaged
defaults and reports, per session, the levels that ran, what each freed, and
what was left, with before/after columns. A history the first three levels
cannot bring under the target line is marked SUMMARIZE: that level needs a
model call, so offline it is reported rather than executed.

The replay is the acceptance harness for the ladder's reach: the share of
large histories that reach the target line without the summarize level, and
the median remainder. It reads session files and writes only to standard
output; every default path derives from Path.home() at runtime, and no session
content ever enters the repository.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from compact import (  # noqa: E402
    MASK_SENTINEL,
    drop_old_reasoning,
    elide_old_commands,
    est_message_chars,
    mask_old_observations,
    original_observation_chars,
    verbatim_tail_span,
)

DEFAULT_SESSIONS_DIR = Path.home() / ".charlie-code" / "sessions"
DEFAULT_CONFIG_PATH = REPO_ROOT / "src" / "config" / "default.yaml"
STATE_PROTOCOL = "tool-calls-v1"

# Fixed-length stand-in for the run-started-at directory, so fabricated
# placeholder paths have the length the real ones have.
REPLAY_RUN_STAMP = "00000000T000000Z"


def restored_message_chars(message, image_tokens):
    """chars/4 input for one message, with masked observations restored.

    A masked observation's placeholder states the original length; that length
    is what the pre-compaction history held, so the estimate uses it instead of
    the placeholder's.
    """
    content = message.get("content")
    if isinstance(content, str) and content.startswith(MASK_SENTINEL):
        return original_observation_chars(content)
    return est_message_chars(message, image_tokens)


def restored_tokens(messages, image_tokens):
    return sum(restored_message_chars(message, image_tokens)
               for message in messages) // 4


def restore_copy(messages):
    """Shallow per-message copy with masked observations re-inflated to their
    original length, so the real ladder functions measure the pre-compaction
    history. The copy is what the replay rewrites; the file is never touched."""
    copy = []
    for message in messages:
        clone = dict(message)
        content = clone.get("content")
        if isinstance(content, str) and content.startswith(MASK_SENTINEL):
            clone["content"] = "x" * original_observation_chars(content)
        copy.append(clone)
    return copy


def replay_ladder(messages, config, sessions_dir, session_id):
    """Run the ladder's first three levels over a restored copy.

    Returns (layers_run, freed_by_level, after_tokens, needs_summarize,
    anchored_reached_at). The stop decision is the plan's replay methodology: a
    fresh estimate of the rewritten history after each level. The agent itself
    stops on its anchored formula (measured prompt_tokens minus freed/4), which
    for tool-call-heavy histories reads higher than a fresh estimate - the
    estimator re-serializes tool_calls, so escapable command characters shrink
    the estimate faster than the raw freed chars. anchored_reached_at reports
    whether the anchored formula would have stopped by then too, so the row
    shows both readings.
    """
    image_tokens = config["image_tokens"]
    ladder = list(config["ladder"])
    keep_tail_tokens = config["keep_tail_tokens"]
    before_tokens = restored_tokens(messages, image_tokens)
    run_dir = sessions_dir / f"{session_id}.d" / REPLAY_RUN_STAMP
    copy = restore_copy(messages)
    tail_start, _ = verbatim_tail_span(copy, keep_tail_tokens)

    # Every tool message is treated as having a command log at the path its
    # placeholder would name; offline the log itself is never needed, only the
    # placeholder length it produces.
    ordinals = {}
    ordinal = 0
    for index, message in enumerate(copy):
        if message.get("role") == "assistant":
            ordinal += 1
        ordinals.setdefault(index, ordinal)
    log_paths = {}
    calls_by_ordinal = {}
    for index, message in enumerate(copy):
        if message.get("role") != "tool":
            continue
        ordinal = ordinals[index]
        call_index = calls_by_ordinal.get(ordinal, 0) + 1
        calls_by_ordinal[ordinal] = call_index
        log_paths[message.get("tool_call_id")] = str(
            run_dir / f"s-{ordinal}-{call_index}.log"
        )

    def save_command(text, span_ordinal, call_index):
        return str(run_dir / f"s-{span_ordinal}-{call_index}.command.txt")

    def save_reasoning(text, span_ordinal):
        return str(run_dir / f"s-{span_ordinal}.reasoning.txt")

    def save_noop(text, *parts):
        return str(run_dir / "replay.txt")

    floor_tokens = keep_tail_tokens + (
        est_message_chars(messages[0], image_tokens)
        + est_message_chars(messages[1], image_tokens)) // 4
    target = max(int(config["target_fraction"] * config["context_window"]),
                 floor_tokens)
    freed_by_level = {}
    layers_run = []
    freed_total = 0
    needs_summarize = False
    anchored_reached_at = None
    after_tokens = before_tokens
    for level in ladder:
        if level == "summarize":
            needs_summarize = True
            break
        if level == "mask":
            freed = mask_old_observations(copy, tail_start, log_paths)
        elif level == "reasoning":
            freed = drop_old_reasoning(copy, tail_start, save=save_noop)
        elif level == "command":
            freed = elide_old_commands(copy, tail_start,
                                       config["command_head_chars"],
                                       save=save_command)
        else:
            raise RuntimeError(f"unknown ladder level {level!r}")
        freed_by_level[level] = freed
        layers_run.append(level)
        freed_total += freed
        after_tokens = sum(est_message_chars(message, image_tokens)
                           for message in copy) // 4
        if after_tokens <= target:
            if max(0, before_tokens - freed_total // 4) <= target:
                anchored_reached_at = len(layers_run)
            break
    return (layers_run, freed_by_level, after_tokens, needs_summarize, target,
            anchored_reached_at)


def load_messages(path):
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("protocol") != STATE_PROTOCOL:
        raise ValueError("not a tool-calls-v1 session state file")
    messages = state.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("session state has no compactable history")
    return messages


def main():
    parser = argparse.ArgumentParser(
        description="Replay the compaction ladder over stored session histories."
    )
    parser.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS_DIR,
                        help="directory of session state files "
                             "(default: %(default)s)")
    parser.add_argument("--min-tokens", type=int, default=100000,
                        help="select histories estimated at or above this many "
                             "tokens, masked observations restored "
                             "(default: %(default)s)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="compact config to replay with (default: %(default)s)")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())["compact"]
    window = config["context_window"]
    threshold = int(config["threshold_fraction"] * window)

    paths = sorted(args.sessions.glob("*.json"))
    if not paths:
        raise SystemExit(f"no session state files under {args.sessions}")

    rows = []
    skipped = 0
    for path in paths:
        try:
            messages = load_messages(path)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            skipped += 1
            continue
        before = restored_tokens(messages, config["image_tokens"])
        if before < args.min_tokens:
            continue
        (layers, freed, after, needs_summarize, target,
         anchored_reached_at) = replay_ladder(messages, config, args.sessions,
                                              path.stem)
        rows.append((path.stem, before, after, target, layers, freed,
                     needs_summarize, anchored_reached_at))

    rows.sort(key=lambda row: row[1], reverse=True)
    print(f"compaction ladder replay  (window={window} threshold={threshold} "
          f"keep_tail={config['keep_tail_tokens']} "
          f"target_fraction={config['target_fraction']})")
    print(f"selected {len(rows)} of {len(paths)} session files at or above "
          f"{args.min_tokens} tokens (restored estimate); "
          f"skipped {skipped} unreadable")
    print()
    header = (f"{'session':<38} {'before':>8} {'after':>8} {'target':>8} "
              f"{'freed m/r/c':>17}  levels")
    print(header)
    print("-" * len(header))
    reached = []
    anchored_disagrees = 0
    for (session_id, before, after, target, layers, freed, needs_summarize,
         anchored_reached_at) in rows:
        mark = " SUMMARIZE" if needs_summarize else ""
        if not needs_summarize and anchored_reached_at is None:
            mark += " (anchored keeps going)"
            anchored_disagrees += 1
        freed_text = "/".join(str(freed.get(level, 0))
                              for level in ("mask", "reasoning", "command"))
        print(f"{session_id:<38} {before:>8} {after:>8} {target:>8} "
              f"{freed_text:>17}  {'+'.join(layers)}{mark}")
        if not needs_summarize:
            reached.append(after)
    print()
    if rows:
        print(f"reached the target without summarize: "
              f"{len(reached)}/{len(rows)} "
              f"({100 * len(reached) / len(rows):.1f}%)")
        if anchored_disagrees:
            print(f"of those, {anchored_disagrees} stop on the fresh estimate "
                  f"but the agent's anchored formula would keep escalating")
    if reached:
        print(f"median remaining after the ladder: {int(statistics.median(reached))}")


if __name__ == "__main__":
    main()
