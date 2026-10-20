"""The transcript: the shape of each record, and the agent writing one record
per message as it enters the context, before the state file, across resets and
resumes."""

import json
from pathlib import Path

import pytest
from litellm.exceptions import ContextWindowExceededError

import transcript as transcript_module
from agent import STATE_PROTOCOL, Agent, load_config, render
from conftest import assistant, tool_call
from environment import Environment
from transcript import PREAMBLE, Transcript

OVERFLOW = "OVERFLOW"
FIXED_TIME = "2026-01-01T00:00:00Z"
# The eight commands of a real failing session, sanitized to their line shapes.
REAL_COMMANDS = json.loads(
    (Path(__file__).parent / "fixtures" / "clc_heredoc_commands_sanitized.json").read_text(
        encoding="utf-8"))


class _Model:
    def __init__(self, *replies):
        self.model_name = "openai/fake-model"
        self._replies = iter(replies)
        self.last_prompt_tokens = None
        self.last_cached_tokens = 0

    def query(self, messages, tools=None):
        reply = next(self._replies)
        if reply == OVERFLOW:
            raise ContextWindowExceededError("too long", model="fake", llm_provider="fake")
        self.last_prompt_tokens = 100
        return reply

    def usage(self):
        return {"n_calls": 1, "input_tokens": 2, "output_tokens": 3}


def _agent(tmp_path, model, state_file, resume=False, step_limit=6, emit=None):
    (tmp_path / "logs").mkdir(exist_ok=True)
    return Agent(
        model=model,
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                                kill_after_seconds=10, log_dir=str(tmp_path / "logs")),
        templates=load_config()["templates"],
        step_limit=step_limit,
        emit=emit,
        compact={"context_window": 10000, "threshold_fraction": 0.5,
                 "command_observation_chars": 5000},
        state_file=str(state_file) if state_file else None,
        resume=resume,
    )


def _state_file(tmp_path, name="abc"):
    path = tmp_path / "sessions" / f"{name}.json"
    path.parent.mkdir(exist_ok=True)
    return path


def _transcript_text(state_file):
    return (state_file.parent / f"{state_file.stem}.d" / "transcript.md").read_text()


def _pointer(state_file):
    session_dir = Path(state_file).with_suffix(".d")
    return render(load_config()["templates"]["reset_pointer"],
                  transcript=str(session_dir / "transcript.md"), session_dir=str(session_dir))


# --- record shapes ---------------------------------------------------------------

def test_record_shapes(monkeypatch):
    monkeypatch.setattr(transcript_module, "_utc_now", lambda: FIXED_TIME)

    assert Transcript.user_record({"role": "user", "content": "do it\n\n"}) == (
        f"\n## user · {FIXED_TIME}\ndo it\n")
    parts = [{"type": "text", "text": "task text\n"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAECAw=="}}]
    assert Transcript.user_record({"role": "user", "content": parts}) == (
        f"\n## user · {FIXED_TIME}\ntask text\n[image 1: image/png, 4 bytes]\n")

    message = {"role": "assistant", "content": "looking", "reasoning_content": "think",
               "tool_calls": [tool_call(1, command="ls -la"), tool_call(2, name="nope", x=1)]}
    assert Transcript.assistant_record(message, 3) == (
        f"\n## assistant · step 3 · v2 · {FIXED_TIME}\n"
        "```text\nthink\n```\n"
        "```text\nlooking\n```\n"
        "```\n$ ls -la\n```\n"
        "```\n$ {\"x\": 1}\n```\n")
    assert Transcript.assistant_record({"role": "assistant", "content": ""}, 4) == (
        f"\n## assistant · step 4 · v2 · {FIXED_TIME}\n")

    assert Transcript.tool_stub(0, "x" * 12345, "/logs/s-1-1.log") == (
        "-> exit 0 · 12,345 chars · 1 lines · /logs/s-1-1.log\n")
    assert Transcript.tool_stub(1, "a\nb\n", "/logs/s-1-2.log") == (
        "-> exit 1 · 4 chars · 3 lines · /logs/s-1-2.log\n")
    assert Transcript.tool_stub(2, "", "/logs/s-1-3.log") == (
        "-> exit 2 · 0 chars · 0 lines · /logs/s-1-3.log\n")
    assert Transcript.invalid_call_stub("Error: no such tool.\nThe only tool is bash.") == (
        "-> invalid call · Error: no such tool.\n")
    assert Transcript.interrupted_stub() == "-> interrupted\n"
    assert Transcript.reset_record("threshold", 12345, 678) == (
        f"\n## reset · {FIXED_TIME} · threshold · pre 12,345 tokens · post 678 tokens\n")


def test_append_creates_the_file_with_the_preamble_once(tmp_path):
    path = tmp_path / "abc.d" / "transcript.md"
    writer = Transcript(path, "abc")

    writer.append("x\n")
    writer.append("y\n")

    assert path.read_text() == PREAMBLE.format(session_id="abc") + "x\ny\n"
    assert path.read_text().splitlines()[0] == "# charlie-code transcript · session abc"


# --- reading records back ----------------------------------------------------------

FORM_FEED_VARIANT = REAL_COMMANDS[0].replace("\n", "\x0c\n", 1)

# Written by the 63b5376 writer: unfenced, so each record reads by the line rules.
# The `## Step one` line inside step 1's heredoc ends that record, as it always did.
LEGACY_TRANSCRIPT = """\
# charlie-code transcript · session abc
# One record per message the model saw, appended as it entered the context; the system message
# is not recorded. "$ " opens a command; "-> " is its result: exit code, output size, and the
# log file holding the full output. Command outputs live only in those log files.

## user · 2026-01-01T00:00:00Z
do it

## assistant · step 1 · 2026-01-01T00:00:01Z
think
looking
$ cat > notes.md <<'EOF'
# Plan
## Step one
EOF
-> exit 0 · 33 chars · 6 lines · /logs/s-1-1.log
$ sleep 9 &
-> background · pid 42 · /logs/s-1-2.log

## user · 2026-01-01T00:00:02Z
reminder
-> background s-1-2 exit 0 · 3 chars · 1 lines · /logs/s-1-2.log

## assistant · step 2 · 2026-01-01T00:00:03Z
$ ls
-> interrupted
$ bad
-> invalid call · Error: no such tool.

## assistant · step 3 · 2026-01-01T00:00:04Z
done

## assistant · step 4 · 2026-01-01T00:00:05Z
$ python3 - <<'EOF'
print(1)

print(2)
EOF
-> exit 0 · 4 chars · 2 lines · /logs/s-4-1.log
$ tail -n 2 notes.md
-> exit 0 · 3 chars · 1 lines · /logs/s-4-2.log

## reset · 2026-01-01T00:00:06Z · threshold · pre 5,000 tokens · post 700 tokens
"""


@pytest.mark.parametrize(
    "command",
    REAL_COMMANDS + [FORM_FEED_VARIANT],
    ids=[f"cmd-{number}" for number in range(1, len(REAL_COMMANDS) + 1)] + ["cmd-1-form-feed"],
)
def test_a_heredoc_command_reads_back_whole_from_its_record(tmp_path, command):
    stub = Transcript.tool_stub(0, "ok\n", "/logs/s-1-1.log")
    message = {"role": "assistant", "content": "",
               "tool_calls": [tool_call(1, command=command)]}
    writer = Transcript(tmp_path / "transcript.md", "abc")
    writer.append(Transcript.assistant_record(message, 1) + stub)

    (record,) = writer.step_records()

    assert record["commands"] == [command.rstrip("\n")]
    assert record["tool_stubs"] == [stub]


def test_fenced_bodies_hold_their_own_structure_lines(tmp_path):
    stub = Transcript.tool_stub(0, "ok\n", "/logs/s-1-1.log")
    message = {
        "role": "assistant",
        "content": "$ not a command\n## not a header\n-> not a stub",
        "reasoning_content": "```\n````\nthink",
        "tool_calls": [tool_call(1, command="cat <<'EOF'\n```\n## x\nEOF")],
    }
    writer = Transcript(tmp_path / "transcript.md", "abc")
    writer.append(Transcript.assistant_record(message, 1) + stub)

    assert writer.step_records() == [{
        "step": 1, "commands": ["cat <<'EOF'\n```\n## x\nEOF"], "tool_stubs": [stub]}]


def test_legacy_records_read_exactly_as_the_line_rules_read_them(tmp_path):
    path = tmp_path / "transcript.md"
    path.write_text(LEGACY_TRANSCRIPT, encoding="utf-8")

    assert Transcript(path, "abc").step_records() == [
        {"step": 1, "commands": ["cat > notes.md <<'EOF'\n# Plan"], "tool_stubs": [None]},
        {"step": 2, "commands": ["ls", "bad"],
         "tool_stubs": ["-> interrupted\n", "-> invalid call · Error: no such tool.\n"]},
        {"step": 3, "commands": [], "tool_stubs": []},
        {"step": 4, "commands": ["python3 - <<'EOF'\nprint(1)\n\nprint(2)\nEOF",
                                 "tail -n 2 notes.md"],
         "tool_stubs": ["-> exit 0 · 4 chars · 2 lines · /logs/s-4-1.log\n",
                        "-> exit 0 · 3 chars · 1 lines · /logs/s-4-2.log\n"]},
    ]


def test_a_malformed_assistant_header_drops_its_record_and_reading_goes_on(tmp_path):
    path = tmp_path / "transcript.md"
    path.write_text(
        "\n## assistant · step two · 2026-01-01T00:00:00Z\n$ lost\n"
        "-> exit 0 · 1 chars · 1 lines · /logs/a.log\n"
        "\n## assistant · step 4 · v2 · 2026-01-01T00:00:01Z\n```\n$ kept\n```\n"
        "-> exit 0 · 1 chars · 1 lines · /logs/b.log\n",
        encoding="utf-8",
    )

    assert Transcript(path, "abc").step_records() == [{
        "step": 4, "commands": ["kept"],
        "tool_stubs": ["-> exit 0 · 1 chars · 1 lines · /logs/b.log\n"]}]


# --- the agent writing it ----------------------------------------------------------

def test_agent_records_each_message_as_it_enters_the_context(tmp_path):
    state_file = _state_file(tmp_path)
    model = _Model(
        assistant("Checking.", tool_calls=[tool_call(1, command="printf 'hello\\nworld\\n'")]),
        assistant("Done.", reasoning_content="deep thought"),
    )

    _agent(tmp_path, model, state_file).run("say hi")

    text = _transcript_text(state_file)
    assert text.startswith(PREAMBLE.format(session_id="abc"))
    assert text.count("\n## user · ") == 1
    task_text = render(load_config()["templates"]["instance"], task="say hi").rstrip("\n")
    assert task_text + "\n" in text
    assert text.count("\n## assistant · step 1 · v2 · ") == 1
    assert ("```text\nChecking.\n```\n```\n$ printf 'hello\\nworld\\n'\n```\n"
            "-> exit 0 · 12 chars · 3 lines · ") in text
    stub = [line for line in text.splitlines() if line.startswith("-> exit 0")][0]
    log_path = stub.split(" · ")[-1]
    assert open(log_path).read() == "hello\nworld\n"
    assert "\n## assistant · step 2 · v2 · " in text
    assert "```text\ndeep thought\n```\n```text\nDone.\n```\n" in text
    assert "## system" not in text
    assert _pointer(state_file) not in text


def test_a_reset_adds_a_reset_record_and_keeps_the_pointer_out(tmp_path):
    state_file = _state_file(tmp_path)
    model = _Model(OVERFLOW, assistant("done"))

    _agent(tmp_path, model, state_file).run("finish")

    text = _transcript_text(state_file)
    user_index = text.index("\n## user · ")
    reset_index = text.index("\n## reset · ")
    assert user_index < reset_index < text.index("\n## assistant · step 1 · ")
    reset_line = text[reset_index + 1:].splitlines()[0]
    assert " · overflow · pre " in reset_line and " tokens · post " in reset_line
    assert _pointer(state_file) not in text


def test_a_resume_appends_to_the_same_transcript(tmp_path):
    state_file = _state_file(tmp_path)
    _agent(tmp_path, _Model(assistant("first")), state_file).run("turn one")
    first = _transcript_text(state_file)

    _agent(tmp_path, _Model(assistant("second")), state_file, resume=True).run("turn two")

    second = _transcript_text(state_file)
    assert second.startswith(first)
    assert second.count("# charlie-code transcript") == 1
    assert second.count("\n## user · ") == 2
    assert "turn two" in second[len(first):]


def test_the_transcript_record_lands_before_the_state_file_is_rewritten(tmp_path, monkeypatch):
    order = []
    real_append = Transcript.append
    real_persist = Agent._persist_messages
    monkeypatch.setattr(Transcript, "append",
                        lambda self, record: (order.append("transcript"), real_append(self, record)))
    monkeypatch.setattr(Agent, "_persist_messages",
                        lambda self: (order.append("persist"), real_persist(self)))
    state_file = _state_file(tmp_path)
    model = _Model(assistant(tool_calls=[tool_call(1, command="true")]), assistant("done"))

    _agent(tmp_path, model, state_file).run("finish")

    assert order.count("transcript") == 4  # task, assistant, tool, final answer
    for index, item in enumerate(order):
        if item == "transcript":
            assert order[index + 1] == "persist"


def test_a_resume_records_an_interrupted_stub_for_each_unanswered_call(tmp_path):
    state_file = _state_file(tmp_path)
    history = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "earlier task"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call(1, command="sleep 99")]},
    ]
    state_file.write_text(json.dumps({"protocol": STATE_PROTOCOL, "messages": history}))

    _agent(tmp_path, _Model(assistant("done")), state_file, resume=True).run("continue")

    text = _transcript_text(state_file)
    assert text.count("-> interrupted\n") == 1
    assert text.index("-> interrupted\n") < text.index("\n## user · ")


def test_a_session_without_a_state_file_keeps_no_transcript(tmp_path):
    agent = _agent(tmp_path, _Model(OVERFLOW, assistant("done")), state_file=None)

    result = agent.run("finish")

    assert result["completed"] is True
    assert agent.transcript is None
    assert list(tmp_path.rglob("transcript.md")) == []
    assert agent.messages[1]["content"] == render(
        load_config()["templates"]["reset_pointer"],
        transcript=str(tmp_path / "transcript.md"), session_dir=str(tmp_path))


def test_a_resumed_run_has_its_own_step_limit_and_numbers_transcript_steps_on(tmp_path):
    state_file = _state_file(tmp_path)
    first = _agent(tmp_path, _Model(assistant(tool_calls=[tool_call(1, command="printf one")]),
                                    assistant(tool_calls=[tool_call(1, command="printf two")])),
                   state_file, step_limit=2)
    with pytest.raises(RuntimeError, match="Step limit"):
        first.run("two steps")

    events = []
    resumed = _agent(tmp_path, _Model(assistant(tool_calls=[tool_call(1, command="printf three")]),
                                      assistant("done")),
                     state_file, resume=True, step_limit=2, emit=events.append)
    result = resumed.run("continue")

    assert result["completed"] is True and result["n_steps"] == 2
    assert [e["step"] for e in events if e["type"] == "context"] == [1, 2]
    assert [e["id"] for e in events if e["type"] == "command"] == ["s-1-1"]
    assert (tmp_path / "logs" / "s-1-1.log").read_text() == "three"
    records = Transcript(state_file.with_suffix(".d") / "transcript.md", "abc").step_records()
    assert [(r["step"], r["commands"]) for r in records] == [
        (1, ["printf one"]), (2, ["printf two"]), (3, ["printf three"]), (4, [])]
