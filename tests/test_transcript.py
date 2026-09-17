"""The transcript: the shape of each record, and the agent writing one record
per message as it enters the context, before the state file, across resets and
resumes."""

import json

from litellm.exceptions import ContextWindowExceededError

import transcript as transcript_module
from agent import STATE_PROTOCOL, Agent, load_config, render
from conftest import assistant, final_answer, tool_call
from environment import Environment
from transcript import PREAMBLE, Transcript

OVERFLOW = "OVERFLOW"
FIXED_TIME = "2026-01-01T00:00:00Z"


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


def _agent(tmp_path, model, state_file, resume=False):
    (tmp_path / "logs").mkdir(exist_ok=True)
    return Agent(
        model=model,
        environment=Environment(cwd=str(tmp_path), progress_notices_seconds=[],
                                kill_after_seconds=10, log_dir=str(tmp_path / "logs")),
        templates=load_config()["templates"],
        step_limit=6,
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
        f"\n## assistant · step 3 · {FIXED_TIME}\nthink\nlooking\n$ ls -la\n$ {{\"x\": 1}}\n")
    assert Transcript.assistant_record({"role": "assistant", "content": ""}, 4) == (
        f"\n## assistant · step 4 · {FIXED_TIME}\n")

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


# --- the agent writing it ----------------------------------------------------------

def test_agent_records_each_message_as_it_enters_the_context(tmp_path):
    state_file = _state_file(tmp_path)
    model = _Model(
        assistant("Checking.", tool_calls=[tool_call(1, command="printf 'hello\\nworld\\n'")]),
        final_answer("Done.", reasoning_content="deep thought"),
    )

    _agent(tmp_path, model, state_file).run("say hi")

    text = _transcript_text(state_file)
    assert text.startswith(PREAMBLE.format(session_id="abc"))
    assert text.count("\n## user · ") == 1
    task_text = render(load_config()["templates"]["instance"], task="say hi").rstrip("\n")
    assert task_text + "\n" in text
    assert text.count("\n## assistant · step 1 · ") == 1
    assert "\nChecking.\n$ printf 'hello\\nworld\\n'\n-> exit 0 · 12 chars · 3 lines · " in text
    stub = [line for line in text.splitlines() if line.startswith("-> exit 0")][0]
    log_path = stub.split(" · ")[-1]
    assert open(log_path).read() == "hello\nworld\n"
    assert "\n## assistant · step 2 · " in text
    assert "\ndeep thought\nDone.\n" in text
    assert "## system" not in text
    assert "Prior context" not in text


def test_reminders_are_recorded_as_user_records(tmp_path):
    state_file = _state_file(tmp_path)
    model = _Model(assistant("not finished yet", finish_reason="stop"), final_answer("done"))

    _agent(tmp_path, model, state_file).run("finish")

    text = _transcript_text(state_file)
    assert text.count("\n## user · ") == 2
    reminder = render(load_config()["templates"]["unfinished_reply_reminder"],
                      completion_sentinel=load_config()["agent"]["completion_sentinel"])
    assert reminder.rstrip("\n") + "\n" in text


def test_a_reset_adds_a_reset_record_and_keeps_the_pointer_out(tmp_path):
    state_file = _state_file(tmp_path)
    model = _Model(OVERFLOW, final_answer("done"))

    _agent(tmp_path, model, state_file).run("finish")

    text = _transcript_text(state_file)
    user_index = text.index("\n## user · ")
    reset_index = text.index("\n## reset · ")
    assert user_index < reset_index < text.index("\n## assistant · step 1 · ")
    reset_line = text[reset_index + 1:].splitlines()[0]
    assert " · overflow · pre " in reset_line and " tokens · post " in reset_line
    assert "Prior context" not in text


def test_a_resume_appends_to_the_same_transcript(tmp_path):
    state_file = _state_file(tmp_path)
    _agent(tmp_path, _Model(final_answer("first")), state_file).run("turn one")
    first = _transcript_text(state_file)

    _agent(tmp_path, _Model(final_answer("second")), state_file, resume=True).run("turn two")

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
    model = _Model(assistant(tool_calls=[tool_call(1, command="true")]), final_answer("done"))

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

    _agent(tmp_path, _Model(final_answer("done")), state_file, resume=True).run("continue")

    text = _transcript_text(state_file)
    assert text.count("-> interrupted\n") == 1
    assert text.index("-> interrupted\n") < text.index("\n## user · ")


def test_a_session_without_a_state_file_keeps_no_transcript(tmp_path):
    agent = _agent(tmp_path, _Model(OVERFLOW, final_answer("done")), state_file=None)

    result = agent.run("finish")

    assert result["completed"] is True
    assert agent.transcript is None
    assert list(tmp_path.rglob("transcript.md")) == []
    assert agent.messages[1]["content"] == render(
        load_config()["templates"]["reset_pointer"],
        transcript=str(tmp_path / "transcript.md"), session_dir=str(tmp_path))
