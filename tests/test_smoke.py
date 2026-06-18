"""Smoke test for the agent loop and bash-block parsing, with the model MOCKED.

No network or SGLang server is touched: we construct a real Model (cheap, no I/O at
construction time) and monkeypatch its `query` to return canned responses, ending
with the completion sentinel.
"""

from agent import COMPLETION_SENTINEL, Agent, load_config
from environment import Environment
from model import Model


def test_loop_parses_bash_and_completes(tmp_path, monkeypatch):
    responses = iter([
        # 1) No bash block -> should trigger the format reminder, not crash.
        "I will start now.",
        # 2) A real command that creates a file.
        "Let me create the file.\n```bash\necho hi > out.txt\n```",
        # 3) Completion sentinel -> ends the session.
        f"All done.\n```bash\necho {COMPLETION_SENTINEL}\n```",
    ])

    config = load_config()
    model = Model(model_name="dummy", api_base="http://unused", api_key="EMPTY")
    monkeypatch.setattr(model, "query", lambda messages: next(responses))

    agent = Agent(
        model=model,
        environment=Environment(cwd=str(tmp_path), timeout=10),
        templates=config["templates"],
        step_limit=40,
    )
    result = agent.run("create out.txt containing hi, then finish")

    assert result["completed"] is True
    assert result["n_steps"] == 3
    assert (tmp_path / "out.txt").read_text().strip() == "hi"

    # Step 1: no bash block -> reminder observation, no command.
    assert result["steps"][0]["command"] is None
    assert "No bash code block" in result["steps"][0]["observation"]
    # Step 2: the command was parsed and executed (exit code 0).
    assert result["steps"][1]["command"] == "echo hi > out.txt"
    assert result["steps"][1]["returncode"] == 0
    # Step 3: completion sentinel command terminated the loop.
    assert COMPLETION_SENTINEL in result["steps"][2]["command"]
