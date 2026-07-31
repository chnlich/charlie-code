"""Smoke test for the agent loop end to end, with the model MOCKED.

Covers one full shape of a run: an empty reply that must not end the session, a real
bash call, and a final text answer that does.
"""

from agent import Agent, load_config
from conftest import ScriptedModel, assistant, tool_call
from environment import Environment


def test_loop_runs_tools_and_completes_on_a_text_reply(tmp_path):
    agent = Agent(
        model=ScriptedModel(
            assistant(""),  # nothing said and nothing called: keeps going
            assistant("Creating the file.",
                      tool_calls=[tool_call(1, command="echo hi > out.txt")]),
            assistant("Created out.txt with the text hi."),
        ),
        environment=Environment(cwd=str(tmp_path), timeout=10),
        templates=load_config()["templates"],
        step_limit=5,
    )

    result = agent.run("create out.txt")

    assert result["completed"] is True
    assert result["n_steps"] == 3
    assert result["final_output"] == "Created out.txt with the text hi."
    assert (tmp_path / "out.txt").read_text() == "hi\n"

    assert result["steps"][0]["note"] == "empty response"
    assert result["steps"][1]["command"] == "echo hi > out.txt"
    assert result["steps"][1]["returncode"] == 0

    roles = [message["role"] for message in agent.messages]
    assert roles == [
        "system", "user",        # prompt
        "assistant", "user",     # empty reply, then the reminder
        "assistant", "tool",     # the bash call and its result
        "assistant",             # the final answer
    ]
