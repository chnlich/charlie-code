"""Thin user-facing CLI for charlie-code. Argument parsing + orchestration only.

All agent logic lives in src/ (agent.py, model.py, environment.py).
"""

import os

import typer

from agent import Agent, load_config
from environment import Environment
from model import Model


def _print_trajectory(result):
    for idx, step in enumerate(result["steps"], start=1):
        print(f"\n{'─' * 24} step {idx} {'─' * 24}")
        if step["thought"]:
            print(f"[thought]\n{step['thought']}")
        if step["command"] is not None:
            print(f"\n[command]\n$ {step['command']}")
        print(f"\n[observation]\n{step['observation']}")

    print(f"\n{'═' * 56}")
    status = "completed" if result["completed"] else "stopped"
    usage = result["usage"]
    print(
        f"Task {status} in {result['n_steps']} step(s). "
        f"LLM calls: {usage['n_calls']}, "
        f"tokens in/out: {usage['input_tokens']}/{usage['output_tokens']}."
    )


def run(
    task: str = typer.Argument(..., help="The task for the agent to solve."),
    model: str = typer.Option(None, "--model", help="litellm model id override."),
    api_base: str = typer.Option(None, "--api-base", help="OpenAI-compatible API base URL override."),
    cwd: str = typer.Option(None, "--cwd", help="Repo directory the agent operates in (default: current dir)."),
    steps: int = typer.Option(None, "--steps", help="Hard step limit override."),
):
    config = load_config()

    model_name = model or os.environ.get("CHARLIE_CODE_MODEL") or config["model"]["model_name"]
    base_url = api_base or os.environ.get("CHARLIE_CODE_API_BASE") or config["model"]["api_base"]
    api_key = os.environ.get("CHARLIE_CODE_API_KEY", "EMPTY")
    working_dir = cwd or os.getcwd()
    step_limit = steps if steps is not None else config["agent"]["step_limit"]

    agent = Agent(
        model=Model(model_name=model_name, api_base=base_url, api_key=api_key),
        environment=Environment(cwd=working_dir, timeout=config["environment"]["timeout"]),
        templates=config["templates"],
        step_limit=step_limit,
    )
    result = agent.run(task)
    _print_trajectory(result)


def main():
    typer.run(run)


if __name__ == "__main__":
    main()
