"""Thin user-facing CLI for charlie-code. Argument parsing + orchestration only.

All agent logic lives in src/ (agent.py, model.py, environment.py).
"""

import contextlib
import json
import os
import sys

import typer

from agent import Agent, load_config
from environment import Environment
from model import Model
from skills import load_skill_catalog


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
    api_base: str = typer.Option(
        None, "--api-base", help="OpenAI-compatible API base URL override."
    ),
    cwd: str = typer.Option(
        None, "--cwd", help="Repo directory the agent operates in (default: current dir)."
    ),
    skills_root: str = typer.Option(
        None, "--skills-root", help="Agent Skills root directory override."
    ),
    steps: int = typer.Option(None, "--steps", help="Hard step limit override."),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit an NDJSON event stream on stdout instead of the human trajectory.",
    ),
):
    if json_output:
        os.environ.setdefault("LITELLM_LOG", "ERROR")
        json_stream = sys.stdout

        def _emit(event):
            print(json.dumps(event), file=json_stream, flush=True)
    else:
        _emit = None

    config = load_config()

    model_name = (
        model or os.environ.get("CHARLIE_CODE_MODEL") or config["model"]["model_name"]
    )
    base_url = (
        api_base or os.environ.get("CHARLIE_CODE_API_BASE") or config["model"]["api_base"]
    )
    api_key = os.environ.get("CHARLIE_CODE_API_KEY", "EMPTY")
    working_dir = cwd or os.getcwd()
    resolved_skills_root = (
        skills_root
        or os.environ.get("CHARLIE_CODE_SKILLS_ROOT")
        or config["skills"]["root"]
    )
    step_limit = steps if steps is not None else config["agent"]["step_limit"]

    agent = Agent(
        model=Model(model_name=model_name, api_base=base_url, api_key=api_key),
        environment=Environment(cwd=working_dir, timeout=config["environment"]["timeout"]),
        templates=config["templates"],
        step_limit=step_limit,
        skills_catalog=load_skill_catalog(resolved_skills_root),
        emit=_emit,
    )

    if json_output:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = agent.run(task)
        except RuntimeError as exc:
            _emit({"type": "error", "message": str(exc)})
            raise typer.Exit(1) from None
        except Exception as exc:
            _emit({"type": "error", "message": str(exc)})
            raise typer.Exit(1) from None

        _emit({
            "type": "result",
            "completed": result["completed"],
            "n_steps": result["n_steps"],
            "usage": result["usage"],
        })
        return

    result = agent.run(task)
    _print_trajectory(result)


def main():
    typer.run(run)


if __name__ == "__main__":
    main()
