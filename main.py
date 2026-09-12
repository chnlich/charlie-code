"""Thin user-facing CLI for charlie-code. Argument parsing + orchestration only.

All agent logic lives in src/ (agent.py, model.py, environment.py).
"""

import contextlib
import json
import os
import sys
import uuid
from pathlib import Path

import typer

from agent import Agent, IMAGE_MIME, load_config
from environment import Environment
from model import Model
from skills import find_repo_root, load_skill_catalog


def _read_task(task_file):
    """The task text from --task-file, verbatim.

    Decoding is strict UTF-8 (no errors="replace": the task is an explicit
    user choice, so a bad byte stops the run), and empty-after-strip content
    is an error. Failure always names the flag.
    """
    source = repr(task_file)
    try:
        raw = Path(task_file).read_bytes()
    except OSError as exc:
        raise typer.BadParameter(
            f"--task-file: cannot read {task_file!r}: {exc.strerror or exc}"
        ) from None
    try:
        task = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise typer.BadParameter(
            f"--task-file: {source} is not valid UTF-8 ({exc})"
        ) from None
    if not task.strip():
        raise typer.BadParameter(f"--task-file: {source} contains no task text")
    return task


# --image contract: at most 4 images per task, each at most 4 MiB measured
# on the raw file bytes, before any base64 encoding.
IMAGE_COUNT_LIMIT = 4
IMAGE_SIZE_LIMIT = 4 * 1024 * 1024


def _validate_images(images):
    """Fail fast on any --image violation, before any Agent construction.

    Same "failure always names the flag" style as _read_task: every message
    starts with "--image: " and names the concrete cause. Each file is read
    once here to prove readability and measure its raw size; the agent reads
    the (already validated) files again when it builds the message parts.
    """
    if len(images) > IMAGE_COUNT_LIMIT:
        fifth = str(images[IMAGE_COUNT_LIMIT])
        raise typer.BadParameter(
            f"--image: at most {IMAGE_COUNT_LIMIT} images per task; the 5th "
            f"occurrence ({fifth!r}) exceeds the count cap"
        )
    for path in images:
        name = str(path)
        if not path.exists():
            raise typer.BadParameter(f"--image: {name!r} does not exist")
        if not path.is_file():
            raise typer.BadParameter(f"--image: {name!r} is not a file")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise typer.BadParameter(
                f"--image: cannot read {name!r}: {exc.strerror or exc}"
            ) from None
        suffix = path.suffix.lower().lstrip(".")
        if suffix not in IMAGE_MIME:
            raise typer.BadParameter(
                f"--image: {name!r} has unsupported suffix {path.suffix!r}; "
                f"expected one of {', '.join(IMAGE_MIME)}"
            )
        if len(raw) > IMAGE_SIZE_LIMIT:
            raise typer.BadParameter(
                f"--image: {name!r} is {len(raw)} bytes; the limit is "
                f"{IMAGE_SIZE_LIMIT} bytes (4 MiB)"
            )
    return list(images)


def _print_log_retention(environment):
    print(f"Command logs retained at: {environment.log_dir}", file=sys.stderr)


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
    task_file: str = typer.Option(
        ...,
        "--task-file",
        help="Read the task text from PATH. The file's full UTF-8 text is "
        "the task, verbatim; it never rides argv.",
    ),
    image: list[Path] = typer.Option(
        [],
        "--image",
        help="Attach an image to the task message; repeat up to 4 times in "
        "reference order. Each PATH must be a readable file with suffix "
        "png/jpg/jpeg/gif/webp and at most 4 MiB.",
    ),
    model: str = typer.Option(None, "--model", help="litellm model id override."),
    api_base: str = typer.Option(
        None, "--api-base", help="OpenAI-compatible API base URL override."
    ),
    cwd: str = typer.Option(
        None, "--cwd", help="Repo directory the agent operates in (default: current dir)."
    ),
    skills_root: str = typer.Option(
        None,
        "--skills-root",
        help="Host-level Agent Skills root; replaces the configured roots for this run.",
    ),
    resume: str = typer.Option(None, "--resume", help="Resume a previous session id."),
    session_dir: str = typer.Option(
        None, "--session-dir", help="Session state directory override."
    ),
    steps: int = typer.Option(None, "--steps", help="Hard step limit override."),
    context_window: int = typer.Option(
        None,
        "--context-window",
        help="Compaction context-window override, in tokens, for this invocation.",
    ),
    stream: bool | None = typer.Option(
        None,
        "--stream/--no-stream",
        help="Stream the model call. Under streaming the timeout is the silence "
        "bound between chunks, so a call that keeps producing is never cut off; "
        "under --no-stream it is the whole-call budget. Default: model.stream "
        "in the config (streaming).",
    ),
    timeout_seconds: int | None = typer.Option(
        None,
        "--timeout-seconds",
        help="Model-call budget passed to litellm as `timeout`: the silence bound "
        "between streamed chunks when streaming, the whole-call bound when not.",
    ),
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

    task = _read_task(task_file)
    images = _validate_images(image)
    if context_window is not None and context_window < 1:
        raise typer.BadParameter("--context-window must be an integer >= 1.")
    if timeout_seconds is not None and timeout_seconds < 1:
        raise typer.BadParameter(
            f"--timeout-seconds must be a positive integer (seconds), "
            f"got {timeout_seconds}."
        )

    config = load_config()
    if context_window is not None:
        # Per-invocation override only; the rest of the compact block is untouched.
        config["compact"]["context_window"] = context_window

    model_name = (
        model or os.environ.get("CHARLIE_CODE_MODEL") or config["model"]["model_name"]
    )
    base_url = (
        api_base or os.environ.get("CHARLIE_CODE_API_BASE") or config["model"]["api_base"]
    )
    api_key = os.environ.get("CHARLIE_CODE_API_KEY", "EMPTY")
    if stream is None:
        stream = config["model"]["stream"]
    if timeout_seconds is None:
        timeout_seconds = config["model"]["timeout_seconds"]
    working_dir = cwd or os.getcwd()
    # Skill roots, repo level first so a repo skill wins a name collision: the repo
    # directories under the git worktree root that contains the working directory
    # (none outside a repo), then the host roots. An explicit host root (flag, then
    # environment variable) replaces the configured host root list for this run.
    host_root_override = skills_root or os.environ.get("CHARLIE_CODE_SKILLS_ROOT")
    host_roots = [host_root_override] if host_root_override else config["skills"]["roots"]
    repo_root = find_repo_root(working_dir)
    repo_skill_dirs = (
        [str(repo_root / rel) for rel in config["skills"]["repo_dirs"]] if repo_root else []
    )
    skill_roots = repo_skill_dirs + list(host_roots)
    resolved_session_dir = os.path.expanduser(
        session_dir
        or os.environ.get("CHARLIE_CODE_SESSION_DIR")
        or config["session"]["dir"]
    )
    os.makedirs(resolved_session_dir, exist_ok=True)
    session_id = resume or str(uuid.uuid4())
    state_file = os.path.join(resolved_session_dir, f"{session_id}.json")
    step_limit = steps if steps is not None else config["agent"]["step_limit"]

    agent = Agent(
        model=Model(
            model_name=model_name,
            api_base=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            stream=stream,
        ),
        environment=Environment(cwd=working_dir, timeout=config["environment"]["timeout"]),
        templates=config["templates"],
        step_limit=step_limit,
        skills_catalog=load_skill_catalog(skill_roots),
        emit=_emit,
        state_file=state_file,
        resume=resume is not None,
        images=images,
        compact=config["compact"],
    )

    if json_output:
        _emit({"type": "session", "session_id": session_id})
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = agent.run(task)
        except KeyboardInterrupt:
            agent.environment.sweep()
            _print_log_retention(agent.environment)
            raise
        except RuntimeError as exc:
            _emit({"type": "error", "message": str(exc)})
            _print_log_retention(agent.environment)
            raise typer.Exit(1) from None
        except Exception as exc:
            _emit({"type": "error", "message": str(exc)})
            _print_log_retention(agent.environment)
            raise typer.Exit(1) from None

        agent.environment.cleanup_log_dir()
        _emit({
            "type": "result",
            "completed": result["completed"],
            "n_steps": result["n_steps"],
            "final_output": result["final_output"],
            "usage": result["usage"],
        })
        return

    try:
        result = agent.run(task)
    except KeyboardInterrupt:
        agent.environment.sweep()
        _print_log_retention(agent.environment)
        raise
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        _print_log_retention(agent.environment)
        raise typer.Exit(1) from None

    agent.environment.cleanup_log_dir()
    _print_trajectory(result)


def main():
    typer.run(run)


if __name__ == "__main__":
    main()
