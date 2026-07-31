#!/usr/bin/env python3
"""Batch evaluation runner for charlie-code.

Materializes each task's fixture into a fresh temp dir, runs one charlie-code
episode per (task, repeat) under that dir, then executes the task's grader.
Aggregates per-run results into runs/<id>/summary.json per the plan 4.1 schema.

  python evals/run.py --suite dev --model glm52|kimi-k3 [--k 1] [--parallel 4] --out runs/<id>
  python evals/run.py --suite dev --null-check

No model calls happen under --null-check: each grader runs against a pristine
fixture dir and must judge every task unresolved.

Only the Python standard library plus pyyaml are used.
"""

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent

DEFAULT_ENV_FILE = Path.home() / ".charlie-code" / "evals.env"

FAIL_CLASSES = ("step_limit", "env_error", "wrong_answer", "infra")


# --------------------------------------------------------------------------- #
# Config: models + env resolution
# --------------------------------------------------------------------------- #

def load_models(path):
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or "models" not in data:
        raise SystemExit(f"{path}: expected a top-level 'models' mapping")
    return data


def _read_env_file(path):
    """Parse KEY=VALUE lines from an env file into a dict (empty if absent)."""
    values = {}
    p = Path(path)
    if not p.exists():
        return values
    for lineno, raw in enumerate(p.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"{p}:{lineno}: expected KEY=VALUE, got {raw!r}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def resolve_env(required_vars, env_file=DEFAULT_ENV_FILE, env=None):
    """Resolve a set of required env vars with precedence process env > file > error.

    Returns {var: value}. Raises SystemExit naming the first missing var.
    """
    env = dict(env if env is not None else os.environ)
    file_values = _read_env_file(env_file)
    resolved = {}
    for var in required_vars:
        if env.get(var):
            resolved[var] = env[var]
        elif file_values.get(var):
            resolved[var] = file_values[var]
        else:
            raise SystemExit(
                f"missing required environment variable: {var} "
                f"(set it in the process env or in {env_file})"
            )
    return resolved


def resolve_model(models_cfg, logical_id, env_file=DEFAULT_ENV_FILE, env=None):
    """Resolve {model_name, api_base, api_key} for a logical model id."""
    models = models_cfg.get("models", {})
    if logical_id not in models:
        known = ", ".join(sorted(models)) or "(none)"
        raise SystemExit(
            f"unknown model id {logical_id!r}; known ids: {known}"
        )
    entry = models[logical_id]
    resolved = resolve_env(
        [entry["model_env"], entry["base_env"]], env_file=env_file, env=env,
    )
    api_key_env = models_cfg.get("api_key_env", "CC_EVAL_API_KEY")
    api_key_default = models_cfg.get("api_key_default", "EMPTY")
    env = dict(env if env is not None else os.environ)
    file_values = _read_env_file(env_file)
    api_key = env.get(api_key_env) or file_values.get(api_key_env) or api_key_default
    return {
        "model_id": logical_id,
        "model_name": resolved[entry["model_env"]],
        "api_base": resolved[entry["base_env"]],
        "api_key": api_key,
    }


def resolve_episode_interpreter(env_file=DEFAULT_ENV_FILE, env=None):
    """Resolve the interpreter for episode subprocesses.

    Precedence is process env > ~/.charlie-code/evals.env > the current
    interpreter (sys.executable). A set-but-nonexistent CC_EVAL_PYTHON is a
    hard error naming the variable, never a silent fallback: a bare
    `python evals/run.py` must not silently measure episodes under a broken
    or wrong interpreter.
    """
    env = dict(env if env is not None else os.environ)
    file_values = _read_env_file(env_file)
    value = env.get("CC_EVAL_PYTHON") or file_values.get("CC_EVAL_PYTHON")
    if not value:
        return sys.executable
    if not Path(value).is_file():
        raise SystemExit(
            f"CC_EVAL_PYTHON points to a nonexistent interpreter: {value!r} "
            f"(set it in the process env or in {env_file})"
        )
    return value


def check_reachable(model_cfg, timeout=10.0):
    """GET <api_base>/models once. Returns True on 200, False otherwise."""
    base = model_cfg["api_base"].rstrip("/")
    url = base + "/models" if base.endswith("/v1") else base + "/v1/models"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {model_cfg['api_key']}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


# --------------------------------------------------------------------------- #
# Config: suite + tasks
# --------------------------------------------------------------------------- #

REQUIRED_TASK_FIELDS = ("id", "prompt", "fixture", "grade", "timeout_s", "step_limit")


def load_suite(suite_dir):
    """Load every <id>.yaml in a suite dir, validating fields and paths."""
    suite_dir = Path(suite_dir)
    if not suite_dir.is_dir():
        raise SystemExit(f"suite directory not found: {suite_dir}")
    tasks = []
    for path in sorted(suite_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise SystemExit(f"{path}: expected a mapping at the top level")
        missing = [f for f in REQUIRED_TASK_FIELDS if f not in data]
        if missing:
            raise SystemExit(f"{path}: missing required field(s): {', '.join(missing)}")
        task = dict(data)
        task.setdefault("tags", [])
        fixture = suite_dir / task["fixture"]
        if not fixture.is_dir():
            raise SystemExit(f"{path}: fixture directory not found: {fixture}")
        grade = suite_dir / task["grade"]
        if not grade.is_file():
            raise SystemExit(f"{path}: grade script not found: {grade}")
        task["_fixture"] = fixture
        task["_grade"] = grade
        tasks.append(task)
    if not tasks:
        raise SystemExit(f"no task YAML files found in {suite_dir}")
    return tasks


def materialize_fixture(fixture_src, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for entry in Path(fixture_src).iterdir():
        target = dest / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)


# --------------------------------------------------------------------------- #
# Episode + grader execution
# --------------------------------------------------------------------------- #

def _parse_result_event(ndjson_text):
    """Pull steps/tokens from the `result` event only (None if absent)."""
    result = None
    error = None
    for line in ndjson_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "result":
            result = event
        elif event.get("type") == "error":
            error = event
    return result, error


def run_episode(task, model_cfg, work_dir, episode_timeout, interpreter):
    """Run one charlie-code episode non-interactively; return (ndjson, fail_class, wall_s)."""
    session_dir = work_dir / ".cc-sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        interpreter, "-m", "main", "--json",
        "--cwd", str(work_dir),
        "--model", model_cfg["model_name"],
        "--api-base", model_cfg["api_base"],
        "--steps", str(task["step_limit"]),
        "--session-dir", str(session_dir),
        task["prompt"],
    ]
    child_env = dict(os.environ)
    child_env["CHARLIE_CODE_API_KEY"] = model_cfg["api_key"]
    child_env["LITELLM_LOG"] = "ERROR"
    # Force main/agent/model/environment to resolve from this repo, not from
    # whichever checkout the interpreter's editable install happens to point
    # at: PYTHONPATH is searched before site-packages .pth entries, so this
    # keeps episodes honest about which charlie-code they are measuring.
    repo_paths = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    existing_pythonpath = child_env.get("PYTHONPATH")
    if existing_pythonpath:
        repo_paths.append(existing_pythonpath)
    child_env["PYTHONPATH"] = os.pathsep.join(repo_paths)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True,
            timeout=episode_timeout, env=child_env,
        )
        wall_s = time.perf_counter() - start
    except subprocess.TimeoutExpired:
        wall_s = time.perf_counter() - start
        return "", "infra", wall_s
    except Exception:
        wall_s = time.perf_counter() - start
        return "", "infra", wall_s
    ndjson = proc.stdout
    result, error = _parse_result_event(ndjson)
    if result is not None:
        return ndjson, None, wall_s
    if error is not None:
        msg = str(error.get("message", ""))
        if "Step limit" in msg:
            return ndjson, "step_limit", wall_s
        return ndjson, "env_error", wall_s
    return ndjson, "infra", wall_s


def run_grader(grade_path, work_dir, timeout_s):
    """Run a grader script in work_dir. Returns (exit_code_or_None, fail_class).

    exit_code 0 = resolved; non-zero = wrong_answer; None = infra (timeout/crash).
    """
    argv = [sys.executable, str(grade_path)]
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True,
            timeout=timeout_s, cwd=str(work_dir),
        )
    except subprocess.TimeoutExpired:
        return None, "infra"
    except Exception:
        return None, "infra"
    if proc.returncode == 0:
        return 0, None
    return proc.returncode, "wrong_answer"


def grade_outcome(task, episode_ndjson, episode_fail_class, episode_wall_s,
                  grader_exit, grader_fail_class):
    """Combine episode + grader into one run record per the summary schema."""
    result, _ = _parse_result_event(episode_ndjson)
    steps = 0
    tokens_in = 0
    tokens_out = 0
    if result is not None:
        steps = int(result.get("n_steps", 0))
        usage = result.get("usage") or {}
        tokens_in = int(usage.get("input_tokens", 0))
        tokens_out = int(usage.get("output_tokens", 0))
    resolved = grader_exit == 0
    if resolved:
        fail_class = None
    elif episode_fail_class is not None:
        fail_class = episode_fail_class
    else:
        fail_class = grader_fail_class or "wrong_answer"
    return {
        "resolved": resolved,
        "steps": steps,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "wall_s": round(episode_wall_s, 3),
        "fail_class": fail_class,
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for k successes out of n. Returns [low, high] in [0,1]."""
    if n <= 0:
        return [0.0, 1.0]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def aggregate(model_id, suite, k, per_task_runs):
    """Build the summary.json dict from per-task run records."""
    resolved_total = 0
    total = 0
    per_task = []
    for task_id, runs in per_task_runs:
        resolved_count = sum(1 for r in runs if r["resolved"])
        resolved_total += resolved_count
        total += len(runs)
        per_task.append({
            "id": task_id,
            "resolve_frac": round(resolved_count / len(runs), 4) if runs else 0.0,
            "runs": runs,
        })
    resolve_rate = round(resolved_total / total, 4) if total else 0.0
    return {
        "model": model_id,
        "suite": suite,
        "k": k,
        "resolved": resolved_total,
        "total": total,
        "resolve_rate": resolve_rate,
        "wilson_ci95": wilson_ci(resolved_total, total),
        "per_task": per_task,
    }


# --------------------------------------------------------------------------- #
# Batch runners
# --------------------------------------------------------------------------- #

def _run_one(task, rep, model_cfg, out_dir, traj_dir, interpreter):
    """Run one (task, repeat) pair and write its trajectory. Returns the run record."""
    with tempfile.TemporaryDirectory(prefix=f"eval-{task['id']}-") as work_dir:
        work_dir = Path(work_dir)
        materialize_fixture(task["_fixture"], work_dir)
        episode_timeout = max(task["step_limit"] * 120, 600)
        ndjson, ep_fail, ep_wall = run_episode(task, model_cfg, work_dir, episode_timeout, interpreter)
        traj_path = traj_dir / f"{task['id']}.{rep}.ndjson"
        traj_path.write_text(ndjson)
        grader_exit, grader_fail = run_grader(task["_grade"], work_dir, task["timeout_s"])
        record = grade_outcome(task, ndjson, ep_fail, ep_wall, grader_exit, grader_fail)
    return task["id"], rep, record


def run_batch(suite_dir, models_path, model_id, k, parallel, out, env_file=DEFAULT_ENV_FILE):
    suite_dir = Path(suite_dir)
    suite_name = suite_dir.name
    tasks = load_suite(suite_dir)
    models_cfg = load_models(models_path)
    model_cfg = resolve_model(models_cfg, model_id, env_file=env_file)
    episode_python = resolve_episode_interpreter(env_file=env_file)

    if not check_reachable(model_cfg):
        raise SystemExit(
            f"endpoint unreachable for model {model_id!r} at {model_cfg['api_base']} "
            f"(GET /v1/models did not return 2xx)"
        )

    out_dir = Path(out)
    traj_dir = out_dir / "traj"
    traj_dir.mkdir(parents=True, exist_ok=True)

    units = [(task, rep) for task in tasks for rep in range(k)]
    per_task_runs = {task["id"]: [None] * k for task in tasks}

    def _work(unit):
        task, rep = unit
        try:
            tid, r, record = _run_one(task, rep, model_cfg, out_dir, traj_dir, episode_python)
            return tid, r, record
        except Exception as exc:  # one task's crash never aborts the batch
            record = {
                "resolved": False, "steps": 0, "tokens_in": 0, "tokens_out": 0,
                "wall_s": 0.0, "fail_class": "infra",
            }
            sys.stderr.write(f"[evals] task {task['id']} rep {rep} failed: {exc}\n")
            return task["id"], rep, record

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        for tid, rep, record in pool.map(_work, units):
            per_task_runs[tid][rep] = record

    ordered = [(tid, per_task_runs[tid]) for tid in [t["id"] for t in tasks]]
    summary = aggregate(model_id, suite_name, k, ordered)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _print_summary(summary)
    return 0


def run_null_check(suite_dir, parallel=1):
    suite_dir = Path(suite_dir)
    suite_name = suite_dir.name
    tasks = load_suite(suite_dir)

    def _grade_one(task):
        with tempfile.TemporaryDirectory(prefix=f"null-{task['id']}-") as work_dir:
            work_dir = Path(work_dir)
            materialize_fixture(task["_fixture"], work_dir)
            exit_code, _ = run_grader(task["_grade"], work_dir, task["timeout_s"])
        return task["id"], exit_code

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(parallel, 1)) as pool:
        for tid, exit_code in pool.map(_grade_one, tasks):
            results[tid] = exit_code

    resolved_any = False
    print(f"null-check [{suite_name}] (no model calls; pristine fixtures):")
    for tid in [t["id"] for t in tasks]:
        code = results[tid]
        status = "UNRESOLVED" if code != 0 else "RESOLVED-BUG"
        if code == 0:
            resolved_any = True
        print(f"  {tid}: grader exit={code} -> {status}")
    if resolved_any:
        print("null-check FAILED: a grader passed on a pristine fixture (no discriminative power)")
        return 1
    print("null-check OK: every task judged unresolved")
    return 0


def _print_summary(summary):
    print(
        f"{summary['model']} / {summary['suite']}: "
        f"{summary['resolved']}/{summary['total']} resolved "
        f"({summary['resolve_rate']:.1%}) "
        f"Wilson95% CI {summary['wilson_ci95'][0]:.1%}-{summary['wilson_ci95'][1]:.1%}"
    )
    for task in summary["per_task"]:
        print(f"  {task['id']}: {task['resolve_frac']:.1%}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None):
    parser = argparse.ArgumentParser(description="charlie-code eval runner")
    parser.add_argument("--suite", required=True, help="suite name, e.g. dev")
    parser.add_argument("--model", help="logical model id (glm52|kimi-k3)")
    parser.add_argument("--k", type=int, default=1, help="repeats per task")
    parser.add_argument("--parallel", type=int, default=4, help="concurrent episodes")
    parser.add_argument("--out", help="run output dir, e.g. runs/baseline-glm52")
    parser.add_argument("--null-check", action="store_true",
                        help="run graders against pristine fixtures; no model calls")
    args = parser.parse_args(argv)

    if args.null_check:
        suite_dir = EVALS_DIR / "suites" / args.suite
        return run_null_check(suite_dir, parallel=args.parallel)

    if not args.model:
        parser.error("--model is required unless --null-check is given")
    if not args.out:
        parser.error("--out is required when running episodes")

    suite_dir = EVALS_DIR / "suites" / args.suite
    models_path = EVALS_DIR / "models.yaml"
    return run_batch(suite_dir, models_path, args.model, args.k, args.parallel, args.out)


if __name__ == "__main__":
    sys.exit(main())
