# charlie-code

A minimal coding-agent prototype, modeled on
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)'s ~100-line core.

- **Bash-only actions.** The agent acts purely by emitting shell commands — no
  function-calling / tool APIs.
- **No Docker.** Commands run as local subprocesses on the host.
- **Linear history.** A flat `[system, user, assistant, user, ...]` message list; each
  step parses exactly one fenced ```` ```bash ```` block from the model, runs it, and
  feeds the combined stdout/stderr + exit code back as the next observation.

## Layout

```
main.py               # thin CLI entry: arg parsing + orchestration only
src/agent.py          # core linear-history loop
src/model.py          # litellm wrapper + step/usage tracking
src/environment.py    # local subprocess executor (fresh process per command)
src/config/default.yaml  # system/instance templates, response format, limits
tests/test_smoke.py   # loop + bash-parsing smoke test (model mocked, no network)
```

`main.py` (repo root) and the flat modules under `src/` are all exposed as top-level
modules by the build backend, so after install `import agent` / `import model` /
`import environment` all resolve.

## Install

```bash
pip install -e .          # runtime: litellm, pyyaml, typer
pip install -e ".[dev]"   # also installs pytest for the smoke test
```

## Run

```bash
charlie-code "<task>" [--model M] [--api-base URL] [--cwd DIR] [--steps N] [--wall-seconds N]
```

- `--cwd` is the repo the agent operates in (default: current directory).
- `--steps` is the hard step limit (default: 40). Exceeding it fails loudly.
- `--wall-seconds` is the hard episode wall-clock budget in seconds (default: 3600).
  Exceeding it fails loudly, checked at step boundaries, right after every model
  call, and between individual tool calls within a step.
- Each run gets a session id and writes message history to
  `~/.charlie-code/sessions/<session_id>.json` by default.
- Use `--resume <session_id>` to append a new task to an existing session history.
- Use `--session-dir DIR` to override the session store directory.

Example:

```bash
charlie-code "create a file hello.txt containing hi, then finish" --cwd /tmp/demo
```

The full trajectory (thought / command / observation per step) is printed to stdout,
followed by a summary line with step count and token usage.

### How a run ends

The agent drives the endpoint's native tool calling: it offers exactly one tool,
`bash`, and reads the response envelope rather than parsing the model's prose.

- **Completion is a whitelist.** The run ends only when a reply satisfies all three:
  `finish_reason` is `stop`, it carries no tool calls, and its text is non-empty.
  That text is the final answer. Any other combination continues or fails.
- **Tool calls.** A reply carrying tool calls runs all of them in the order given,
  one tool result fed back per call, then the loop continues.
- **Truncation.** `finish_reason: length` raises immediately. A reply cut off
  mid-answer is shape-identical to a finished one, so only the envelope can tell
  them apart, and guessing would silently accept a half-finished answer.
- **Empty reply.** No tool call and no text ends nothing; a short reminder is
  appended and the loop continues.
- **Step limit.** The loop raises after `--steps` steps (default 40) — it fails loud
  rather than silently stopping.
- **Withheld output.** Command output containing a model's own structure markers is
  not fed back. The serving stack parses generated text back into tool calls, so a
  marker that reaches the transcript can be echoed by the model and promoted from
  data into a real, executed call. The exit code still comes through, and the agent
  is told to re-read the content through a transform such as `base64`.
- **Resuming.** Session files are stamped with the protocol they were recorded under;
  a session from the older bash-block protocol is refused rather than replayed.

There is **no cost-based limit** — the SGLang model has no litellm pricing, so the only
budget is the step count.

### Unattended-run bounds

Every run is time-bounded end to end, for unattended use under a harness like
CharlieBot:

- **Command execution never blocks indefinitely.** Each command runs with its
  stdout/stderr redirected to a per-command log file under a run-scoped temp
  directory (not a pipe), with `stdin` closed so interactive commands see EOF
  immediately instead of hanging. A command returning inside `environment.timeout`
  (default 60s) has its own process group reaped right away, which also cleans up
  any `cmd &` survivors it spawned.
- **A command still running at the timeout is demoted, not killed.** The
  observation reports it is still running, its pid, and its log file's absolute
  path, with neutral guidance: poll it, keep working and check back later, or kill
  it yourself — all equally fine. Demoted jobs are tracked for the rest of the run
  and SIGKILLed when it ends (success, step limit, wall-clock limit, or error).
- **Escape hatch.** A command that daemonizes itself with `setsid` (a new session,
  hence a different process group) leaves harness jurisdiction by design — that is
  the supported way to start a real background service meant to outlive the run.
- **The model call has its own timeout and no retries.** `model.model_timeout`
  (default 300s) is passed straight to litellm, along with `num_retries=0` —
  litellm's OpenAI-compatible handler otherwise retries internally by default,
  tripling the worst-case cost of a stalled endpoint.
- **The episode has a wall-clock budget**, `--wall-seconds` / `agent.wall_seconds`
  (default 3600s), so the whole run is bounded by roughly
  `wall_seconds + one command budget + one model-call budget` in the worst case.
- **Log lifecycle.** A run's log directory is deleted when it completes normally;
  on any non-zero exit it is kept and its path is printed for forensics.

## Model / endpoint

Defaults (from `src/config/default.yaml`) target **your-model** served via an
OpenAI-compatible SGLang endpoint, accessed through litellm:

| setting       | default                                            |
| ------------- | --------------------------------------------------- |
| model         | `openai/your-org/your-model`                        |
| api_base      | `https://YOUR_SGLANG_HOST/v1`                       |
| model_timeout | `300` (seconds; hard litellm call timeout, no retries) |

Override precedence is **CLI flag > environment variable > YAML default**:

- model: `--model` / `CHARLIE_CODE_MODEL`
- api base: `--api-base` / `CHARLIE_CODE_API_BASE`
- session dir: `--session-dir` / `CHARLIE_CODE_SESSION_DIR`
- api key: `CHARLIE_CODE_API_KEY` (default `"EMPTY"` — the SGLang server does not
  require a key, so a placeholder is sent).

your-model returns its chain-of-thought in a separate `reasoning_content` field. We use
**only** the main message `content` for action parsing and ignore `reasoning_content`.

## Tests

```bash
pytest tests/
```

The smoke test exercises the full loop and bash-block parsing with `model.query`
monkeypatched to return canned responses (including the completion sentinel). It never
touches the network or the SGLang server.
## Manual live run

To try a real run against the endpoint (requires the SGLang server to be
reachable):

```bash
# check reachability first
curl -sf https://YOUR_SGLANG_HOST/v1/models

mkdir -p /tmp/cc_demo
charlie-code "create a file hello.txt containing hi, then finish" --cwd /tmp/cc_demo --steps 10
cat /tmp/cc_demo/hello.txt   # -> hi
```

## Evaluation harness (`evals/`)

`evals/` is a plain-scripts directory (not an installable package) that measures
charlie-code's task resolve rate. It contains a model registry, a batch runner,
a report generator, the improve-loop goal file, and per-task suites.

### Model endpoints

No endpoint values live in the repo. The runner resolves them from the
environment with precedence **process env > `~/.charlie-code/evals.env` > hard
failure**. Set these before running (the key is optional and defaults to
`EMPTY`):

| variable                | meaning                                   |
| ----------------------- | ----------------------------------------- |
| `CC_EVAL_GLM52_MODEL`   | litellm model id for the GLM-5.2 endpoint  |
| `CC_EVAL_GLM52_BASE`    | OpenAI-compatible base URL (`.../v1`)      |
| `CC_EVAL_KIMI_K3_MODEL` | litellm model id for the Kimi-K3 endpoint |
| `CC_EVAL_KIMI_K3_BASE`  | OpenAI-compatible base URL (`.../v1`)      |
| `CC_EVAL_API_KEY`       | shared api key (optional; default `EMPTY`) |

`evals/models.yaml` maps the logical ids `glm52` and `kimi-k3` to these
variable names.

`CC_EVAL_PYTHON` (optional) overrides the interpreter used to run episode
subprocesses, falling back to the runner's own interpreter when unset (same
env precedence chain as the endpoint vars).

### Commands

```bash
# null-check: run every grader against a pristine fixture, no model calls.
# Exits 0 only if every task is judged unresolved.
python evals/run.py --suite dev --null-check

# baseline / iteration run: one episode per task, then grade.
python evals/run.py --suite dev --model glm52 [--k 1] [--parallel 4] --out runs/<id>

# render a self-contained HTML report from one or more summaries.
python evals/report.py runs/<id>/summary.json [-o report.html]
python evals/report.py runs/a/summary.json runs/b/summary.json -o runs/delta.html
```

### Outputs

- `runs/<id>/summary.json` — schema: `{model, suite, k, resolved, total,
  resolve_rate, wilson_ci95, per_task: [{id, resolve_frac, runs: [{resolved,
  steps, tokens_in, tokens_out, wall_s, fail_class}]}]}`. `fail_class` is one of
  `step_limit` / `env_error` / `wrong_answer` / `infra` (null when resolved).
- `runs/<id>/traj/<task_id>.<rep>.ndjson` — the raw NDJSON event stream per
  (task, repeat) episode. Steps and tokens are parsed from the `result` event
  only.

`runs/` is gitignored. The improve-loop goal file at `evals/loop/goal.md` drives
the `charliebot improve` cycle against this harness.
