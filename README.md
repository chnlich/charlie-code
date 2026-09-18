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

A host that runs charlie-code from a non-editable `uv tool` install of this checkout
picks up a merge only after a reinstall: `scripts/reinstall-tool.sh` fast-forwards the
checkout to `origin/main`, reinstalls the tool through `uv tool upgrade` (keeping the
install's recorded dependency constraints) and asserts the installed copy is the new
code, failing loudly on any step.

## Run

```bash
charlie-code --task-file TASK.md [--image PATH ...] [--model M] [--api-base URL] [--cwd DIR] [--steps N] [--stream|--no-stream] [--timeout-seconds N]
```

- `--task-file PATH` (required) supplies the task text: the file's full UTF-8
  contents verbatim.
- `--image PATH` (repeatable, up to 4) attaches an image to the task message:
  each PATH must be a readable file whose suffix is one of `png`, `jpg`,
  `jpeg`, `gif`, `webp`, and at most 4 MiB. The task message then carries the
  rendered task text plus one base64 `image_url` part per image, in the order
  given; with no `--image` the message is the plain-text form. Images persist
  verbatim in the session history and resume unchanged.
- `--cwd` is the repo the agent operates in (default: current directory).
- `--steps` is the hard step limit (default: 1000). Exceeding it fails loudly.
- `--stream` / `--no-stream` selects how the model call is made (default: the
  configured `model.stream`, streaming). Under streaming the call stays alive as
  long as chunks keep arriving; under `--no-stream` the timeout becomes the
  whole-call budget instead of the inter-chunk silence bound.
- `--timeout-seconds N` is the model-call budget passed to litellm as `timeout`:
  the silence bound between streamed chunks when streaming, the whole-call bound
  otherwise (default: `model.timeout_seconds`, 1200).
- Each run gets a session id and writes message history to
  `~/.charlie-code/sessions/<session_id>.json` by default.
- Use `--resume <session_id>` to append a new task to an existing session history.
- Use `--session-dir DIR` to override the session store directory.

Example:

```bash
printf 'create a file hello.txt containing hi, then finish\n' > /tmp/demo-task.md
charlie-code --task-file /tmp/demo-task.md --cwd /tmp/demo
```

The full trajectory (thought / command / observation per step) is printed to stdout,
followed by a summary line with step count and token usage.

### AGENTS.md convention

On a **fresh** session start, charlie-code reads `AGENTS.md` from the working
directory (`--cwd`): when the file is present, its full text is appended after
the rendered system template in the system message. A missing or whitespace-only
file is silently skipped; a file that cannot be read or decoded as UTF-8
produces one stderr warning naming the file, and the session continues without
it. Resumed sessions replay the stored history and never re-read the file.

### Skills

On a **fresh** session start, charlie-code lists the skills it can find in the
system message: one line per skill with its `name` and `description`, followed by
the absolute path of its `SKILL.md`, so the model reads a skill's full text with
`cat` only when a task calls for it. Two levels feed the list:

- Host level: every `<root>/<name>/SKILL.md` under the configured roots, by default
  `~/.agents/skills` (the Agent Skills convention shared with Codex and Gemini CLI)
  and `~/.claude/skills` (Claude Code). `--skills-root DIR`, or the environment
  variable `CHARLIE_CODE_SKILLS_ROOT`, names a single directory that replaces the
  configured roots for that run.
- Repo level: `.claude/skills/` and `.agents/skills/` under the git worktree root
  that contains the working directory (`--cwd`), found as the first ancestor holding
  a `.git` entry, directory or file. A working directory outside any repository has
  no repo level.

Repo-level directories are scanned first, then the host roots; the first directory
that supplies a name wins, so a repo skill overrides a host skill of the same name,
and a skill linked under the same name into both host roots lists once. A
`SKILL.md` without frontmatter or without a `description` is skipped, as is a
missing directory. The list is computed once at startup; resumed sessions replay
their stored system message. The root and directory lists live under `skills:` in
`src/config/default.yaml`.

### How a run ends

The agent drives the endpoint's native tool calling: it offers exactly one tool,
`bash`, and reads the response envelope rather than parsing the model's prose.

- **Completion.** The run ends when a reply carries no tool calls and
  `finish_reason` is `stop`; its text is the final output, delivered as written.
  A reply with neither tool calls nor text raises.
- **Tool calls.** A reply carrying tool calls runs all of them in the order given,
  one tool result fed back per call, then the loop continues.
- **Truncation.** `finish_reason: length` raises immediately. A generation stopped
  by a sampled stop token reports `stop`, is shape-identical to a finished reply
  on the wire, and is delivered as written; the system prompt's control-marker
  rule keeps the model from spelling the markers that sample as stop tokens.
- **Step limit.** The loop raises after `--steps` steps (default 1000) — it fails loud
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

Runs are bounded by progress, not elapsed time, for unattended use under a
harness like CharlieBot:

- **Command execution never blocks indefinitely.** Each command runs with its
  stdout/stderr redirected to a per-command log file under the run's session log
  directory (not a pipe), with `stdin` closed so interactive commands see EOF
  immediately instead of hanging. The command runs in the foreground until it
  exits; a command returning on its own has its process group reaped right
  away, which also cleans up any `cmd &` survivors it spawned.
- **A running command reports progress.** At each `environment.progress_notices_seconds`
  tick (default 60 and 300 seconds) the `--json` stream carries a
  `command_progress` event with the command's id, pid and log path, which a
  harness renders as a chat note.
- **A command still running at `environment.kill_after_seconds` is terminated.**
  Its process group gets SIGTERM, then SIGKILL five seconds later if it is still
  there (default cap 900 seconds). The observation carries the output so far,
  the process's real exit code (-15 or -9), and a note naming the cap, the pid
  and the log path; a final `command_progress` event with `killed: true` marks
  the termination. The system prompt tells the model to run in the foreground
  only work expected within a minute, to start longer work with
  `background: true`, and to detach anything expected to outlast the cap.
- **A command can run in the background.** With `background: true` the `bash`
  call returns at once with the task id, pid and log path, and a supervising
  thread waits on the command under the same cap, reaps it the instant it exits,
  and records its exit code. The record, its output bounded like any observation,
  is appended to the next tool result that completes (at most four records per
  result; the rest follow in exit order), so the model gets it without polling;
  `timeout N tail --pid=PID -f LOG` waits for it explicitly and `kill PID` stops
  it. A reply that ends the run while a task is still running gets a reminder
  and the run continues; every exit path kills the remaining background groups,
  so no task outlives the run.
- **SIGTERM to charlie-code is a controlled exit.** The running command's process
  group and every background task's group are SIGKILLed, the session state is
  persisted, and the process exits with
  status 143, inside the five-second window a supervisor allows before it
  escalates to SIGKILL.
- **Escape hatch.** A command that daemonizes itself with `setsid` (a new session,
  hence a different process group) leaves harness jurisdiction by design — that is
  the supported way to start a real background service meant to outlive the run.
- **The model call is streamed by default.** `model.timeout_seconds` (default
  1200) is passed to litellm as `timeout`, which under streaming bounds the
  silence between chunks rather than the whole call: the call fails with
  `RuntimeError("model produced no output for ...s")` after that many seconds
  without a chunk, and a model that keeps producing is never cut off. With
  `--no-stream` the same number becomes the whole-call budget. Either way it is
  combined with `num_retries=0` (litellm's OpenAI-compatible handler otherwise
  retries internally by default, tripling the worst-case cost of a stalled
  endpoint). A run has no total-duration budget and ends only by task
  completion, the step limit, model silence or an endpoint error, or a
  truncated generation (`finish_reason=length`).
- **Log lifecycle.** A run's log directory is deleted when it completes normally;
  on any non-zero exit it is kept and its path is printed for forensics.

## Model / endpoint

Defaults (from `src/config/default.yaml`) target **your-model** served via an
OpenAI-compatible SGLang endpoint, accessed through litellm:

| setting       | default                                            |
| ------------- | --------------------------------------------------- |
| model         | `openai/your-org/your-model`                        |
| api_base      | `https://YOUR_SGLANG_HOST/v1`                       |
| stream        | `true` (stream the model call)                      |
| timeout_seconds | 1200 (silence bound between streamed chunks, whole-call bound when not; no retries) |

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
monkeypatched to return canned responses. It never touches the network or the
SGLang server.
## Manual live run

To try a real run against the endpoint (requires the SGLang server to be
reachable):

```bash
# check reachability first
curl -sf https://YOUR_SGLANG_HOST/v1/models

mkdir -p /tmp/cc_demo
printf 'create a file hello.txt containing hi, then finish\n' > /tmp/cc_demo-task.md
charlie-code --task-file /tmp/cc_demo-task.md --cwd /tmp/cc_demo --steps 10
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
