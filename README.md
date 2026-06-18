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
charlie-code "<task>" [--model M] [--api-base URL] [--cwd DIR] [--steps N]
```

- `--cwd` is the repo the agent operates in (default: current directory).
- `--steps` is the hard step limit (default: 40). Exceeding it fails loudly.

Example:

```bash
charlie-code "create a file hello.txt containing hi, then finish" --cwd /tmp/demo
```

The full trajectory (thought / command / observation per step) is printed to stdout,
followed by a summary line with step count and token usage.

### How a run ends

- **Completion sentinel.** When the task is solved the model emits a final command
  `echo CHARLIE_CODE_COMPLETE`; the loop detects the sentinel and stops successfully.
  This marker is documented in the system template in `src/config/default.yaml`.
- **Step limit.** If the sentinel is never emitted, the loop raises after `--steps`
  steps (default 40) — it fails loud rather than silently stopping.
- **No bash block.** If the model's reply contains no bash block, a short
  format-reminder observation is appended and the loop continues (it does not crash).
- **Multiple bash blocks.** Only the first is executed; the observation notes this.

There is **no cost-based limit** — the SGLang model has no litellm pricing, so the only
budget is the step count.

## Model / endpoint

Defaults (from `src/config/default.yaml`) target **your-model** served via an
OpenAI-compatible SGLang endpoint, accessed through litellm:

| setting   | default                                                |
| --------- | ------------------------------------------------------ |
| model     | `openai/your-org/your-model`                           |
| api_base  | `https://YOUR_SGLANG_HOST/v1`     |

Override precedence is **CLI flag > environment variable > YAML default**:

- model: `--model` / `CHARLIE_CODE_MODEL`
- api base: `--api-base` / `CHARLIE_CODE_API_BASE`
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

To try a real run against the endpoint (requires the SGLang server to be reachable):

```bash
# check reachability first
curl -sf https://YOUR_SGLANG_HOST/v1/models

mkdir -p /tmp/cc_demo
charlie-code "create a file hello.txt containing hi, then finish" --cwd /tmp/cc_demo --steps 10
cat /tmp/cc_demo/hello.txt   # -> hi
```
