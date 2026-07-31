Read skills: improve-worker

# Goal: raise charlie-code task resolve rate on the dev suite

You are optimizing the charlie-code agent (prompts, context, budget parameters).
The evaluation harness under `evals/` is a frozen ruler: it measures whether a
change is an improvement or a regression. Optimize the agent; do not touch the
ruler.

## Ordered goals

1. Raise the `dev` suite resolve rate (primary metric).
2. Shrink the most frequent failure class in the failure-class ledger (tie-breaker
   when goal 1 is flat).

Both are read off the summary.json produced by the acceptance command below. The
baseline magnitude is "the latest baseline report under `runs/`" — never a
hard-coded number. Re-read the latest `runs/*/summary.json` to get the current
baseline before each iteration.

## How to work

- Work through `evals/run.py` only. Do NOT rewrite the harness (`run.py`,
  `report.py`, `models.yaml`, `loop/goal.md`, the suite YAML, fixtures, or
  graders). The harness is the frozen control; changing it is an infraction.
- One variable per iteration: change one lever (a prompt template, a context
  addition, or a budget parameter in `src/`), then run the acceptance command.
- Real model episodes use the pinned interpreter. The environment variables
  `CC_EVAL_GLM52_*` / `CC_EVAL_KIMI_K3_*` / `CC_EVAL_API_KEY` supply the
  endpoints, and `CC_EVAL_PYTHON` supplies the episode interpreter; if any are
  missing, that is a blocker, not something to fix by editing the harness.

## Acceptance command sequence (run every iteration)

1. Freeze gate (must pass with zero output / clean exit):
   ```
   git diff --exit-code main -- evals/
   ```
   If the diff is non-empty, the iteration is an infrastructure infraction: discard
   the change and stop. The harness must stay byte-identical to `main`.

2. Eval run (GLM-5.2, single model, k=1):
   ```
   python evals/run.py --suite dev --model glm52 --out runs/iter-<n>
   ```
   Compare `runs/iter-<n>/summary.json` `resolved` against the current baseline.

## Retention rule (k=1)

Let `Delta` = (solved this iteration) - (solved at baseline).

- `Delta >= 3`: keep the change, commit, update the baseline to this run.
- `Delta in {1, 2}`: rerun only the flipped tasks once. Keep the change only if the
  flips reproduce; otherwise revert.
- `Delta <= 0`: revert the change (including `Delta == 0`).

Zero progress is an acceptable outcome across an iteration: it is not a failure,
and it does not license piling more changes onto the same iteration to force a
result. Whether an iteration is kept or reverted, record what was tried and what
`Delta` came out to (e.g. a short commit note), then move to the next iteration.

After the final iteration, rerun the accepted state on `kimi-k3` once to confirm
any gain is not single-model-specific, and render the comparison with
`python evals/report.py runs/baseline-glm52/summary.json runs/iter-<n>/summary.json -o runs/delta.html`.
