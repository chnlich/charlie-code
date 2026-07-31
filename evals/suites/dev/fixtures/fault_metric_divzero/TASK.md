# Task: fault_metric_divzero (fault localization)

A metrics computation script crashes with `ZeroDivisionError` on edge cases.

- `metrics.py` defines `precision(tp, fp, fn, tn)`, `recall(tp, fp, fn, tn)`,
  and `f1_score(tp, fp, fn, tn)`.
- `precision = tp / (tp + fp)` crashes when `tp + fp == 0` (no positive
  predictions).
- `recall = tp / (tp + fn)` crashes when `tp + fn == 0` (no positive ground
  truth).
- `f1_score` crashes when `precision + recall == 0`.
- `evaluate.py` calls these functions on both normal and edge-case inputs;
  it currently crashes on the edge cases.

Fix `metrics.py` so that:

1. `precision`, `recall`, and `f1_score` return `0.0` when their denominator
   is zero (instead of crashing).
2. They return the correct values for normal cases.
3. `python evaluate.py` runs to completion without error.

Do not change `evaluate.py`.

The grader checks normal-case values against known references, edge-case
returns (must be 0.0, not crash), and runs `evaluate.py` to completion.
