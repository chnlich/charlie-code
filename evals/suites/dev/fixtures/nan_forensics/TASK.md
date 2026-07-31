# Task: fault localization (numerical instability)

A training run is crashing instead of producing a finite loss.

- `train.py` calls `cross_entropy(logits, target)` from `loss.py` on logits
  `[400.0, 1000.0, 600.0]` with `target = 0`.
- Running `python train.py` currently raises an exception (a numerical
  overflow) instead of printing `final_loss=<finite number>`.

Investigate `loss.py`, find the root cause of the instability, and fix it so
that:

1. `python train.py` runs to completion and prints a finite `final_loss`.
2. `cross_entropy([400.0, 1000.0, 600.0], 0)` returns a finite number.

Fix the numerical guard in the loss computation; do not work around it by
removing data or changing the inputs in `train.py`.
