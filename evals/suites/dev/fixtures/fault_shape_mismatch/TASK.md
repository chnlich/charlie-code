# Task: fault_shape_mismatch (fault localization)

A 2-layer MLP forward pass crashes with an `IndexError`.

- `model.py` defines `init_weights(in_dim, hidden, out_dim, seed)` and
  `forward(weights, x)`.
- `init_weights` is correct: it produces `W1` (hidden x in_dim), `b1`
  (hidden), `W2` (out_dim x hidden), `b2` (out_dim).
- `forward` has a bug: it accesses `W2[i][k]` instead of `W2[k][i]` when
  computing the output layer, transposing the index order. Because the
  hidden dimension (3) is larger than the output dimension (2), this goes
  out of bounds and raises `IndexError`.
- `run.py` calls `init_weights(4, 3, 2, seed=42)` and `forward` on a fixed
  input; it currently crashes.

Find the bug in `model.py` and fix the index order in `forward` so that:

1. `python run.py` runs to completion and prints `output:...`.
2. `forward(weights, x)` produces the correct output for the given weights.

Do not change `init_weights` or the network architecture (2 layers, ReLU
activation on the hidden layer).

The grader checks the forward output against an independent reference
implementation (using the same weights from `init_weights`), verifies
outputs differ across different inputs, and runs `run.py` to completion.
