# Task: layernorm (tensor-op correctness)

Implement `layernorm(x, gamma, beta, eps=1e-5)` in `solution.py`.

- `x` is a list of lists of numbers with shape (batch, dim).
- `gamma` is a list of length dim (per-feature scale).
- `beta` is a list of length dim (per-feature shift).
- `eps` is a small float (default 1e-5) for numerical stability.
- Return a list of lists of floats with shape (batch, dim). For each row:
  1. Compute the mean over the dim features.
  2. Compute the variance over the dim features (divide by dim, not dim-1).
  3. Normalize: `(x_i - mean) / sqrt(var + eps)`.
  4. Scale and shift: `gamma_i * normalized_i + beta_i`.
- Use plain Python. Do not import numpy or any third-party library.
- Must be numerically stable for large input values (e.g. values up to 1e6).

The grader checks a known case, a large-value numerical-stability case, and
several fixed-seed random cases against an independent reference, to
numerical tolerance.
