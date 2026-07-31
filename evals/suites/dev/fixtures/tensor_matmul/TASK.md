# Task: matmul (tensor-op correctness)

Implement `matmul(a, b)` in `solution.py`.

- Inputs `a` and `b` are lists of lists of numbers (`a` is m x k, `b` is k x n).
- Return their matrix product as a list of lists (m x n) of floats.
- Use a plain triple loop. Do not import numpy or any third-party library.
- Must work for square and rectangular matrices, and for integer inputs.

The grader checks a known case, a rectangular case, and several fixed-seed
random cases against an independent reference, to numerical tolerance.
