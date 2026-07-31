"""Grader for fault_shape_mismatch: forward must not crash and must be correct.

Runs with cwd = the episode work dir, which contains the agent's (possibly
fixed) model.py and run.py. Deterministic: fixed seed. No external
dependencies.

The grader checks the forward output against an independent reference
implementation (using the same weights from init_weights), verifies outputs
differ across different inputs, and runs run.py to completion.

Exit 0 = solved; non-zero = not solved. Diagnostics go to stderr.
"""

import math
import os
import subprocess
import sys
import traceback


def fail(msg):
    sys.stderr.write("FAIL: " + msg + "\n")
    return 1


def _reference_forward(weights, x):
    W1, b1 = weights["W1"], weights["b1"]
    W2, b2 = weights["W2"], weights["b2"]
    hidden = len(W1)
    out_dim = len(b2)
    z1 = [sum(W1[i][j] * x[j] for j in range(len(x))) + b1[i] for i in range(hidden)]
    h1 = [max(0.0, z) for z in z1]
    z2 = [sum(W2[k][i] * h1[i] for i in range(hidden)) + b2[k] for k in range(out_dim)]
    return z2


def main():
    sys.path.insert(0, os.getcwd())
    sys.modules.pop("model", None)
    try:
        import model
    except Exception:
        return fail("could not import model:\n" + traceback.format_exc())
    if not hasattr(model, "init_weights") or not hasattr(model, "forward"):
        return fail("model missing init_weights or forward")

    weights = model.init_weights(4, 3, 2, seed=42)
    x1 = [1.0, 2.0, 3.0, 4.0]
    try:
        got1 = model.forward(weights, x1)
    except Exception:
        return fail("forward raised on test input:\n" + traceback.format_exc())

    expected1 = _reference_forward(weights, x1)
    if len(got1) != len(expected1):
        return fail("output length mismatch: got %r expected %r" % (got1, expected1))
    for g, e in zip(got1, expected1):
        if not math.isfinite(g) or abs(g - e) > 1e-6:
            return fail("output mismatch: got %r expected %r" % (got1, expected1))

    x2 = [0.5, -1.0, 2.0, 0.0]
    try:
        got2 = model.forward(weights, x2)
    except Exception:
        return fail("forward raised on second input:\n" + traceback.format_exc())
    expected2 = _reference_forward(weights, x2)
    for g, e in zip(got2, expected2):
        if not math.isfinite(g) or abs(g - e) > 1e-6:
            return fail("second output mismatch: got %r expected %r" % (got2, expected2))

    if got1 == got2:
        return fail("outputs identical for different inputs (constant return?)")

    proc = subprocess.run(
        [sys.executable, "run.py"], capture_output=True, text=True,
        cwd=os.getcwd(),
    )
    if proc.returncode != 0:
        return fail("run.py exited %d:\n%s" % (proc.returncode, proc.stderr))

    return 0


if __name__ == "__main__":
    sys.exit(main())
