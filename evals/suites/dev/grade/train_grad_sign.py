"""Grader for train_grad_sign: gradient sign must be correct and loss must converge.

Runs with cwd = the episode work dir, which contains the agent's (possibly
fixed) model.py and train.py. Deterministic: no randomness. No external
dependencies. Exit 0 = solved; non-zero = not solved.

The grader checks the gradient function directly (so hardcoding the weight
or loss in train.py cannot pass) and also runs train.py end-to-end.
"""

import math
import os
import subprocess
import sys
import traceback


def fail(msg):
    sys.stderr.write("FAIL: " + msg + "\n")
    return 1


def _reference_gradient(w, x, y):
    n = len(x)
    total = 0.0
    for i in range(n):
        total += (w * x[i] - y[i]) * x[i]
    return 2.0 * total / n


def main():
    sys.path.insert(0, os.getcwd())
    sys.modules.pop("model", None)
    try:
        import model
    except Exception:
        return fail("could not import model:\n" + traceback.format_exc())
    if not hasattr(model, "gradient"):
        return fail("model has no gradient attribute")

    for w_val, xv, yv in [
        (0.0, [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]),
        (1.0, [1.0, 2.0], [3.0, 5.0]),
        (2.0, [1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0]),
    ]:
        expected = _reference_gradient(w_val, xv, yv)
        try:
            got = model.gradient(w_val, xv, yv)
        except Exception:
            return fail("gradient raised:\n" + traceback.format_exc())
        if not isinstance(got, (int, float)) or not math.isfinite(got):
            return fail("gradient not finite for w=%r: %r" % (w_val, got))
        if abs(got - expected) > 1e-9:
            return fail("gradient wrong for w=%r x=%r y=%r: got %r expected %r"
                        % (w_val, xv, yv, got, expected))

    proc = subprocess.run(
        [sys.executable, "train.py"], capture_output=True, text=True,
        cwd=os.getcwd(),
    )
    if proc.returncode != 0:
        return fail("train.py exited %d:\n%s" % (proc.returncode, proc.stderr))

    final_loss = None
    weights = None
    for line in proc.stdout.splitlines():
        if line.startswith("final_loss:"):
            try:
                final_loss = float(line.split(":", 1)[1])
            except ValueError:
                final_loss = None
        elif line.startswith("weights:"):
            try:
                weights = float(line.split(":", 1)[1])
            except ValueError:
                weights = None

    if final_loss is None or not math.isfinite(final_loss):
        return fail("train.py did not print a finite final_loss; stdout=%r" % proc.stdout)
    if final_loss >= 1e-4:
        return fail("final_loss %r not below 1e-4" % final_loss)
    if weights is None or not math.isfinite(weights):
        return fail("train.py did not print finite weights; stdout=%r" % proc.stdout)
    if abs(weights - 2.0) > 0.01:
        return fail("weight %r not close to 2.0" % weights)

    return 0


if __name__ == "__main__":
    sys.exit(main())
