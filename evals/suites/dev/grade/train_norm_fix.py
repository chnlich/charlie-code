"""Grader for train_norm_fix: MSE must drop below 0.5, verified independently.

Runs with cwd = the episode work dir, which contains the agent's (possibly
fixed) train.py. Deterministic: the dataset is regenerated with the same
fixed seed. No external dependencies.

The grader independently regenerates the dataset (same seed, same hidden
true weights), parses the printed weights, recomputes the MSE on the
original (un-normalized) data, and checks that (a) the printed loss matches
the recomputed loss and (b) the recomputed loss is below 0.5. This prevents
hardcoding a fake loss or printing weights in the wrong (normalized) space.

Exit 0 = solved; non-zero = not solved. Diagnostics go to stderr.
"""

import math
import os
import random
import subprocess
import sys


THRESHOLD = 0.5


def fail(msg):
    sys.stderr.write("FAIL: " + msg + "\n")
    return 1


def _make_data():
    rng = random.Random(20260731)
    w0_true = rng.gauss(0, 1)
    w1_true = rng.gauss(0, 1)
    w2_true = rng.gauss(0, 1)
    data = []
    for _ in range(100):
        x1 = rng.random()
        x2 = rng.random() * 100
        noise = rng.gauss(0, 0.1)
        y = w0_true + w1_true * x1 + w2_true * x2 + noise
        data.append((x1, x2, y))
    return data


def _mse_loss(w, data):
    w0, w1, w2 = w
    total = 0.0
    for x1, x2, y in data:
        pred = w0 + w1 * x1 + w2 * x2
        total += (pred - y) ** 2
    return total / len(data)


def main():
    proc = subprocess.run(
        [sys.executable, "train.py"], capture_output=True, text=True,
        cwd=os.getcwd(),
    )
    if proc.returncode != 0:
        return fail("train.py exited %d:\n%s" % (proc.returncode, proc.stderr))

    final_loss = None
    weights = None
    for line in proc.stdout.splitlines():
        if line.startswith("weights:"):
            parts = line.split(":", 1)[1].strip()
            try:
                weights = [float(v) for v in parts.split(",")]
            except ValueError:
                weights = None
        elif line.startswith("final_loss:"):
            try:
                final_loss = float(line.split(":", 1)[1])
            except ValueError:
                final_loss = None

    if weights is None or len(weights) != 3:
        return fail("could not parse 3 weights from output: %r" % proc.stdout)
    if final_loss is None or not math.isfinite(final_loss):
        return fail("train.py did not print a finite final_loss; stdout=%r" % proc.stdout)

    data = _make_data()
    recomputed = _mse_loss(weights, data)
    if not math.isfinite(recomputed):
        return fail("recomputed MSE is non-finite for weights %r" % weights)

    if abs(recomputed - final_loss) > 0.01:
        return fail("printed loss %r does not match recomputed MSE %r" % (final_loss, recomputed))
    if recomputed >= THRESHOLD:
        return fail("MSE %r not below threshold %r" % (recomputed, THRESHOLD))

    return 0


if __name__ == "__main__":
    sys.exit(main())
