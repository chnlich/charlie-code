"""Grader for train_lr_attain: loss must drop below 0.3, verified independently.

Runs with cwd = the episode work dir, which contains the agent's (possibly
tuned) train.py. Deterministic: the dataset is regenerated with the same
fixed seed. No external dependencies.

The grader independently regenerates the dataset, parses the printed weights,
recomputes the logistic loss, and checks that (a) the printed loss matches the
recomputed loss and (b) the recomputed loss is below 0.3. This prevents
hardcoding a fake final_loss or printing wrong weights.

Exit 0 = solved; non-zero = not solved. Diagnostics go to stderr.
"""

import math
import os
import random
import subprocess
import sys


THRESHOLD = 0.3


def fail(msg):
    sys.stderr.write("FAIL: " + msg + "\n")
    return 1


def _make_data():
    rng = random.Random(20260731)
    data = []
    for _ in range(50):
        x1 = rng.gauss(-2, 1)
        x2 = rng.gauss(-1, 1)
        data.append((x1, x2, 0))
    for _ in range(50):
        x1 = rng.gauss(2, 1)
        x2 = rng.gauss(1, 1)
        data.append((x1, x2, 1))
    return data


def _logistic_loss(weights, data):
    w0, w1, w2 = weights
    total = 0.0
    for x1, x2, y in data:
        z = w0 + w1 * x1 + w2 * x2
        total += max(z, 0) - z * y + math.log(1 + math.exp(-abs(z)))
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
        return fail("could not parse weights from output: %r" % proc.stdout)
    if final_loss is None or not math.isfinite(final_loss):
        return fail("train.py did not print a finite final_loss; stdout=%r" % proc.stdout)

    data = _make_data()
    recomputed = _logistic_loss(weights, data)
    if not math.isfinite(recomputed):
        return fail("recomputed loss is non-finite for weights %r" % weights)

    if abs(recomputed - final_loss) > 0.01:
        return fail("printed loss %r does not match recomputed loss %r" % (final_loss, recomputed))
    if recomputed >= THRESHOLD:
        return fail("loss %r not below threshold %r" % (recomputed, THRESHOLD))

    return 0


if __name__ == "__main__":
    sys.exit(main())
