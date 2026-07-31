"""Grader for nan_forensics: the loss must be finite and train.py must run clean.

Runs with cwd = the episode work dir, which contains the agent's (possibly
fixed) loss.py and train.py. Deterministic: no randomness. No external
dependencies. Exit 0 = solved; non-zero = not solved.
"""

import math
import os
import subprocess
import sys
import traceback


def fail(msg):
    sys.stderr.write("FAIL: " + msg + "\n")
    return 1


def _reference_ce(logits, target):
    m = max(logits)
    denom = sum(math.exp(z - m) for z in logits)
    p_true = math.exp(logits[target] - m) / denom
    return -math.log(p_true)


def main():
    sys.path.insert(0, os.getcwd())
    # drop any stale imported module so the agent's edited version is loaded
    sys.modules.pop("loss", None)
    try:
        import loss
    except Exception:
        return fail("could not import loss:\n" + traceback.format_exc())
    if not hasattr(loss, "cross_entropy"):
        return fail("loss has no cross_entropy attribute")

    try:
        val = loss.cross_entropy([400.0, 1000.0, 600.0], 0)
    except Exception:
        return fail("cross_entropy raised on the degenerate input:\n" + traceback.format_exc())
    if not isinstance(val, (int, float)) or not math.isfinite(val):
        return fail("cross_entropy not finite on degenerate input: %r" % (val,))

    # a normal input must match an independent reference to a tight tolerance,
    # so a degenerate constant-return "fix" cannot pass.
    normal = [1.0, 2.0, 3.0]
    expected = _reference_ce(normal, 2)
    try:
        got = loss.cross_entropy(normal, 2)
    except Exception:
        return fail("cross_entropy raised on the normal input:\n" + traceback.format_exc())
    if not math.isfinite(got) or abs(got - expected) > 1e-6:
        return fail("normal case mismatch: got %r expected %r" % (got, expected))

    # train.py must run to completion and print a finite final_loss
    proc = subprocess.run(
        [sys.executable, "train.py"], capture_output=True, text=True,
        cwd=os.getcwd(),
    )
    if proc.returncode != 0:
        return fail("train.py exited %d:\n%s" % (proc.returncode, proc.stderr))
    final = None
    for line in proc.stdout.splitlines():
        if line.startswith("final_loss="):
            try:
                final = float(line.split("=", 1)[1])
            except ValueError:
                final = None
    if final is None or not math.isfinite(final):
        return fail("train.py did not print a finite final_loss; stdout=%r" % (proc.stdout,))

    return 0


if __name__ == "__main__":
    sys.exit(main())
