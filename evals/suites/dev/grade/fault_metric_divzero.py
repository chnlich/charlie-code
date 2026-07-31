"""Grader for fault_metric_divzero: metrics must not crash on edge cases.

Runs with cwd = the episode work dir, which contains the agent's (possibly
fixed) metrics.py and evaluate.py. Deterministic: no randomness. No
external dependencies.

The grader checks normal-case values against known references, edge-case
returns (must be 0.0, not crash), and runs evaluate.py to completion.

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


def _check(fn, args, expected, msg):
    try:
        got = fn(*args)
    except Exception:
        return fail(msg + " (raised):\n" + traceback.format_exc())
    if not isinstance(got, (int, float)) or not math.isfinite(got):
        return fail(msg + " (not finite): %r" % got)
    if abs(got - expected) > 1e-9:
        return fail(msg + ": got %r expected %r" % (got, expected))
    return 0


def main():
    sys.path.insert(0, os.getcwd())
    sys.modules.pop("metrics", None)
    try:
        import metrics
    except Exception:
        return fail("could not import metrics:\n" + traceback.format_exc())
    for attr in ("precision", "recall", "f1_score"):
        if not hasattr(metrics, attr):
            return fail("metrics has no %s attribute" % attr)

    # Normal cases
    rc = _check(metrics.precision, (10, 5, 3, 20), 10.0 / 15.0, "precision normal")
    if rc:
        return rc
    rc = _check(metrics.recall, (10, 5, 3, 20), 10.0 / 13.0, "recall normal")
    if rc:
        return rc
    p = 10.0 / 15.0
    r = 10.0 / 13.0
    rc = _check(metrics.f1_score, (10, 5, 3, 20), 2 * p * r / (p + r), "f1 normal")
    if rc:
        return rc

    # Edge case: no positive predictions (tp=0, fp=0)
    rc = _check(metrics.precision, (0, 0, 5, 20), 0.0, "precision edge (tp+fp=0)")
    if rc:
        return rc
    rc = _check(metrics.recall, (0, 0, 5, 20), 0.0, "recall edge (tp=0,fn>0)")
    if rc:
        return rc
    rc = _check(metrics.f1_score, (0, 0, 5, 20), 0.0, "f1 edge (tp+fp=0)")
    if rc:
        return rc

    # Edge case: no positive ground truth (tp=0, fn=0)
    rc = _check(metrics.precision, (0, 5, 0, 20), 0.0, "precision edge2 (tp=0,fp>0)")
    if rc:
        return rc
    rc = _check(metrics.recall, (0, 5, 0, 20), 0.0, "recall edge2 (tp+fn=0)")
    if rc:
        return rc
    rc = _check(metrics.f1_score, (0, 5, 0, 20), 0.0, "f1 edge2 (tp+fn=0)")
    if rc:
        return rc

    # evaluate.py must run to completion
    proc = subprocess.run(
        [sys.executable, "evaluate.py"], capture_output=True, text=True,
        cwd=os.getcwd(),
    )
    if proc.returncode != 0:
        return fail("evaluate.py exited %d:\n%s" % (proc.returncode, proc.stderr))

    return 0


if __name__ == "__main__":
    sys.exit(main())
