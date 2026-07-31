"""Grader for tensor_matmul: numerical correctness of a pure-Python matmul.

Runs with cwd = the episode work dir, which contains the agent's solution.py.
Deterministic: fixed seed for the random cases. No external dependencies.

Exit 0 = solved; non-zero = not solved. Diagnostics go to stderr.
"""

import math
import os
import random
import sys
import traceback


def reference_matmul(a, b):
    m = len(a)
    k = len(a[0])
    n = len(b[0])
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += a[i][t] * b[t][j]
            out[i][j] = s
    return out


def approx_equal(x, y, tol=1e-6):
    if len(x) != len(y):
        return False
    for ri, rj in zip(x, y):
        if len(ri) != len(rj):
            return False
        for vi, vj in zip(ri, rj):
            if not math.isfinite(vi) or not math.isfinite(vj):
                return False
            if abs(vi - vj) > tol:
                return False
    return True


def fail(msg):
    sys.stderr.write("FAIL: " + msg + "\n")
    return 1


def main():
    sys.path.insert(0, os.getcwd())
    try:
        import solution
    except Exception:
        return fail("could not import solution:\n" + traceback.format_exc())
    if not hasattr(solution, "matmul"):
        return fail("solution has no matmul attribute")

    known = solution.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    if not approx_equal(known, [[19.0, 22.0], [43.0, 50.0]]):
        return fail("known case mismatch: got %r" % (known,))

    rect = solution.matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])
    if not approx_equal(rect, [[58.0, 64.0], [139.0, 154.0]]):
        return fail("rectangular case mismatch: got %r" % (rect,))

    rng = random.Random(20260731)
    for case_idx in range(8):
        m = rng.randint(1, 6)
        k = rng.randint(1, 6)
        n = rng.randint(1, 6)
        a = [[rng.uniform(-1.0, 1.0) for _ in range(k)] for _ in range(m)]
        b = [[rng.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(k)]
        try:
            got = solution.matmul(a, b)
        except Exception:
            return fail("matmul raised on case %d:\n%s" % (case_idx, traceback.format_exc()))
        expected = reference_matmul(a, b)
        if not approx_equal(got, expected):
            return fail("random case %d mismatch: got %r expected %r" % (case_idx, got, expected))

    return 0


if __name__ == "__main__":
    sys.exit(main())
