"""Grader for tensor_layernorm: numerical correctness + stability.

Runs with cwd = the episode work dir, which contains the agent's solution.py.
Deterministic: fixed seed for the random cases. No external dependencies.

Exit 0 = solved; non-zero = not solved. Diagnostics go to stderr.
"""

import math
import os
import random
import sys
import traceback


def reference_layernorm(x, gamma, beta, eps=1e-5):
    out = []
    for row in x:
        d = len(row)
        mean = sum(row) / d
        var = sum((v - mean) ** 2 for v in row) / d
        std = math.sqrt(var + eps)
        out.append([gamma[i] * (row[i] - mean) / std + beta[i] for i in range(d)])
    return out


def approx_equal(x, y, tol=1e-5, rtol=1e-5):
    if len(x) != len(y):
        return False
    for ri, rj in zip(x, y):
        if len(ri) != len(rj):
            return False
        for vi, vj in zip(ri, rj):
            if not math.isfinite(vi) or not math.isfinite(vj):
                return False
            if abs(vi - vj) > tol + rtol * abs(vj):
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
    if not hasattr(solution, "layernorm"):
        return fail("solution has no layernorm attribute")

    known = solution.layernorm(
        [[1.0, 2.0, 3.0]],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    )
    if not approx_equal(known, reference_layernorm([[1.0, 2.0, 3.0]], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])):
        return fail("known case mismatch: got %r" % (known,))

    large = solution.layernorm(
        [[1e6, 2e6, 3e6], [-1e6, 0.0, 1e6]],
        [2.0, 3.0, 4.0],
        [1.0, 1.0, 1.0],
        eps=1e-5,
    )
    expected_large = reference_layernorm(
        [[1e6, 2e6, 3e6], [-1e6, 0.0, 1e6]],
        [2.0, 3.0, 4.0],
        [1.0, 1.0, 1.0],
        eps=1e-5,
    )
    for row in large:
        for v in row:
            if not math.isfinite(v):
                return fail("non-finite output on large-value case: %r" % (large,))
    if not approx_equal(large, expected_large, tol=1e-3, rtol=1e-5):
        return fail("large-value case mismatch: got %r expected %r" % (large, expected_large))

    rng = random.Random(20260731)
    for case_idx in range(8):
        batch = rng.randint(1, 5)
        dim = rng.randint(2, 6)
        x = [[rng.uniform(-10.0, 10.0) for _ in range(dim)] for _ in range(batch)]
        gamma = [rng.uniform(-2.0, 2.0) for _ in range(dim)]
        beta = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        try:
            got = solution.layernorm(x, gamma, beta)
        except Exception:
            return fail("layernorm raised on case %d:\n%s" % (case_idx, traceback.format_exc()))
        expected = reference_layernorm(x, gamma, beta)
        if not approx_equal(got, expected):
            return fail("random case %d mismatch: got %r expected %r" % (case_idx, got, expected))

    return 0


if __name__ == "__main__":
    sys.exit(main())
