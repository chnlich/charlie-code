"""Grader for tensor_conv2d: numerical correctness of a pure-Python 2D conv.

Runs with cwd = the episode work dir, which contains the agent's solution.py.
Deterministic: fixed seed for the random cases. No external dependencies.

Exit 0 = solved; non-zero = not solved. Diagnostics go to stderr.
"""

import math
import os
import random
import sys
import traceback


def reference_conv2d(image, kernel):
    h = len(image)
    w = len(image[0])
    kh = len(kernel)
    kw = len(kernel[0])
    oh = h - kh + 1
    ow = w - kw + 1
    out = [[0.0] * ow for _ in range(oh)]
    for i in range(oh):
        for j in range(ow):
            s = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    s += image[i + ki][j + kj] * kernel[ki][kj]
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
    if not hasattr(solution, "conv2d"):
        return fail("solution has no conv2d attribute")

    known = solution.conv2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 0], [0, -1]])
    if not approx_equal(known, [[-4.0, -4.0], [-4.0, -4.0]]):
        return fail("known case mismatch: got %r" % (known,))

    rect = solution.conv2d(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[1, 0, 0], [0, 1, 0]],
    )
    if not approx_equal(rect, [[7.0, 9.0], [15.0, 17.0]]):
        return fail("rectangular-kernel case mismatch: got %r" % (rect,))

    rng = random.Random(20260731)
    for case_idx in range(8):
        h = rng.randint(3, 8)
        w = rng.randint(3, 8)
        kh = rng.randint(1, 3)
        kw = rng.randint(1, 3)
        image = [[rng.uniform(-1.0, 1.0) for _ in range(w)] for _ in range(h)]
        kernel = [[rng.uniform(-1.0, 1.0) for _ in range(kw)] for _ in range(kh)]
        try:
            got = solution.conv2d(image, kernel)
        except Exception:
            return fail("conv2d raised on case %d:\n%s" % (case_idx, traceback.format_exc()))
        expected = reference_conv2d(image, kernel)
        if not approx_equal(got, expected):
            return fail("random case %d mismatch: got %r expected %r" % (case_idx, got, expected))

    return 0


if __name__ == "__main__":
    sys.exit(main())
