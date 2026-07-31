"""Grader for speedup_dedup: correctness + measured speedup vs an O(n^2) baseline.

Runs with cwd = the episode work dir, which contains the agent's solution.py.
Deterministic: fixed seed and fixed input size. The same machine runs both the
agent's version and the baseline back-to-back, so the timing ratio is stable
even when absolute times vary. No external dependencies.

Exit 0 = solved; non-zero = not solved. Diagnostics go to stderr.
"""

import os
import random
import sys
import time
import traceback


def naive_count_unique(values):
    count = 0
    seen_so_far = []
    for v in values:
        if v not in seen_so_far:
            seen_so_far.append(v)
            count += 1
    return count


REQUIRED_RATIO = 10.0
N = 10000


def fail(msg):
    sys.stderr.write("FAIL: " + msg + "\n")
    return 1


def _time(fn, arg, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(arg)
        best = min(best, time.perf_counter() - start)
    return best


def main():
    sys.path.insert(0, os.getcwd())
    try:
        import solution
    except Exception:
        return fail("could not import solution:\n" + traceback.format_exc())
    if not hasattr(solution, "count_unique"):
        return fail("solution has no count_unique attribute")

    rng = random.Random(20260731)
    data = [rng.randint(0, 5000) for _ in range(N)]
    expected = len(set(data))

    # correctness on small + benchmark inputs
    small = [1, 1, 2, 3, 2, 3, 3, 4]
    if solution.count_unique(small) != len(set(small)):
        return fail("small-case mismatch: got %r expected %r"
                    % (solution.count_unique(small), len(set(small))))
    try:
        got = solution.count_unique(data)
    except Exception:
        return fail("count_unique raised on benchmark:\n" + traceback.format_exc())
    if got != expected:
        return fail("benchmark correctness mismatch: got %r expected %r" % (got, expected))

    # speedup: baseline and agent timed back-to-back on the same machine
    t_naive = _time(naive_count_unique, data)
    t_agent = _time(solution.count_unique, data)
    ratio = t_naive / t_agent if t_agent > 0 else float("inf")
    sys.stderr.write("naive=%.4fs agent=%.4fs ratio=%.1f\n" % (t_naive, t_agent, ratio))
    if ratio < REQUIRED_RATIO:
        return fail("speedup %.1fx below required %.0fx" % (ratio, REQUIRED_RATIO))

    return 0


if __name__ == "__main__":
    sys.exit(main())
