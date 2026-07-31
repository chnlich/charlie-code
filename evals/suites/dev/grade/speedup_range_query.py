"""Grader for speedup_range_query: correctness + measured speedup vs O(n*q) baseline.

Runs with cwd = the episode work dir, which contains the agent's solution.py.
Deterministic: fixed seed and fixed input size. The same machine runs both the
agent's version and the baseline back-to-back, so the timing ratio is stable
even when absolute times vary.

Timing uses warmup + repeated trials with a median statistic (not a single
shot) for stability. No external dependencies.

Exit 0 = solved; non-zero = not solved. Diagnostics go to stderr.
"""

import os
import random
import sys
import time
import traceback


def naive_range_sum(arr, queries):
    results = []
    for lo, hi in queries:
        s = 0
        for i in range(lo, hi + 1):
            s += arr[i]
        results.append(s)
    return results


REQUIRED_RATIO = 10.0
N = 10000
Q = 10000


def fail(msg):
    sys.stderr.write("FAIL: " + msg + "\n")
    return 1


def _time_median(fn, *args, warmup=1, trials=5):
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - start)
    times.sort()
    mid = len(times) // 2
    if len(times) % 2 == 0:
        return (times[mid - 1] + times[mid]) / 2
    return times[mid]


def main():
    sys.path.insert(0, os.getcwd())
    try:
        import solution
    except Exception:
        return fail("could not import solution:\n" + traceback.format_exc())
    if not hasattr(solution, "range_sum"):
        return fail("solution has no range_sum attribute")

    rng = random.Random(20260731)
    arr = [rng.randint(-1000, 1000) for _ in range(N)]
    queries = [(rng.randint(0, N // 2), rng.randint(N // 2, N - 1)) for _ in range(Q)]
    expected = naive_range_sum(arr, queries)

    small_arr = [3, 1, 4, 1, 5, 9, 2, 6]
    small_q = [(0, 2), (3, 5), (0, 7)]
    if solution.range_sum(small_arr, small_q) != [8, 15, 31]:
        return fail("small-case mismatch: got %r expected [8, 15, 31]"
                    % solution.range_sum(small_arr, small_q))

    try:
        got = solution.range_sum(arr, queries)
    except Exception:
        return fail("range_sum raised on benchmark:\n" + traceback.format_exc())
    if got != expected:
        return fail("benchmark correctness mismatch: got %r expected %r" % (got, expected))

    t_naive = _time_median(naive_range_sum, arr, queries)
    t_agent = _time_median(solution.range_sum, arr, queries)
    ratio = t_naive / t_agent if t_agent > 0 else float("inf")
    sys.stderr.write("naive=%.4fs agent=%.4fs ratio=%.1f\n" % (t_naive, t_agent, ratio))
    if ratio < REQUIRED_RATIO:
        return fail("speedup %.1fx below required %.0fx" % (ratio, REQUIRED_RATIO))

    return 0


if __name__ == "__main__":
    sys.exit(main())
