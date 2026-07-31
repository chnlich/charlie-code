"""Grader for speedup_pair_count: correctness + measured speedup vs O(n^2) baseline.

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


def naive_count_pairs(values, target):
    count = 0
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            if values[i] + values[j] == target:
                count += 1
    return count


REQUIRED_RATIO = 10.0
N = 10000


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
    if not hasattr(solution, "count_pairs"):
        return fail("solution has no count_pairs attribute")

    rng = random.Random(20260731)
    data = [rng.randint(0, 9999) for _ in range(N)]
    target = rng.randint(0, 9999)
    expected = naive_count_pairs(data, target)

    small = [1, 2, 3, 4, 5]
    if solution.count_pairs(small, 6) != 2:
        return fail("small-case mismatch: got %r expected 2"
                    % solution.count_pairs(small, 6))

    try:
        got = solution.count_pairs(data, target)
    except Exception:
        return fail("count_pairs raised on benchmark:\n" + traceback.format_exc())
    if got != expected:
        return fail("benchmark correctness mismatch: got %r expected %r" % (got, expected))

    t_naive = _time_median(naive_count_pairs, data, target)
    t_agent = _time_median(solution.count_pairs, data, target)
    ratio = t_naive / t_agent if t_agent > 0 else float("inf")
    sys.stderr.write("naive=%.4fs agent=%.4fs ratio=%.1f\n" % (t_naive, t_agent, ratio))
    if ratio < REQUIRED_RATIO:
        return fail("speedup %.1fx below required %.0fx" % (ratio, REQUIRED_RATIO))

    return 0


if __name__ == "__main__":
    sys.exit(main())
