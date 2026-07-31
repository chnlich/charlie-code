# Task: count_pairs speedup

`solution.py` contains a correct but slow `count_pairs(values, target)`.

The function counts the number of index pairs `(i, j)` with `i < j` such
that `values[i] + values[j] == target`. The current implementation is
O(n^2): it checks every pair with a double loop.

Rewrite it so it is:
1. still correct (returns the number of pairs summing to target), and
2. at least 10x faster than the O(n^2) baseline on a 10000-element benchmark.

Keep the function name `count_pairs` and the two-argument signature
`(values, target)`. Do not use any third-party library; the standard
library is allowed.

The grader checks correctness on small and benchmark inputs, then times
both the baseline and your implementation with warmup + multiple trials and
a median statistic. Fast-and-wrong fails.
