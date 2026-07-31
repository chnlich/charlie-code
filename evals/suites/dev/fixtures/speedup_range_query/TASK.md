# Task: range_sum speedup

`solution.py` contains a correct but slow `range_sum(arr, queries)`.

- `arr` is a list of numbers.
- `queries` is a list of `(lo, hi)` tuples (inclusive indices).
- The function returns a list of the same length as `queries`, where each
  element is the sum of `arr[lo..hi]` (inclusive).

The current implementation is O(n * q): for each query it loops over the
range and sums element by element.

Rewrite it so it is:
1. still correct (returns the correct range sums), and
2. at least 10x faster than the O(n * q) baseline on a 10000-element array
   with 10000 queries.

Keep the function name `range_sum` and the two-argument signature
`(arr, queries)`. Do not use any third-party library; the standard library
is allowed.

The grader checks correctness on small and benchmark inputs, then times
both the baseline and your implementation with warmup + multiple trials and
a median statistic. Fast-and-wrong fails.
