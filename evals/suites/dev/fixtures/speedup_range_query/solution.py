def range_sum(arr, queries):
    """Return a list of sums for each (lo, hi) query (inclusive indices).

    This implementation is correct but O(n * q): for each query it loops
    over the range and sums element by element. Rewrite it so it is correct
    AND substantially faster (at least 10x on a 10000-element array with
    10000 queries), keeping the same function name and signature.
    """
    results = []
    for lo, hi in queries:
        s = 0
        for i in range(lo, hi + 1):
            s += arr[i]
        results.append(s)
    return results
