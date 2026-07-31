def count_unique(values):
    """Return the number of distinct values in the list.

    This implementation is correct but O(n^2): for each value it scans every
    earlier value to decide whether it is new. Rewrite it so it is correct AND
    substantially faster (at least 10x on a 10000-element benchmark), keeping
    the same function name and signature.
    """
    count = 0
    seen_so_far = []
    for v in values:
        if v not in seen_so_far:
            seen_so_far.append(v)
            count += 1
    return count
