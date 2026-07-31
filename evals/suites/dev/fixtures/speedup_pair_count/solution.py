def count_pairs(values, target):
    """Return the number of index pairs (i, j) with i < j such that
    values[i] + values[j] == target.

    This implementation is correct but O(n^2): it checks every pair with a
    double loop. Rewrite it so it is correct AND substantially faster (at
    least 10x on a 10000-element benchmark), keeping the same function name
    and signature.
    """
    count = 0
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            if values[i] + values[j] == target:
                count += 1
    return count
