def precision(tp, fp, fn, tn):
    """Precision: tp / (tp + fp).

    BUG: no guard against tp + fp == 0. When there are no positive
    predictions, this divides by zero and crashes.
    """
    return tp / (tp + fp)


def recall(tp, fp, fn, tn):
    """Recall: tp / (tp + fn).

    BUG: no guard against tp + fn == 0. When there is no positive ground
    truth, this divides by zero and crashes.
    """
    return tp / (tp + fn)


def f1_score(tp, fp, fn, tn):
    """F1 score: harmonic mean of precision and recall.

    BUG: crashes when precision + recall == 0 (both are zero).
    """
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)
