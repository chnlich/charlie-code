import math


def cross_entropy(logits, target):
    """Softmax cross-entropy for a single example.

    `logits` is a list of floats; `target` is the index of the true class.
    Returns -log(softmax(logits)[target]).

    This implementation is numerically unstable: it exponentiates the raw
    logits directly, so a large logit overflows (math.exp raises OverflowError)
    and the run crashes. Find and fix the root cause so the loss is finite for
    the inputs in train.py.
    """
    denom = sum(math.exp(z) for z in logits)
    p_true = math.exp(logits[target]) / denom
    return -math.log(p_true)
