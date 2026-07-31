def gradient(w, x, y):
    """Gradient of the MSE loss mean((w * x_i - y_i) ^ 2) w.r.t. w.

    w: scalar weight.
    x: list of input values.
    y: list of target values (same length as x).
    Returns the gradient as a float.

    BUG: the sign is flipped. The correct gradient is
    2 * mean((w * x_i - y_i) * x_i), but this returns
    2 * mean((y_i - w * x_i) * x_i), which is the negative. This makes
    gradient descent ascend the loss instead of descend it, so the weight
    diverges away from the optimum.
    """
    n = len(x)
    total = 0.0
    for i in range(n):
        total += (y[i] - w * x[i]) * x[i]
    return 2.0 * total / n
