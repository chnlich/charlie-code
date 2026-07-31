import random


def init_weights(in_dim, hidden, out_dim, seed=0):
    """Initialize weights for a 2-layer MLP.

    W1: (hidden, in_dim), b1: (hidden,)
    W2: (out_dim, hidden), b2: (out_dim,)
    """
    rng = random.Random(seed)
    W1 = [[rng.gauss(0, 1) for _ in range(in_dim)] for _ in range(hidden)]
    b1 = [rng.gauss(0, 1) for _ in range(hidden)]
    W2 = [[rng.gauss(0, 1) for _ in range(hidden)] for _ in range(out_dim)]
    b2 = [rng.gauss(0, 1) for _ in range(out_dim)]
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward(weights, x):
    """Forward pass through a 2-layer MLP with ReLU activation.

    z1 = W1 @ x + b1, h1 = relu(z1), z2 = W2 @ h1 + b2.
    Returns z2 as a list of floats.
    """
    W1, b1 = weights["W1"], weights["b1"]
    W2, b2 = weights["W2"], weights["b2"]
    hidden = len(W1)
    out_dim = len(b2)
    z1 = [sum(W1[i][j] * x[j] for j in range(len(x))) + b1[i] for i in range(hidden)]
    h1 = [max(0.0, z) for z in z1]
    # BUG: W2[i][k] instead of W2[k][i] — transposed index order.
    # With hidden=3 > out_dim=2, W2[i] for i=2 is out of bounds -> IndexError.
    z2 = [sum(W2[i][k] * h1[i] for i in range(hidden)) + b2[k] for k in range(out_dim)]
    return z2
