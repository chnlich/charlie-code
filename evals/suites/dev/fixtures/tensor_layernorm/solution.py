def layernorm(x, gamma, beta, eps=1e-5):
    """Layer normalization over the last dimension.

    x: (batch, dim) list of lists of numbers.
    gamma: (dim,) per-feature scale.
    beta: (dim,) per-feature shift.
    eps: small constant for numerical stability.
    Return a (batch, dim) list of lists of floats where each row is normalized
    to zero mean and unit variance, then scaled by gamma and shifted by beta.
    Use plain Python; do not use numpy or any external library. Must be
    numerically stable for large input values.
    """
    raise NotImplementedError("implement layernorm with plain Python")
