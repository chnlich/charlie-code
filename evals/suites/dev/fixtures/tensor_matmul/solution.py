def matmul(a, b):
    """Matrix product of a (m x k) times b (k x n), returned as (m x n).

    a and b are lists of lists of numbers (ints or floats). Implement the product
    with a plain triple loop over rows of a, columns of b, and the shared k
    dimension. Do not use numpy or any external library. The result must be a
    list of lists of floats with shape (m, n); handle rectangular and square
    matrices.
    """
    raise NotImplementedError("implement matmul with a triple loop")
