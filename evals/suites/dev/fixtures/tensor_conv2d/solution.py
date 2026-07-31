def conv2d(image, kernel):
    """Valid-padded 2D convolution.

    image: (H, W) list of lists of numbers.
    kernel: (kh, kw) list of lists of numbers.
    Return the convolution result as a (H-kh+1, W-kw+1) list of lists of floats,
    where each output pixel is the sum of the element-wise product of the
    kernel with the corresponding image patch. Use nested loops; do not use
    numpy or any external library. Handle rectangular kernels and integer
    inputs.
    """
    raise NotImplementedError("implement conv2d with nested loops")
