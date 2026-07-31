# Task: conv2d (tensor-op correctness)

Implement `conv2d(image, kernel)` in `solution.py`.

- `image` is a list of lists of numbers with shape (H, W).
- `kernel` is a list of lists of numbers with shape (kh, kw) where kh <= H
  and kw <= W.
- Return the valid-padded convolution: a list of lists of floats with shape
  (H - kh + 1, W - kw + 1). Each output pixel is the sum of the element-wise
  product of the kernel and the corresponding image patch.
- Use nested loops. Do not import numpy or any third-party library.
- Must work for square and rectangular kernels, and for integer inputs.

The grader checks a known case, a rectangular-kernel case, and several
fixed-seed random cases against an independent reference, to numerical
tolerance.
