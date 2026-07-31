# Task: train_grad_sign (training-script fix)

A gradient descent script is diverging instead of converging.

- `train.py` trains a linear model `y = w * x` on five fixed data points
  where `y = 2 * x`, using gradient descent with `lr = 0.01` and 200 epochs.
- The gradient is provided by `model.gradient(w, x, y)` in `model.py`.
- The gradient function has the **wrong sign**: it returns the negative of
  the true gradient, so the weight update ascends the loss instead of
  descending.

Fix `model.py` so that `gradient(w, x, y)` returns the correct gradient of
the MSE loss `mean((w * x_i - y_i) ^ 2)` with respect to `w`, which is
`2 * mean((w * x_i - y_i) * x_i)`.

Keep the same function name and signature. Do not change the data, the
learning rate, or the number of epochs in `train.py`.

When fixed, `python train.py` should print a `final_loss` below `1e-4` and
a weight close to `2.0`.
