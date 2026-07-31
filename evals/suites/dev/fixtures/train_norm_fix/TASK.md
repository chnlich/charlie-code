# Task: train_norm_fix (training-script fix)

`train.py` runs linear regression via gradient descent on a 2-feature dataset
of 100 points. The features have very different scales: feature 1 is in
`[0, 1]` and feature 2 is in `[0, 100]`. The learning rate `0.001` is too
large for the wide-scale feature, so the gradient explodes and the loss
diverges to infinity.

Fix the **feature scaling** in `train.py` so that gradient descent converges
and the final MSE loss drops **below 0.5**.

Constraints:

- Keep the dataset generation code unchanged (same seed, same formula).
- The printed `weights:w0,w1,w2` and `final_loss:X` must be in the
  **original (un-normalized) feature space** — if you normalize features
  internally, convert the weights back before printing.
- Do not use numpy or any third-party library.

The grader independently regenerates the dataset, parses the printed weights,
recomputes the MSE on the original data, and checks that (a) the printed loss
matches the recomputed loss and (b) the recomputed loss is below 0.5.
