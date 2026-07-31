# Task: train_lr_attain (training-script attain)

`train.py` runs logistic regression (with bias) on a fixed 2-feature binary
classification dataset of 100 points. The dataset is generated with a fixed
seed inside `train.py`.

The current configuration uses `lr = 0.001` and `epochs = 50`, which is too
slow: the final loss stays around 0.64, well above the target.

Tune the **learning rate** and/or the **number of epochs** in `train.py` so
that the final loss drops **below 0.3**.

Constraints:

- Keep the dataset generation code unchanged (same seed, same distributions).
- Keep the loss function and gradient computation unchanged.
- The script must print `weights:w0,w1,w2` (comma-separated) and
  `final_loss:X` as the last two lines.
- Do not use numpy or any third-party library.

The grader independently regenerates the same dataset, parses the printed
weights, recomputes the logistic loss, and checks that (a) the printed loss
matches the recomputed loss and (b) the recomputed loss is below 0.3.
