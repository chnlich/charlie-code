import math
import random


def make_data():
    rng = random.Random(20260731)
    # True weights drawn from the RNG — not obvious from reading the code.
    w0_true = rng.gauss(0, 1)
    w1_true = rng.gauss(0, 1)
    w2_true = rng.gauss(0, 1)
    data = []
    for _ in range(100):
        x1 = rng.random()           # U(0, 1)
        x2 = rng.random() * 100     # U(0, 100) — very different scale
        noise = rng.gauss(0, 0.1)
        y = w0_true + w1_true * x1 + w2_true * x2 + noise
        data.append((x1, x2, y))
    return data


def mse_loss(w, data):
    w0, w1, w2 = w
    total = 0.0
    for x1, x2, y in data:
        pred = w0 + w1 * x1 + w2 * x2
        total += (pred - y) ** 2
    return total / len(data)


data = make_data()

# BUG: features are not normalized. Feature 2 has scale ~100 while feature 1
# has scale ~1, so the gradient for w2 is ~100x larger than for w1. With
# lr=0.001, the w2 update overshoots and the loss diverges.
lr = 0.001
epochs = 200

w = [0.0, 0.0, 0.0]
for epoch in range(epochs):
    gw = [0.0, 0.0, 0.0]
    for x1, x2, y in data:
        pred = w[0] + w[1] * x1 + w[2] * x2
        err = pred - y
        gw[0] += err
        gw[1] += err * x1
        gw[2] += err * x2
    n = len(data)
    w[0] -= lr * gw[0] / n
    w[1] -= lr * gw[1] / n
    w[2] -= lr * gw[2] / n

final_loss = mse_loss(w, data)
print("weights:%s,%s,%s" % (w[0], w[1], w[2]))
print("final_loss:%s" % final_loss)
