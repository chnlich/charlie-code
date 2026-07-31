import math
import random


def make_data():
    rng = random.Random(20260731)
    data = []
    for _ in range(50):
        x1 = rng.gauss(-2, 1)
        x2 = rng.gauss(-1, 1)
        data.append((x1, x2, 0))
    for _ in range(50):
        x1 = rng.gauss(2, 1)
        x2 = rng.gauss(1, 1)
        data.append((x1, x2, 1))
    return data


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def logistic_loss(weights, data):
    w0, w1, w2 = weights
    total = 0.0
    for x1, x2, y in data:
        z = w0 + w1 * x1 + w2 * x2
        total += max(z, 0) - z * y + math.log(1 + math.exp(-abs(z)))
    return total / len(data)


data = make_data()

lr = 0.001
epochs = 50

w = [0.0, 0.0, 0.0]
for epoch in range(epochs):
    gw = [0.0, 0.0, 0.0]
    for x1, x2, y in data:
        z = w[0] + w[1] * x1 + w[2] * x2
        p = sigmoid(z)
        err = p - y
        gw[0] += err
        gw[1] += err * x1
        gw[2] += err * x2
    n = len(data)
    w[0] -= lr * gw[0] / n
    w[1] -= lr * gw[1] / n
    w[2] -= lr * gw[2] / n

final_loss = logistic_loss(w, data)
print("weights:%s,%s,%s" % (w[0], w[1], w[2]))
print("final_loss:%s" % final_loss)
