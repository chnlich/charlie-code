from model import gradient

x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.0, 4.0, 6.0, 8.0, 10.0]

w = 0.0
lr = 0.01
for epoch in range(200):
    g = gradient(w, x, y)
    w -= lr * g

final_loss = sum((w * xi - yi) ** 2 for xi, yi in zip(x, y)) / len(x)
print("weights:%s" % w)
print("final_loss:%s" % final_loss)
