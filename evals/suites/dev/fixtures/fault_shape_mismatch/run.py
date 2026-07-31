from model import init_weights, forward

weights = init_weights(4, 3, 2, seed=42)
x = [1.0, 2.0, 3.0, 4.0]
output = forward(weights, x)
print("output:%s" % ",".join(str(v) for v in output))
