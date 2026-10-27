"""Self-check for part1_nn_lib.py: compares backprop gradients against
finite-difference estimates, and checks the Preprocessor round trip.

Run from Coursework_1/:  .venv/bin/python check_part1.py
"""
import sys
import numpy as np

sys.path.insert(0, "Neural_Networks_fz823")
from part1_nn_lib import MultiLayerNetwork, MSELossLayer, Preprocessor, LinearLayer

np.random.seed(0)
x = np.random.randn(5, 4)
# One output: the provided _mse averages over all elements but _mse_grad only
# divides by batch size, so they agree exactly only when there is 1 output.
y = np.random.randn(5, 1)
net = MultiLayerNetwork(4, [6, 3, 1], ["relu", "sigmoid", "identity"])
loss_layer = MSELossLayer()


def loss():
    return loss_layer(net(x), y)


# Analytic gradients from backprop
loss()
grad_x = net.backward(loss_layer.backward())

# Numerical gradients: (L(p + eps) - L(p - eps)) / 2eps for each parameter p
eps = 1e-6
for layer in [l for l in net._layers if isinstance(l, LinearLayer)]:
    for param, analytic in [(layer._W, layer._grad_W_current),
                            (layer._b, layer._grad_b_current)]:
        numeric = np.zeros_like(param)
        for idx in np.ndindex(param.shape):
            old = param[idx]
            param[idx] = old + eps; up = loss()
            param[idx] = old - eps; down = loss()
            param[idx] = old
            numeric[idx] = (up - down) / (2 * eps)
        assert np.allclose(analytic, numeric, atol=1e-7), (analytic, numeric)

numeric_x = np.zeros_like(x)
for idx in np.ndindex(x.shape):
    old = x[idx]
    x[idx] = old + eps; up = loss()
    x[idx] = old - eps; down = loss()
    x[idx] = old
    numeric_x[idx] = (up - down) / (2 * eps)
assert np.allclose(grad_x, numeric_x, atol=1e-7)
assert grad_x.shape == x.shape
print("gradient check passed (W, b and inputs match finite differences)")

# One gradient-descent step should lower the loss
before = loss()
net.backward(loss_layer.backward())
net.update_params(0.01)
assert loss() < before
print("update_params lowers the loss")

# Preprocessor: range [0, 1], exact round trip, input untouched
data = np.random.rand(20, 3) * 50 - 10
data[:, 2] = 7.0  # constant column must not produce NaN
copy = data.copy()
prep = Preprocessor(data)
normed = prep.apply(data)
assert np.array_equal(data, copy)
assert normed.min() >= 0 and normed.max() <= 1 and not np.isnan(normed).any()
assert np.allclose(prep.revert(normed), data)
print("Preprocessor checks passed")
