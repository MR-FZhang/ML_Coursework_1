# Coursework 1: how the solution works

These notes sit outside the repo (`Neural_Networks_fz823/`), so they are never submitted.
The code itself is in `part1_nn_lib.py` and `part2_house_value_regression.py`.
Each `START OF YOUR CODE` block has short comments that point back to the sections below.
For a list of every change from the original template, see `CHANGES.md`.

## 0. What the spec requires (and how we comply)

| Spec rule | What we did |
|---|---|
| Part 1: NumPy only, no autodiff libraries | Only `numpy` is used in `part1_nn_lib.py` |
| No Python loops over data in `forward`/`backward` | Every layer uses whole-matrix operations. The only loops are over *layers* in `MultiLayerNetwork` |
| `"identity"` = linear passthrough with no activation layer | `MultiLayerNetwork` adds no layer for `"identity"` |
| Don't rename classes or functions. New args need defaults | `Regressor(x, nb_epoch=150, hidden_layers=(64, 64), learning_rate=1e-3, batch_size=64)` |
| Part 2 must use PyTorch | `torch.nn` model, `torch.optim.Adam` |
| Commit `part2_model.pickle`, saved by `save_regressor` | Created by running `main()` (see section 4) |
| Library versions must match LabTS. CPU only | `Coursework_1/.venv` is pinned to `requirements.txt`. Nothing touches a GPU |
| Don't commit environments or caches | The venv lives outside the repo, and `.gitignore` already covers `__pycache__` |

---

## 1. Part 1: the three jobs of every layer

Each layer does three things:

1. **`forward(x)`** computes the output and **caches** whatever `backward` will need.
2. **`backward(grad_z)`** receives `grad_z = dL/d(output)` and returns `dL/d(input)`.
   A layer with parameters also stores `dL/dW` and `dL/db`.
3. **`update_params(lr)`** takes one gradient-descent step: `W ← W − lr · dL/dW`.

Backprop is the chain rule applied one layer at a time.
Each layer only has to know its own local derivative and multiply it by the gradient arriving from above.

### Sigmoid: σ(x) = 1 / (1 + e^(−x))

- The derivative is `σ'(x) = σ(x)(1 − σ(x))`, so we **cache the output** σ(x), not the input.
- Backward: `grad_z * s * (1 - s)`. This is an element-wise product (`*`), not a matrix product.
  Each output depends only on its own input, so the Jacobian is diagonal and multiplying by it is element-wise.

### ReLU: max(0, x)

- The derivative is 1 where x > 0 and 0 elsewhere. We **cache the input** so we know which units were active.
- Backward: `grad_z * (x > 0)`. The boolean mask zeroes the gradient for inactive units.

### Linear layer: Z = XW + b

Shapes: `X (batch, n_in)`, `W (n_in, n_out)`, `b (n_out,)`, `Z (batch, n_out)`.

| Gradient | Formula | Shape check |
|---|---|---|
| `dL/dW` | `X.T @ grad_z` | (n_in, batch) @ (batch, n_out) = (n_in, n_out) ✓ same as W |
| `dL/db` | `grad_z.sum(axis=0)` | (n_out,) ✓ same as b |
| `dL/dX` | `grad_z @ W.T` | (batch, n_out) @ (n_out, n_in) = (batch, n_in) ✓ same as X |

**Shape-matching trick:** a gradient always has the same shape as the thing it differentiates.
If you forget a formula, find the arrangement of `X`, `W` and `grad_z` (transposed as needed) whose shapes multiply out correctly.
There is usually only one.

Why is `dL/db` a sum? The same `b` is added to every row of the batch.
So `b` affects the loss through every row, and the chain rule adds those contributions together.

Why do we cache `X`? `dL/dW` needs it. This is why `forward` must run before `backward`.

**Init:** `W` uses the provided `xavier_init`, which draws from U(±gain·√(6/(n_in+n_out))).
That keeps the signal's variance roughly constant from layer to layer. `b` starts at zero.

### MultiLayerNetwork

- **Constructor:** for each `(n_out, activation)` pair, add a `LinearLayer(n_in, n_out)` and then the activation layer (if any).
  This layer's `n_out` becomes the next layer's `n_in`.
  For example, `MultiLayerNetwork(4, [16, 3], ["relu", "identity"])` gives `[Linear(4,16), Relu, Linear(16,3)]`.
- **forward:** pass `x` through the layers in order.
- **backward:** pass `grad_z` through the layers in **reverse** order. This is the "back" in backpropagation.
- **update_params:** call it on every layer. Activation layers inherit the base class's no-op.

### Preprocessor: min-max scaling to [0, 1]

- `x' = (x − min) / (max − min)`, computed **per column** (`axis=0`), so each feature gets its own scale.
- `revert` is the exact inverse: `x = x' · (max − min) + min`.
- Fit it on the **training data only**, then apply the same statistics to validation and test data.
  If min/max came from the test data, information about the test set would leak into training.
- A constant column would divide by zero. We use a range of 1 instead, so it maps to 0 and still reverts exactly.

### A quirk found while testing

The provided `MSELossLayer._mse` averages over *all* elements, but `_mse_grad` divides only by the batch size.
With k outputs the "gradient" is therefore k× the true gradient of the reported loss.
With a single output they agree. This is provided code and harmless (it just scales the effective learning rate), so it is left alone.
The finite-difference check (`check_part1.py`) uses a 1-output network for this reason.

### Verify it yourself

```bash
cd Coursework_1
.venv/bin/python check_part1.py   # gradient check vs finite differences + preprocessor round trip
cd Neural_Networks_fz823 && ../.venv/bin/python part1_nn_lib.py   # iris demo, ~93% val accuracy
```

The **gradient check** nudges each weight by ±ε and measures the loss change: `(L(w+ε) − L(w−ε)) / 2ε`.
If that matches what `backward` computed, the backprop is correct.
This is the standard way to debug any hand-written backward pass.

### Rebuild it from memory (recommended by spec section 4)

Cover the code and try to write each piece from these prompts:
1. What does Sigmoid cache, and why the output and not the input?
2. Write all three Linear-layer gradients using only the shape-matching trick.
3. Why does `backward` in the network iterate in reverse?
4. Why is `dL/db` a sum over the batch?

---

## 2. Part 2: the full ML pipeline in PyTorch

```
raw DataFrame → _preprocessor → tensors in [0,1] → MLP → scaled prediction → _revert_preprocessor → $
```

**Preprocessing (provided).** `_preprocessor(training=True)` learns its statistics: the medians used to fill missing `total_bedrooms`, the min/max for scaling, and the one-hot categories for `ocean_proximity`.
With `training=False` it reuses those stored statistics, which is why `predict` passes `training=False`.
The result has 13 input features: 8 numeric plus 5 one-hot.
The target is min-max scaled too. Predicting values in [0, 1] instead of [15k, 500k] keeps the gradients a sensible size.

**`__init__`.** It builds `nn.Sequential(Linear(13,64), ReLU, Linear(64,64), ReLU, Linear(64,1))`.
The last layer has no activation, because a regression output must be free to take any value.
(A sigmoid would cap it, and a ReLU could get stuck at 0.)

**`fit`.** This is the standard PyTorch training loop. Each epoch:
1. `torch.randperm` shuffles the rows, then we take mini-batches of 64.
2. `optimiser.zero_grad()`: PyTorch *adds* new gradients onto old ones, so they must be cleared every step.
3. `loss = MSELoss(model(xb), yb)`: forward pass plus loss.
4. `loss.backward()`: autograd does what you hand-wrote in Part 1 and fills `.grad` on every weight.
5. `optimiser.step()`: the update. Adam is SGD with a per-weight adaptive step size and momentum, and needs far less learning-rate tuning.

Mini-batches give many cheap, slightly noisy updates per epoch.
Full-batch training gives one exact but expensive update.
Mini-batches usually reach a good solution much faster.

**`predict`.**
- `model.eval()` switches layers like dropout and batch-norm to inference behaviour. We use neither, but it's a good habit.
- `torch.no_grad()` skips building the autograd graph, which saves time and memory.
- `_revert_preprocessor` turns the scaled outputs back into dollars.

**`evaluate`.** RMSE = √(mean((y − ŷ)²)), in dollars, via sklearn's `root_mean_squared_error`.
sklearn also checks that the shapes of `y` and `ŷ` line up.

**`main`.** It holds out 20% as a validation set, trains on the other 80%, saves the model, and prints train and validation RMSE.
Validation RMSE is your estimate of the hidden LabTS test score.
If train RMSE is far below validation RMSE, the model is overfitting.

### Hyperparameter experiment

Validation RMSE on an 80/20 split, checked every 25 epochs up to 300:

| Config | ms/epoch | Best val RMSE | Behaviour |
|---|---|---|---|
| **(64, 64), lr 1e-3, batch 64** (chosen) | 55 | ~52.3k | plateaus after ~100–150 epochs, small train/val gap |
| (128, 128) | 63 | ~52.7k | train keeps falling, val rises: **overfitting** |
| (64, 64, 32) | 62 | ~53.3k | noisier, no better |
| (64, 64), lr 3e-3, batch 256 | 19 | ~52.5k | just as good and 3× faster per epoch |

Lesson: a bigger network lowered *train* error but not *validation* error. The extra capacity went into memorising the training set.
`nb_epoch` defaults to 150, where validation RMSE stops improving. (The skeleton had 1000. Changing a default is allowed, and it keeps any LabTS test that uses the defaults fast.)

**Final saved model:** train RMSE ≈ $49.1k, validation RMSE ≈ $53.1k, about $50.0k on the full `housing.csv` (the public LabTS performance data).
Always predicting the mean scores $115.7k, so the model is far better than the baseline.

### Pickle gotchas (spec notes 5–7)

- Pickle stores an object's class by *module path*. Running `python part2_house_value_regression.py` directly would record it as `__main__.Regressor`, and LabTS (which *imports* the module) might fail to find it.
  The `if __name__ == "__main__":` block therefore imports the module and calls its `main()`, so the class is recorded as `part2_house_value_regression.Regressor`.
- The pickle also contains pandas and NumPy objects (the preprocessing statistics), so it must be created with the pinned versions. Always use `Coursework_1/.venv`.
- Everything is trained on the CPU, so LabTS (which has no GPU) can load it.

---

## 3. One risk to check on LabTS early

The bias `_b` is stored with shape `(n_out,)`, the same convention as PyTorch's `nn.Linear`.
`_grad_b_current` has the same shape.
Neither the spec nor the docstrings fix this shape, and `(1, n_out)` is the other common choice.
If a LabTS unit test fails on the bias shape, change both lines to `np.zeros((1, n_out))` and `grad_z.sum(axis=0, keepdims=True)`.

## 4. Submitting

```bash
cd Coursework_1/Neural_Networks_fz823
../.venv/bin/python part2_house_value_regression.py   # trains, saves part2_model.pickle, prints RMSEs
git add part1_nn_lib.py part2_house_value_regression.py part2_model.pickle
git commit -m "Complete part 1 and part 2"
git push
```

Then press **Request Test** on LabTS for that commit, read the PDF report, and submit the hash through LabTS and Scientia.
